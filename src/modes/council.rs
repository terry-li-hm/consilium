//! Full council deliberation mode: blind phase, debate rounds, judge synthesis, CollabEval.

use crate::api::{query_judge, query_model, query_model_async, run_parallel};
use crate::config::{
    detect_consensus, detect_social_context, is_error_response, parse_confidence,
    resolved_judge_model, sanitize_speaker_content, CostTracker, Message, ModelEntry,
    resolved_critique_model, SessionResult, EXTRACTION_MODEL,
};
use crate::prompts::{
    council_blind_system, council_challenger_addition, council_debate_system,
    council_first_speaker_system, council_first_speaker_with_blind, council_social_constraint,
    council_xpol_system, domain_context, EXTRACTION_PROMPT,
};
use crate::session::Output;
use chrono::Local;
use rand::seq::SliceRandom;
use regex::Regex;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;

/// Extract self-reported confidence score from a response (e.g. "**Confidence: 7/10**").
/// Searches the last 10 lines for the pattern. Returns None if not found.
fn extract_confidence_score(text: &str) -> Option<u8> {
    for line in text.lines().rev().take(10) {
        let lower = line.to_lowercase();
        if lower.contains("confidence:") {
            if let Some(slash_pos) = lower.rfind("/10") {
                let before = &lower[..slash_pos];
                let digits: String = before
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                if let Ok(n) = digits.parse::<u8>() {
                    if n <= 10 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

static ARRAY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\[.*?\]").expect("valid regex"));
static BULLET_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:[-*]|\d+[.)])\s*").expect("valid regex"));

static RE_RECOMMENDATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)## Recommendation[^\n]*\n(.*?)(?:\n## |\z)").expect("valid regex")
});
static RE_DO_NOW: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)### Do Now[^\n]*\n(.*?)(?:\n### |\z)").expect("valid regex"));
static RE_DO_NOW_BOLD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\*\*\d+\.\s*(.+?)\*\*").expect("valid regex"));
static RE_CONSIDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)### Consider Later[^\n]*\n(.*?)(?:\n### |\z)").expect("valid regex")
});
static RE_SKIP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)### Skip[^\n]*\n(.*?)(?:\n### |\n---|\z)").expect("valid regex")
});
static RE_BULLET_BOLD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[-*]\s+\*\*(.+?)\*\*").expect("valid regex"));

static RE_SYNTHESIS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)## Synthesis[^\n]*\n(.*?)(?:\n## |\z)").expect("valid regex")
});

#[derive(Debug, Clone, Default)]
struct RecommendationItems {
    do_now: Vec<String>,
    consider_later: Vec<String>,
    skip: Vec<String>,
}

fn append_optional_context(
    mut prompt: String,
    domain_hint: Option<&str>,
    social_mode: bool,
    persona: Option<&str>,
) -> String {
    if let Some(ctx) = domain_hint {
        prompt.push_str(&format!(
            "\n\nDOMAIN CONTEXT: {ctx}\n\nApply this regulatory domain context to your analysis."
        ));
    }

    if social_mode {
        prompt.push_str(&council_social_constraint());
    }

    if let Some(p) = persona {
        prompt.push_str(&format!(
            "\n\nIMPORTANT CONTEXT about the person asking:\n{p}\n\nFactor this into your advice — don't just give strategically optimal answers, consider what fits THIS person."
        ));
    }

    prompt
}

fn extract_section(text: &str, header: &str) -> Option<String> {
    let pattern = format!(r"(?s)## {}[^\n]*\n(.*?)(?=\n## |\z)", regex::escape(header));
    let re = Regex::new(&pattern).ok()?;
    re.captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
}

fn parse_recommendation_bullets(section: &str) -> Vec<String> {
    let bold_items: Vec<String> = RE_BULLET_BOLD
        .captures_iter(section)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .collect();

    if !bold_items.is_empty() {
        return bold_items;
    }

    section
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if (trimmed.starts_with('-') || trimmed.starts_with('*')) && trimmed.len() > 5 {
                Some(
                    trimmed
                        .trim_start_matches(&['-', '*'][..])
                        .trim()
                        .to_string(),
                )
            } else {
                None
            }
        })
        .collect()
}

fn parse_recommendation_items(judge_response: &str) -> RecommendationItems {
    let mut out = RecommendationItems::default();

    let Some(rec_match) = RE_RECOMMENDATION.captures(judge_response) else {
        return out;
    };
    let rec_text = rec_match
        .get(1)
        .map(|m| m.as_str())
        .unwrap_or_default()
        .to_string();

    if let Some(do_now) = RE_DO_NOW.captures(&rec_text).and_then(|c| c.get(1)) {
        out.do_now = RE_DO_NOW_BOLD
            .captures_iter(do_now.as_str())
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().trim().trim_end_matches('.').to_string())
            .filter(|s| s.len() > 5)
            .collect();
    }

    if let Some(consider) = RE_CONSIDER.captures(&rec_text).and_then(|c| c.get(1)) {
        out.consider_later = parse_recommendation_bullets(consider.as_str());
    }

    if let Some(skip) = RE_SKIP.captures(&rec_text).and_then(|c| c.get(1)) {
        out.skip = parse_recommendation_bullets(skip.as_str());
    }

    out
}

fn extract_for_llm(judge_response: &str) -> String {
    let mut parts = Vec::new();
    for header in [
        "Competing Hypotheses",
        "Points of Disagreement",
        "Synthesis",
    ] {
        if let Some(content) = extract_section(judge_response, header) {
            parts.push(format!("## {header}\n{content}"));
        }
    }

    if parts.is_empty() {
        judge_response.to_string()
    } else {
        parts.join("\n\n")
    }
}

pub async fn extract_structured_summary(
    judge_response: &str,
    question: &str,
    models_used: &[String],
    rounds: usize,
    duration: f64,
    cost: f64,
    client: &Client,
    api_key: Option<&str>,
    cost_tracker: Option<&CostTracker>,
) -> Value {
    let code_items = parse_recommendation_items(judge_response);

    let mut extracted = Value::Null;

    if let Some(key) = api_key {
        let focused_input = extract_for_llm(judge_response);
        let messages = vec![
            Message::system(EXTRACTION_PROMPT),
            Message::user(focused_input),
        ];
        let mut raw = query_model(
            client,
            key,
            EXTRACTION_MODEL,
            &messages,
            600,
            30.0,
            2,
            cost_tracker,
        )
        .await
        .trim()
        .to_string();

        if raw.starts_with("```") {
            raw = raw
                .split_once('\n')
                .map(|(_, rest)| rest.to_string())
                .unwrap_or_else(|| raw.trim_start_matches("```").to_string());
        }
        if raw.ends_with("```") {
            raw = raw[..raw.len().saturating_sub(3)].to_string();
        }

        if let Ok(v) = serde_json::from_str::<Value>(raw.trim()) {
            extracted = v;
        }
    }

    if !extracted.is_object() {
        let synthesis = RE_SYNTHESIS
            .captures(judge_response)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();

        let fallback_source = if synthesis.is_empty() {
            judge_response
        } else {
            &synthesis
        };

        let decision = fallback_source
            .lines()
            .map(str::trim)
            .find(|line| {
                !line.is_empty()
                    && line.len() > 30
                    && !line.starts_with('#')
                    && !line.starts_with('*')
                    && !line.starts_with('-')
                    && !line.starts_with("**Do")
                    && !line.starts_with("**Consider")
                    && !line.starts_with("**Skip")
                    && !line.to_lowercase().starts_with("do now")
                    && !line.to_lowercase().starts_with("consider")
                    && !line.to_lowercase().starts_with("recommendation")
            })
            .unwrap_or("See transcript for details")
            .chars()
            .take(500)
            .collect::<String>();

        extracted = json!({
            "decision": decision,
            "confidence": "medium",
            "winning_hypothesis": "",
            "reasoning_summary": if synthesis.is_empty() { "See transcript for details" } else { synthesis.as_str() },
            "dissents": []
        });
    }

    if let Some(obj) = extracted.as_object_mut() {
        if !code_items.do_now.is_empty() {
            obj.insert("do_now".to_string(), json!(code_items.do_now));
        }
        if !code_items.consider_later.is_empty() {
            obj.insert(
                "consider_later".to_string(),
                json!(code_items.consider_later),
            );
        }
        if !code_items.skip.is_empty() {
            obj.insert("skip".to_string(), json!(code_items.skip));
        }

        let has_action_items = obj
            .get("action_items")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

        if !has_action_items && !code_items.do_now.is_empty() {
            obj.insert(
                "action_items".to_string(),
                Value::Array(
                    code_items
                        .do_now
                        .iter()
                        .map(|item| json!({"action": item, "priority": "high"}))
                        .collect(),
                ),
            );
        }

        obj.insert("schema_version".to_string(), json!("1.0"));
        obj.insert("question".to_string(), json!(question));
        obj.insert(
            "meta".to_string(),
            json!({
                "timestamp": Local::now().to_rfc3339(),
                "models_used": models_used,
                "rounds": rounds,
                "duration_seconds": duration,
                "estimated_cost_usd": cost,
            }),
        );
    }

    extracted
}

pub async fn decompose_question(
    question: &str,
    api_key: &str,
    output: &mut dyn Output,
    cost_tracker: &CostTracker,
) -> Vec<String> {
    let client = Client::new();
    let judge_model = resolved_judge_model();

    let judge_name = judge_model
        .split('/')
        .next_back()
        .unwrap_or(judge_model.as_str());
    let _ = output.write_str(&format!("### Question Decomposition ({judge_name})\n"));

    let messages = vec![
        Message::system("Decompose the user's complex question into 2-3 focused, non-overlapping sub-questions.\n\nOutput STRICT JSON only: an array of strings.\nNo markdown, no prose, no explanation.\nEach sub-question should be actionable for independent analysis."),
        Message::user(format!("Question:\n{question}")),
    ];

    let response = query_model(
        &client,
        api_key,
        judge_model.as_str(),
        &messages,
        300,
        120.0,
        2,
        Some(cost_tracker),
    )
    .await;

    let _ = output.write_str("\n");

    let candidate = ARRAY_RE
        .find(&response)
        .map(|m| m.as_str())
        .unwrap_or(response.as_str());

    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(candidate) {
        let sub_questions: Vec<String> = items
            .into_iter()
            .filter_map(|v| v.as_str().map(str::trim).map(str::to_string))
            .filter(|s| !s.is_empty())
            .take(3)
            .collect();

        if sub_questions.len() >= 2 {
            return sub_questions;
        }
    }

    let fallback: Vec<String> = response
        .lines()
        .map(|line| BULLET_RE.replace(line, "").trim().to_string())
        .filter(|line| !line.is_empty())
        .take(3)
        .collect();

    if fallback.len() >= 2 {
        fallback
    } else {
        vec![question.to_string()]
    }
}

async fn compress_round_context(
    round_responses: &[(String, String)],
    question: &str,
    client: &Client,
    api_key: &str,
    cost_tracker: &CostTracker,
) -> String {
    let mut round_summary = String::new();
    for (name, response) in round_responses {
        round_summary.push_str(&format!("**{name}**: {response}\n\n"));
    }

    let prompt = format!(
        "Summarize this debate round. For each speaker, capture:\n1. Core position (1 sentence)\n2. Key new argument or rebuttal (1 sentence)\n3. Whether they agree/disagree with majority\n\nKeep exact quotes only if they contain specific data points or citations.\n\nQuestion: {question}\n\nRound responses:\n{round_summary}"
    );

    let messages = vec![Message::user(prompt)];

    let result = query_model(
        client,
        api_key,
        crate::config::COMPRESSION_MODEL,
        &messages,
        500,
        30.0,
        2,
        Some(cost_tracker),
    )
    .await;

    if result.starts_with("[Error:") {
        // Fallback: original responses concatenated
        round_responses
            .iter()
            .map(|(name, response)| format!("{name}: {response}"))
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        result
    }
}

async fn run_blind_phase_parallel(
    question: &str,
    council_config: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    timeout: f64,
    output: &mut dyn Output,
    persona: Option<&str>,
    domain_hint: Option<&str>,
    sub_questions: Option<&[String]>,
    cost_tracker: &CostTracker,
) -> Vec<(String, String, String)> {
    let mut blind_system = council_blind_system();
    blind_system = append_optional_context(blind_system, domain_hint, false, persona);

    let _ = output.begin_phase("BLIND PHASE");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\nBLIND PHASE (independent claims)\n");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\n\n(querying all models in parallel...)\n");

    let mut user_content = format!("Question:\n\n{question}");
    if let Some(sq) = sub_questions {
        if sq.len() > 1 {
            let numbered = sq
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {s}", i + 1))
                .collect::<Vec<_>>()
                .join("\n");
            user_content.push_str(&format!("\n\nSub-questions to address:\n{numbered}"));
        }
    }

    let messages = vec![Message::system(blind_system), Message::user(user_content)];

    let result = run_parallel(
        council_config,
        &messages,
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        openai_api_key,
        xai_api_key,
        1500,
        timeout,
        Some(cost_tracker),
        None,
    )
    .await;

    for (_, model_name, claims) in &result {
        let _ = output.write_str(&format!("\n### {model_name}\n{claims}\n\n"));
        let _ = output.end_participant(model_name, claims, 0);
    }

    let _ = output.write_str("\n");

    result
}

async fn run_xpol_phase_parallel(
    question: &str,
    blind_claims: &[(String, String, String)],
    council_config: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    timeout: f64,
    output: &mut dyn Output,
    persona: Option<&str>,
    domain_hint: Option<&str>,
    display_names: &HashMap<String, String>,
    cost_tracker: &CostTracker,
) -> Vec<(String, String, String)> {
    let mut xpol_system = council_xpol_system();
    xpol_system = append_optional_context(xpol_system, domain_hint, false, persona);

    let _ = output.begin_phase("CROSS-POLLINATION PHASE");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\nCROSS-POLLINATION PHASE (extend, don't argue)\n");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\n\n(querying all models in parallel...)\n");

    let blind_summary = blind_claims
        .iter()
        .filter(|(_, _, claims)| !is_error_response(claims))
        .map(|(name, _, claims)| {
            let dname = display_names
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone());
            format!("**{dname}**: {claims}")
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let messages = vec![
        Message::system(xpol_system),
        Message::user(format!(
            "Question:\n\n{question}\n\n---\n\nBLIND CLAIMS from all speakers:\n\n{blind_summary}"
        )),
    ];

    let result = run_parallel(
        council_config,
        &messages,
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        openai_api_key,
        xai_api_key,
        1500,
        timeout,
        Some(cost_tracker),
        None,
    )
    .await;

    for (_, model_name, claims) in &result {
        let _ = output.write_str(&format!("\n### {model_name}\n{claims}\n\n"));
        let _ = output.end_participant(model_name, claims, 0);
    }

    let _ = output.write_str("\n");

    result
}

pub async fn run_followup_discussion(
    question: &str,
    topic: &str,
    council_config: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    timeout: f64,
    domain_hint: Option<&str>,
    social_mode: bool,
    persona: Option<&str>,
    output: &mut dyn Output,
    cost_tracker: &CostTracker,
) -> String {
    let client = Client::new();
    let followup_models = &council_config[..council_config.len().min(2)];
    let mut transcript_parts = vec![format!("### Followup Discussion: {topic}\n")];

    let _ = output.begin_phase("FOLLOWUP DISCUSSION");
    let _ = output.write_str("\n");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str(&format!("\nFOLLOWUP: {topic}\n"));
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\n\n");

    let mut followup_system = [
        "You are participating in a FOCUSED FOLLOWUP discussion on a specific topic.",
        "",
        "The main council has concluded, and we're now drilling down into:",
        &format!("TOPIC: {topic}"),
        "",
        "Keep your response focused on this specific topic. Don't rehash the full council deliberation.",
        "Be concise and practical.",
    ]
    .join("\n");

    followup_system = append_optional_context(followup_system, domain_hint, social_mode, persona);

    for &(name, model, fallback) in followup_models {
        let participant_t0 = Instant::now();
        let _ = output.begin_participant(name);
        let _ = output.write_str(&format!("### {name}\n"));

        let messages = vec![
            Message::system(followup_system.clone()),
            Message::user(format!(
                "Original Question:\n\n{question}\n\nFocus your response on: {topic}"
            )),
        ];

        let (_, model_name, response) = query_model_async(
            &client,
            api_key,
            model,
            &messages,
            name,
            fallback,
            google_api_key,
            zhipu_api_key,
            moonshot_api_key,
            openai_api_key,
            xai_api_key,
            1200,
            timeout,
            2,
            Some(cost_tracker),
        )
        .await;

        let _ = output.write_str(&format!("{response}\n\n"));
        let _ =
            output.end_participant(name, &response, participant_t0.elapsed().as_millis() as u64);

        transcript_parts.push(format!("### {model_name}\n{response}\n"));
    }

    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\nFOLLOWUP COMPLETE\n");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\n\n");

    transcript_parts.join("\n\n")
}

pub async fn run_council(
    question: &str,
    council_config: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    rounds: usize,
    timeout: f64,
    output: &mut dyn Output,
    anonymous: bool,
    blind: bool,
    context: Option<&str>,
    persona: Option<&str>,
    domain: Option<&str>,
    challenger_idx: Option<usize>,
    format: &str,
    collabeval: bool,
    judge: bool,
    sub_questions: Option<Vec<String>>,
    cross_pollinate: bool,
    followup: bool,
    thorough: bool,
) -> SessionResult {
    let start = Instant::now();
    let client = Client::new();
    let cost_tracker = CostTracker::new();
    let judge_model = resolved_judge_model();

    let domain_hint = domain.and_then(domain_context);
    let social_mode = detect_social_context(question);

    let mut blind_claims = Vec::new();
    let mut failed_models: Vec<String> = Vec::new();
    let council_names: Vec<&str> = council_config.iter().map(|(name, _, _)| *name).collect();

    if blind {
        blind_claims = run_blind_phase_parallel(
            question,
            council_config,
            api_key,
            google_api_key,
            zhipu_api_key,
            moonshot_api_key,
            openai_api_key,
            xai_api_key,
            timeout,
            output,
            persona,
            domain_hint,
            sub_questions.as_deref(),
            &cost_tracker,
        )
        .await;

        for (_, model_name, claims) in &blind_claims {
            if is_error_response(claims) {
                failed_models.push(format!("{model_name} (blind): {claims}"));
            }
        }
    }

    let display_names: HashMap<String, String> = if anonymous {
        council_config
            .iter()
            .enumerate()
            .map(|(i, (name, _, _))| (name.to_string(), format!("Speaker {}", i + 1)))
            .collect()
    } else {
        council_config
            .iter()
            .map(|(name, _, _)| (name.to_string(), name.to_string()))
            .collect()
    };

    let _ = output.write_str(&format!("Council members: {:?}\n", council_names));
    if anonymous {
        let _ = output.write_str("(Models see each other as Speaker 1, 2, etc. to prevent bias)\n");
    }
    let _ = output.write_str(&format!("Rounds: {rounds}\n"));
    if let Some(d) = domain {
        let _ = output.write_str(&format!("Domain context: {d}\n"));
    }
    if social_mode {
        let _ = output.write_str("Social context detected: applying conversational constraint\n");
    }
    let question_preview = if question.chars().count() > 100 {
        format!("{}...", &question.chars().take(97).collect::<String>())
    } else {
        question.to_string()
    };
    let _ = output.write_str(&format!("Question: {question_preview}\n\n"));
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\nCOUNCIL DELIBERATION\n");
    let _ = output.write_str(&"=".repeat(60));
    let _ = output.write_str("\n\n");

    let mut conversation: Vec<(String, String)> = Vec::new();
    let mut compressed_summaries: Vec<String> = Vec::new();
    let mut output_parts: Vec<String> = Vec::new();
    let mut confidences: HashMap<String, Vec<u8>> = HashMap::new();
    let mut current_round = 0usize;

    for (_, model_name, claims) in &blind_claims {
        output_parts.push(format!("### {model_name} (blind)\n{claims}"));
    }

    let blind_context = if blind_claims.is_empty() {
        String::new()
    } else {
        let mut valid_blind_count = 0usize;
        let mut lines = Vec::new();

        for (name, _, claims) in &blind_claims {
            let dname = display_names
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.clone());
            if is_error_response(claims) {
                lines.push(format!("**{dname}**: *(unavailable for this phase)*"));
            } else {
                lines.push(format!("**{dname}**: {}", sanitize_speaker_content(claims)));
                valid_blind_count += 1;
            }
        }

        if valid_blind_count < blind_claims.len() {
            lines.push(format!(
                "\n*Note: {valid_blind_count} of {} models responded in blind phase.*",
                blind_claims.len()
            ));
        }

        lines.join("\n\n")
    };

    let mut xpol_context = String::new();

    if cross_pollinate && !blind_claims.is_empty() {
        let xpol_claims = run_xpol_phase_parallel(
            question,
            &blind_claims,
            council_config,
            api_key,
            google_api_key,
            zhipu_api_key,
            moonshot_api_key,
            openai_api_key,
            xai_api_key,
            timeout,
            output,
            persona,
            domain_hint,
            &display_names,
            &cost_tracker,
        )
        .await;

        for (_, model_name, claims) in &xpol_claims {
            if is_error_response(claims) {
                failed_models.push(format!("{model_name} (xpol): {claims}"));
            }
            output_parts.push(format!("### {model_name} (cross-pollination)\n{claims}"));
        }

        let xpol_lines: Vec<String> = xpol_claims
            .iter()
            .filter(|(_, _, claims)| !is_error_response(claims))
            .map(|(name, _, claims)| {
                let dname = display_names
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| name.clone());
                format!(
                    "**{dname}** (cross-pollination): {}",
                    sanitize_speaker_content(claims)
                )
            })
            .collect();

        if !xpol_lines.is_empty() {
            xpol_context = xpol_lines.join("\n\n");
        }
    }

    let rounds = rounds.max(1);

    for round_num in 0..rounds {
        let _ = output.begin_phase(&format!("DEBATE ROUND {}", round_num + 1));
        current_round = round_num + 1;
        let mut round_speakers = Vec::new();

        let current_challenger = challenger_idx
            .map(|idx| (idx + round_num) % council_config.len())
            .unwrap_or(round_num % council_config.len());

        for (idx, &(name, model, fallback)) in council_config.iter().enumerate() {
            let dname = display_names
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string());

            let mut system_prompt = if idx == 0 && round_num == 0 {
                if !blind_context.is_empty() {
                    council_first_speaker_with_blind(&dname, round_num + 1)
                } else {
                    council_first_speaker_system(&dname, round_num + 1)
                }
            } else {
                let previous = if !round_speakers.is_empty() {
                    round_speakers.join(", ")
                } else {
                    council_config
                        .iter()
                        .map(|(n, _, _)| {
                            display_names
                                .get(*n)
                                .cloned()
                                .unwrap_or_else(|| n.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                council_debate_system(&dname, round_num + 1, &previous)
            };

            system_prompt =
                append_optional_context(system_prompt, domain_hint, social_mode, persona);

            if idx == current_challenger {
                system_prompt.push_str(&council_challenger_addition());
            }

            let mut user_content = format!("Question for the council:\n\n{question}");
            if !blind_context.is_empty() {
                user_content.push_str(&format!(
                    "\n\n---\n\nBLIND CLAIMS (independent initial positions):\n\n{blind_context}"
                ));
            }
            if !xpol_context.is_empty() {
                user_content.push_str(&format!(
                    "\n\n---\n\nCROSS-POLLINATION (gap analysis after reading blind claims):\n\n{xpol_context}"
                ));
            }

            let mut messages = vec![Message::system(system_prompt), Message::user(user_content)];

            if !thorough && !compressed_summaries.is_empty() {
                for (i, summary) in compressed_summaries.iter().enumerate() {
                    messages.push(Message::user(format!(
                        "Summary of Round {}:\n{}",
                        i + 1,
                        summary
                    )));
                }

                let current_round_start_idx = round_num * council_config.len();
                let mut own_msgs: Vec<String> = Vec::new();
                let mut other_msgs: Vec<String> = Vec::new();
                for (speaker, text) in conversation.iter().skip(current_round_start_idx) {
                    if is_error_response(text) {
                        continue;
                    }
                    let sanitized_text = sanitize_speaker_content(text);
                    if speaker == name {
                        own_msgs.push(sanitized_text);
                    } else {
                        let speaker_dname = display_names
                            .get(speaker)
                            .cloned()
                            .unwrap_or_else(|| speaker.clone());
                        other_msgs.push(format!("[{speaker_dname}]: {sanitized_text}"));
                    }
                }
                other_msgs.shuffle(&mut rand::thread_rng());
                for msg in own_msgs {
                    messages.push(Message::assistant(msg));
                }
                for msg in other_msgs {
                    messages.push(Message::user(msg));
                }
            } else {
                let mut own_msgs: Vec<String> = Vec::new();
                let mut other_msgs: Vec<String> = Vec::new();
                for (speaker, text) in &conversation {
                    if is_error_response(text) {
                        continue;
                    }
                    let sanitized_text = sanitize_speaker_content(text);
                    if speaker == name {
                        own_msgs.push(sanitized_text);
                    } else {
                        let speaker_dname = display_names
                            .get(speaker)
                            .cloned()
                            .unwrap_or_else(|| speaker.clone());
                        other_msgs.push(format!("[{speaker_dname}]: {sanitized_text}"));
                    }
                }
                other_msgs.shuffle(&mut rand::thread_rng());
                for msg in own_msgs {
                    messages.push(Message::assistant(msg));
                }
                for msg in other_msgs {
                    messages.push(Message::user(msg));
                }
            }

            let challenger_indicator = if idx == current_challenger {
                " (challenger)"
            } else {
                ""
            };

            let model_name = model.split('/').next_back().unwrap_or(model);
            let speaker_t0 = Instant::now();
            let _ = output.begin_participant(model_name);
            let _ = output.write_str(&format!("### {model_name}{challenger_indicator}\n"));

            let (speaker_name, model_name, response) = query_model_async(
                &client,
                api_key,
                model,
                &messages,
                name,
                fallback,
                google_api_key,
                zhipu_api_key,
                moonshot_api_key,
                openai_api_key,
                xai_api_key,
                1500,
                timeout,
                2,
                Some(&cost_tracker),
            )
            .await;

            let _ = output.write_str(&format!("{response}\n\n"));
            let _ = output.end_participant(
                &model_name,
                &response,
                speaker_t0.elapsed().as_millis() as u64,
            );

            if is_error_response(&response) {
                failed_models.push(format!("{model_name}: {response}"));
            }

            conversation.push((speaker_name.clone(), response.clone()));
            round_speakers.push(dname);

            if let Some(conf) = parse_confidence(&response) {
                confidences.entry(speaker_name).or_default().push(conf);
            }

            output_parts.push(format!(
                "### {model_name}{challenger_indicator}\n{response}"
            ));
        }

        if !thorough {
            let (converged, reason) =
                detect_consensus(&conversation, council_config, Some(current_challenger));
            if converged {
                let _ = output.write_str(&format!(
                    ">>> CONSENSUS DETECTED ({reason}) - proceeding to judge\n\n"
                ));
                break;
            }
        }

        if !thorough && rounds > 1 && round_num < rounds - 1 {
            let round_start = round_num * council_config.len();
            let round_responses = &conversation[round_start..];
            let _ = output.write_str(&format!(
                "(compressing round {} context...)\n",
                round_num + 1
            ));
            let summary =
                compress_round_context(round_responses, question, &client, api_key, &cost_tracker)
                    .await;
            compressed_summaries.push(summary);
        }
    }

    let mut confidence_line: Option<String> = None;
    if !confidences.is_empty() {
        let drift_parts: Vec<String> = confidences
            .iter()
            .filter_map(|(name, scores)| {
                if scores.is_empty() {
                    return None;
                }

                let model_name = council_config
                    .iter()
                    .find(|(n, _, _)| *n == name)
                    .map(|(_, model, _)| model.split('/').next_back().unwrap_or(model).to_string())
                    .unwrap_or_else(|| name.to_string());

                if scores.len() >= 2 {
                    Some(format!(
                        "{model_name} {}→{}",
                        scores[0],
                        scores[scores.len() - 1]
                    ))
                } else {
                    Some(format!("{model_name} {}/10", scores[0]))
                }
            })
            .collect();

        if !drift_parts.is_empty() {
            let line = format!("Confidence drift: {}", drift_parts.join(", "));
            confidence_line = Some(line.clone());
            output_parts.push(line);
        }
    }

    let valid_conversation: Vec<(String, String)> = conversation
        .iter()
        .filter(|(_, text)| !is_error_response(text))
        .cloned()
        .collect();

    let mut deliberation_text = valid_conversation
        .iter()
        .map(|(speaker, text)| {
            let dname = display_names
                .get(speaker)
                .cloned()
                .unwrap_or_else(|| speaker.clone());
            format!("**{dname}**: {}", sanitize_speaker_content(text))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    // Extract final confidence score per speaker (last response wins).
    // Only present in debate rounds — blind responses don't request scores.
    let mut final_confidence: HashMap<String, u8> = HashMap::new();
    for (speaker, text) in valid_conversation.iter().rev() {
        if !final_confidence.contains_key(speaker) {
            if let Some(score) = extract_confidence_score(text) {
                final_confidence.insert(speaker.clone(), score);
            }
        }
    }
    let confidence_summary: String = {
        let lines: Vec<String> = council_config
            .iter()
            .filter_map(|(name, _, _)| {
                let label = display_names
                    .get(*name)
                    .cloned()
                    .unwrap_or_else(|| name.to_string());
                final_confidence.get(*name).map(|s| format!("- {label}: {s}/10"))
            })
            .collect();
        if lines.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nFinal self-reported confidence scores (post-debate):\n{}\nHigh confidence + independent blind agreement = strong signal. A confidence drop after debate may indicate genuine persuasion or sycophancy — cross-check with POSITION CHANGE labels.",
                lines.join("\n")
            )
        }
    };

    let blind_convergence_note = if !blind_context.is_empty() {
        let valid_blind_count = blind_claims
            .iter()
            .filter(|(_, _, text)| !is_error_response(text))
            .count();
        format!(
            "\n\n[Blind phase: {valid_blind_count}/{} models responded independently before seeing peers]",
            blind_claims.len()
        )
    } else {
        String::new()
    };

    if valid_conversation.len() < conversation.len() {
        let failed_count = conversation.len() - valid_conversation.len();
        deliberation_text.push_str(&format!(
            "\n\n*Note: {failed_count} model response(s) were unavailable and excluded from this transcript.*"
        ));
    }

    let blind_section = if !blind_context.is_empty() {
        format!(
            "BLIND CLAIMS (independent positions before debate):\n\n{blind_context}{blind_convergence_note}\n\n---\n\n"
        )
    } else {
        String::new()
    };

    let mut final_judge_response: Option<String> = None;

    if !judge {
        let _ = output.write_str("(judge=false — skipping synthesis for external judge)\n\n");

        output_parts.push(format!(
            "## Council Deliberation Transcript\n\n{deliberation_text}"
        ));

        let duration = start.elapsed().as_secs_f64();
        let total_cost = (cost_tracker.total() * 10_000.0).round() / 10_000.0;

        let mut meta = json!({
            "judge": "external",
            "question": question,
            "models_used": council_config.iter().map(|(name, _, _)| *name).collect::<Vec<_>>(),
            "rounds": current_round.max(1),
            "duration_seconds": (duration * 10.0).round() / 10.0,
            "estimated_cost_usd": total_cost,
        });

        if let Some(c) = context {
            meta["context"] = json!(c);
        }
        if let Some(d) = domain {
            meta["domain"] = json!(d);
        }
        if let Some(p) = persona {
            meta["persona"] = json!(p);
        }
        if social_mode {
            meta["social_mode"] = json!(true);
        }

        output_parts.push(format!(
            "\n\n---\n\n{}",
            serde_json::to_string_pretty(&meta).unwrap_or_else(|_| "{}".to_string())
        ));
    } else {
        let context_hint = context
            .map(|c| format!("\n\nContext about this question: {c}\nConsider this context when weighing perspectives and forming recommendations."))
            .unwrap_or_default();

        let domain_hint_text = domain_hint
            .map(|ctx| format!("\n\nDOMAIN CONTEXT: {ctx}\nConsider this regulatory domain context when weighing perspectives and forming recommendations."))
            .unwrap_or_default();

        let social_judge_section = if social_mode {
            "\n\n## Social Calibration Check\n[Would the recommendation feel natural in conversation? Is it something you'd actually say, or does it sound like strategic over-optimization? If the council produced something too formal/structured, suggest a simpler, more human alternative.]"
        } else {
            ""
        };

        let judge_system = format!(
            "You are the Judge (Claude), responsible for synthesizing the council's deliberation.{context_hint}{domain_hint_text}\n\nYou did NOT participate in the deliberation — you're seeing it fresh. This gives you objectivity.\n\nSYNTHESIS METHOD — Analysis of Competing Hypotheses:\nRather than seeking the consensus view, first list ALL plausible conclusions from the deliberation (typically 2-4). For each piece of evidence or argument raised by the council, evaluate how well it supports or undermines EACH hypothesis. Eliminate conclusions that are inconsistent with the strongest evidence. The surviving hypothesis is your recommendation.\n\nCONVERGENCE SIGNAL:\nWhen independent agents with different models and training reached the SAME conclusion in the blind phase, treat this as a multiplicatively strong signal — independent agreement from different priors is more reliable than the same conclusion repeated. Push your confidence further toward certainty than a simple average — compare the BLIND CLAIMS section above against each speaker's final debate position. A speaker whose blind claim matches their final position held it independently under no social pressure; a speaker whose final position diverged from their blind claim may have been sycophantically influenced.\n\nSYCOPHANCY CHECK:\nFlag any agent that changed position during debate WITHOUT citing a specific new argument or piece of evidence. Position changes labeled POSITION CHANGE with clear reasoning are healthy. Unlabeled shifts toward consensus are sycophancy — discount these.\n\nAfter applying this method, structure your response as:\n\n## Competing Hypotheses\n[List 2-4 plausible conclusions. For each, note which council arguments support/undermine it]\n\n## Points of Agreement\n[What the council agrees on — and whether that consensus should be trusted given the sycophancy check]\n\n## Points of Disagreement\n[Where views genuinely diverged and why — these often point to the crux]\n\n## Judge's Own Take\n[Your independent perspective. What did the council miss or underweight?]\n\n## Synthesis\n[The integrated perspective, combining council views with your own ACH analysis]\n\n## Recommendation\n[Your final recommendation]{social_judge_section}\nBe balanced and fair. Acknowledge minority views if valid. END with three clear sections: **Do Now** (MAX 3 items, argue against each first), **Consider Later**, and **Skip** (with reasons)."
        );

        let judge_messages = vec![
            Message::system(judge_system),
            Message::user(format!(
                "Question:\n{question}\n\n---\n\n{blind_section}Council Deliberation:\n\n{deliberation_text}{confidence_summary}"
            )),
        ];

        let judge_name = judge_model
            .split('/')
            .next_back()
            .unwrap_or(judge_model.as_str());
        let judge_t0 = Instant::now();
        let _ = output.begin_phase("JUDGMENT");
        let _ = output.begin_participant(&format!("Judge ({judge_name})"));
        let _ = output.write_str(&format!("### Judge ({judge_name})\n"));

        let mut judge_response = query_judge(
            &client,
            api_key,
            judge_model.as_str(),
            &judge_messages,
            1200,
            300.0,
            2,
            Some(&cost_tracker),
        )
        .await;

        let _ = output.write_str(&format!("{judge_response}\n\n"));
        let _ = output.end_participant(
            &format!("Judge ({judge_name})"),
            &judge_response,
            judge_t0.elapsed().as_millis() as u64,
        );

        output_parts.push(format!("### Judge ({judge_name})\n{judge_response}"));

        if collabeval {
            let critique_system = format!(
                "You are an independent critic reviewing a judge's synthesis of a multi-model council deliberation.\n\nYour job is to find WEAKNESSES in the judge's synthesis — not to agree with it.\n\nLook for:\n1. Points the judge dismissed too quickly or weighted incorrectly\n2. Minority views that deserved more consideration\n3. Logical gaps or unsupported leaps in the recommendation\n4. Practical concerns the judge missed\n5. Whether the \"Do Now\" items are truly the right priorities\n6. Logical fallacies: unsupported premises, invalid inferences, false dichotomies, correlation-causation conflation, strawman arguments — judges often overlook these even when the conclusion seems reasonable\n\nBe specific and concise. Name the exact weakness and why it matters.\nIf the synthesis is genuinely strong, say so briefly — but try hard to find something.{}",
                domain
                    .map(|d| format!(" Consider the {d} regulatory context."))
                    .unwrap_or_default()
            );

            let critique_messages = vec![
                Message::system(critique_system),
                Message::user(format!(
                    "Question:\n{question}\n\nJudge's Synthesis:\n\n{judge_response}"
                )),
            ];

            let critique_model = resolved_critique_model();
            let critique_name = critique_model
                .split('/')
                .next_back()
                .unwrap_or(&critique_model);
            let critique_t0 = Instant::now();
            let _ = output.begin_phase("JUDGMENT");
            let _ = output.begin_participant(&format!("Critique ({critique_name})"));
            let _ = output.write_str(&format!("### Critique ({critique_name})\n"));

            let critique_response = query_model(
                &client,
                api_key,
                &critique_model,
                &critique_messages,
                800,
                300.0,
                2,
                Some(&cost_tracker),
            )
            .await;

            let _ = output.write_str(&format!("{critique_response}\n\n"));
            let _ = output.end_participant(
                &format!("Critique ({critique_name})"),
                &critique_response,
                critique_t0.elapsed().as_millis() as u64,
            );

            output_parts.push(format!(
                "### Critique ({critique_name})\n{critique_response}"
            ));

            if is_error_response(&critique_response) {
                let _ = output.write_str("(Critique unavailable — synthesis is unreviewed)\n\n");
                output_parts.push("*(Critique unavailable — synthesis is unreviewed)*".to_string());
            } else {
                let final_t0 = Instant::now();
                let _ = output.begin_phase("JUDGMENT");
                let _ = output.begin_participant(&format!("Final Synthesis ({judge_name})"));
                let _ = output.write_str(&format!("### Final Synthesis ({judge_name})\n"));

                let mut revision_messages = judge_messages.clone();
                revision_messages.push(Message::assistant(judge_response.clone()));
                revision_messages.push(Message::user(format!(
                    "An independent critic has reviewed your synthesis:\n\n{critique_response}\n\nRevise your synthesis considering this critique. Keep what's right, fix what's wrong. If the critique raises valid points, integrate them. If not, explain briefly why you stand by your original position. Output your FINAL revised synthesis in the same format."
                )));

                let final_response = query_judge(
                    &client,
                    api_key,
                    judge_model.as_str(),
                    &revision_messages,
                    1200,
                    300.0,
                    2,
                    Some(&cost_tracker),
                )
                .await;

                let _ = output.write_str(&format!("{final_response}\n\n"));
                let _ = output.end_participant(
                    &format!("Final Synthesis ({judge_name})"),
                    &final_response,
                    final_t0.elapsed().as_millis() as u64,
                );

                output_parts.push(format!(
                    "### Final Synthesis ({judge_name})\n{final_response}"
                ));
                judge_response = final_response;
            }
        }

        final_judge_response = Some(judge_response.clone());

        if format != "prose" {
            let models_used: Vec<String> = council_config
                .iter()
                .map(|(name, _, _)| name.to_string())
                .collect();
            let total_cost = (cost_tracker.total() * 10_000.0).round() / 10_000.0;
            let duration = start.elapsed().as_secs_f64();

            let structured = extract_structured_summary(
                &judge_response,
                question,
                &models_used,
                current_round.max(1),
                duration,
                total_cost,
                &client,
                Some(api_key),
                Some(&cost_tracker),
            )
            .await;

            let rendered = if format == "json" {
                serde_json::to_string_pretty(&structured).unwrap_or_else(|_| "{}".to_string())
            } else {
                serde_yaml::to_string(&structured).unwrap_or_else(|_| "{}\n".to_string())
            };

            let json_block = format!("\n\n---\n\n{rendered}");
            output_parts.push(json_block.clone());
            let _ = output.write_str(&format!("{json_block}\n"));
        }
    }

    if followup && judge {
        let topic = final_judge_response
            .as_deref()
            .map(parse_recommendation_items)
            .and_then(|items| items.do_now.first().cloned())
            .unwrap_or_else(|| "biggest unresolved trade-off".to_string());

        let followup_text = run_followup_discussion(
            question,
            &topic,
            council_config,
            api_key,
            google_api_key,
            zhipu_api_key,
            moonshot_api_key,
            openai_api_key,
            xai_api_key,
            timeout,
            domain_hint,
            social_mode,
            persona,
            output,
            &cost_tracker,
        )
        .await;

        output_parts.push(followup_text);
    }

    if let Some(line) = confidence_line {
        let _ = output.write_str(&format!("  {line}\n\n"));
    }

    let mut transcript = output_parts.join("\n\n");

    if anonymous {
        for &(name, model, _) in council_config {
            let anon_name = display_names
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string());
            let model_name = model.split('/').next_back().unwrap_or(model);

            transcript =
                transcript.replace(&format!("### {anon_name}"), &format!("### {model_name}"));
            transcript = transcript.replace(&format!("[{anon_name}]"), &format!("[{model_name}]"));
            transcript =
                transcript.replace(&format!("**{anon_name}**"), &format!("**{model_name}**"));
            transcript =
                transcript.replace(&format!("with {anon_name}"), &format!("with {model_name}"));
            transcript = transcript.replace(&format!("{anon_name}'s"), &format!("{model_name}'s"));
        }
    }

    if !failed_models.is_empty() {
        let _ = output.write_str("\n");
        let _ = output.write_str(&"=".repeat(60));
        let _ = output.write_str("\nMODEL FAILURES\n");
        let _ = output.write_str(&"=".repeat(60));
        let _ = output.write_str("\n");
        for failure in &failed_models {
            let _ = output.write_str(&format!("  - {failure}\n"));
        }
        let unique_failed = failed_models
            .iter()
            .filter_map(|f| f.split(':').next())
            .map(|s| s.split(" (").next().unwrap_or(s).trim().to_string())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let working_count = council_config.len().saturating_sub(unique_failed);
        let _ = output.write_str(&format!(
            "\nCouncil ran with {working_count}/{} models\n",
            council_config.len()
        ));
        let _ = output.write_str(&"=".repeat(60));
        let _ = output.write_str("\n\n");
    }

    let duration = start.elapsed().as_secs_f64();
    let total_cost = (cost_tracker.total() * 10_000.0).round() / 10_000.0;

    let _ = output.write_str(&format!("({duration:.1}s, ~${total_cost:.2})\n"));

    SessionResult {
        transcript,
        cost: total_cost,
        duration,
        failures: if failed_models.is_empty() {
            None
        } else {
            Some(failed_models)
        },
    }
}
