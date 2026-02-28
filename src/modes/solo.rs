//! Solo council mode: one model, structured deliberation with multiple perspectives.

use crate::api::{query_model, run_parallel_with_different_messages};
use crate::config::{
    parse_confidence, sanitize_speaker_content, CostTracker, Message, SessionResult, JUDGE_MODEL,
};
use crate::prompts::{
    role_library, solo_blind_system, solo_challenger_addition, solo_debate_system, solo_judge_system,
    SOLO_DEFAULT_ROLES,
};
use reqwest::Client;
use std::time::Instant;

pub async fn run_solo(
    question: &str,
    model: &str,
    api_key: &str,
    google_api_key: Option<&str>,
    roles: Option<String>,
    _format: &str,
    timeout: f64,
    quiet: bool,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();
    let verbose = !quiet;

    let role_lib = role_library();
    let role_names: Vec<String> = if let Some(r) = roles {
        r.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        SOLO_DEFAULT_ROLES.iter().map(|s| s.to_string()).collect()
    };

    let mut perspectives: Vec<(&'static str, &'static str)> = Vec::new();
    for role in role_names {
        let mut description = None;
        for (lib_name, lib_desc) in &role_lib {
            if lib_name.to_lowercase() == role.to_lowercase() {
                description = Some(lib_desc.to_string());
                break;
            }
        }

        let desc = description.unwrap_or_else(|| {
            format!(
                "You approach this as a {role}. Bring your professional lens, domain expertise, and the specific concerns someone in your role would have. Be specific and opinionated — generic advice is worthless."
            )
        });
        
        // Leak for 'static refs in CLI tool
        let leaked_name: &'static str = Box::leak(role.into_boxed_str());
        let leaked_desc: &'static str = Box::leak(desc.into_boxed_str());
        perspectives.push((leaked_name, leaked_desc));
    }

    if perspectives.len() < 2 {
        // Should have at least 2 for a debate
        for role in SOLO_DEFAULT_ROLES {
            if !perspectives
                .iter()
                .any(|(n, _)| n.to_lowercase() == role.to_lowercase())
            {
                perspectives.push((*role, role_lib.get(role).unwrap()));
            }
            if perspectives.len() >= 3 {
                break;
            }
        }
    }

    let challenger_name = perspectives[1].0;
    let leaked_model: &'static str = Box::leak(model.to_string().into_boxed_str());
    let mut transcript_parts = Vec::new();

    if verbose {
        println!("============================================================");
        println!("SOLO COUNCIL");
        println!("============================================================");
        println!();
    }

    // Phase 1: BLIND (parallel)
    if verbose {
        println!(
            "## Blind Phase
(generating {} perspectives in parallel...)",
            perspectives.len()
        );
    }

    let mut blind_messages_list = Vec::with_capacity(perspectives.len());
    for (name, desc) in &perspectives {
        blind_messages_list.push(vec![
            Message::system(solo_blind_system(name, desc)),
            Message::user(question),
        ]);
    }

    // Using the same model for all perspectives
    let pseudo_panelists: Vec<(&'static str, &'static str, Option<(&'static str, &'static str)>)> =
        perspectives
            .iter()
            .map(|(name, _)| (*name, leaked_model, None))
            .collect();

    let blind_results = run_parallel_with_different_messages(
        &pseudo_panelists,
        &blind_messages_list,
        api_key,
        google_api_key,
        500,
        Some(&cost_tracker),
        verbose,
    )
    .await;

    transcript_parts.push("## Blind Phase".to_string());
    let mut blind_claims = Vec::new();
    for (name, _, response) in blind_results {
        transcript_parts.push(format!("### {name}
{response}"));
        blind_claims.push((name, response));
    }

    if verbose {
        println!();
    }

    // Phase 2: DEBATE (sequential)
    if verbose {
        println!("## Debate
");
    }
    transcript_parts.push("## Debate".to_string());

    let blind_text = blind_claims
        .iter()
        .map(|(name, text)| {
            format!(
                "**{name}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("

");

    let mut debate_responses = Vec::new();
    let mut confidences: Vec<(String, Vec<u8>)> = Vec::new();

    // Collect blind phase confidences
    for (name, text) in &blind_claims {
        if let Some(c) = parse_confidence(text) {
            confidences.push((name.to_string(), vec![c]));
        }
    }

    for (name, desc) in &perspectives {
        let mut system = solo_debate_system(name, desc);
        if *name == challenger_name {
            system += &solo_challenger_addition();
        }

        let mut debate_context = format!("Blind phase perspectives:

{blind_text}");
        if !debate_responses.is_empty() {
            let debate_text = debate_responses
                .iter()
                .map(|(n, t): &(String, String)| {
                    format!(
                        "**{n}**: {t}",
                        t = sanitize_speaker_content(t)
                    )
                })
                .collect::<Vec<_>>()
                .join("

");
            debate_context = format!("{debate_context}

Debate responses so far:

{debate_text}");
        }

        let messages = vec![
            Message::system(system),
            Message::user(format!(
                "Question:

{question}

---

{debate_context}"
            )),
        ];

        let is_challenger = *name == challenger_name;
        let challenger_tag = if is_challenger { " (challenger)" } else { "" };

        if verbose {
            println!("### {name}{challenger_tag}");
        }

        let response = query_model(
            &client,
            api_key,
            model,
            &messages,
            500,
            timeout,
            2,
            Some(&cost_tracker),
        )
        .await;

        if verbose {
            println!("{response}
");
        }
        transcript_parts.push(format!("### {name}{challenger_tag}
{response}"));
        debate_responses.push((name.to_string(), response.clone()));

        if let Some(c) = parse_confidence(&response) {
            if let Some(v) = confidences.iter_mut().find(|(n, _)| n == name) {
                v.1.push(c);
            } else {
                confidences.push((name.to_string(), vec![c]));
            }
        }
    }

    // Confidence drift
    if !confidences.is_empty() && verbose {
        let drift_parts: Vec<String> = confidences
            .iter()
            .map(|(name, scores)| {
                if scores.len() >= 2 {
                    format!("{} {} -> {}", name, scores[0], scores.last().unwrap())
                } else {
                    format!("{} {}/10", name, scores[0])
                }
            })
            .collect();
        if !drift_parts.is_empty() {
            println!("  Confidence: {}
", drift_parts.join(", "));
            transcript_parts.push(format!("Confidence drift: {}", drift_parts.join(", ")));
        }
    }

    // Phase 3: SYNTHESIS
    if verbose {
        println!("## Synthesis
### Judge (Claude)");
    }
    transcript_parts.push("## Synthesis".to_string());

    let mut full_deliberation = format!("Blind phase:

{blind_text}

Debate:

");
    full_deliberation += &debate_responses
        .iter()
        .map(|(name, text)| {
            format!(
                "**{name}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("

");

    let judge_messages = vec![
        Message::system(solo_judge_system()),
        Message::user(format!(
            "Question:
{question}

---

Full deliberation:

{full_deliberation}"
        )),
    ];

    let judge_response = query_model(
        &client,
        api_key,
        JUDGE_MODEL,
        &judge_messages,
        1200,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;

    if verbose {
        println!("{judge_response}
");
    }
    transcript_parts.push(format!("### Judge (Claude)
{judge_response}"));

    let duration = start.elapsed().as_secs_f64();
    let cost = (cost_tracker.total() * 10000.0).round() / 10000.0;

    if verbose {
        println!("({:.1}s, ~${:.2})", duration, cost);
    }

    SessionResult {
        transcript: transcript_parts.join("

"),
        cost,
        duration,
        failures: None,
    }
}
