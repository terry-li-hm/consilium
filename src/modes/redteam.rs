//! Red team mode: adversarial stress-test of a plan or decision.

use crate::api::{query_model, run_parallel_with_different_messages};
use crate::config::{
    sanitize_speaker_content, CostTracker, Message, ModelEntry, SessionResult, DISCUSS_HOST,
};
use crate::prompts::{
    redteam_attacker_deepen, redteam_attacker_system, redteam_host_analysis, redteam_host_deepen,
    redteam_host_triage,
};
use reqwest::Client;
use std::time::Instant;

pub async fn run_redteam(
    question: &str,
    panelists: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    _context: Option<String>,
    _format: &str,
    timeout: f64,
    quiet: bool,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();
    let verbose = !quiet;

    let mut transcript_parts = Vec::new();
    let mut conversation_history: Vec<(String, String)> = Vec::new();

    if verbose {
        println!("============================================================");
        println!("RED TEAM");
        println!("============================================================");
        println!();
    }

    // Phase 1: ANALYSIS
    if verbose {
        println!("## Analysis
### Host (Claude)");
    }
    let analysis_messages = vec![
        Message::system(redteam_host_analysis()),
        Message::user(question),
    ];
    let host_analysis = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &analysis_messages,
        500,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    if verbose {
        println!("{host_analysis}
");
    }
    transcript_parts.push(format!("## Analysis

### Host (Claude)
{host_analysis}"));
    conversation_history.push(("Host (Claude)".to_string(), host_analysis.clone()));

    // Phase 2: ATTACKS (parallel)
    if verbose {
        println!(
            "## Attacks
(querying {} attackers in parallel...)",
            panelists.len()
        );
    }
    let mut attack_messages_list = Vec::with_capacity(panelists.len());
    for (name, _, _) in panelists {
        let attacker_system =
            redteam_attacker_system(name, &sanitize_speaker_content(&host_analysis));
        attack_messages_list.push(vec![
            Message::system(attacker_system),
            Message::user(format!("Plan/decision to attack:

{question}")),
        ]);
    }

    let attack_results = run_parallel_with_different_messages(
        panelists,
        &attack_messages_list,
        api_key,
        google_api_key,
        600,
        Some(&cost_tracker),
        verbose,
    )
    .await;

    transcript_parts.push("## Attacks".to_string());
    for (name, _, response) in attack_results {
        transcript_parts.push(format!("### {name}
{response}"));
        conversation_history.push((name, response));
    }

    if verbose {
        println!();
    }

    // Phase 3: DEEPENING (Host)
    if verbose {
        println!("## Deepening
### Host (Claude)");
    }
    let attacks_text = conversation_history[1..]
        .iter()
        .map(|(speaker, text)| {
            format!(
                "**{speaker}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("

");

    let deepen_messages = vec![
        Message::system(redteam_host_deepen()),
        Message::user(format!(
            "Plan/decision:
{question}

Initial attacks:

{attacks_text}"
        )),
    ];
    let host_deepen = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &deepen_messages,
        500,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    if verbose {
        println!("{host_deepen}
");
    }
    transcript_parts.push(format!(
        "## Deepening

### Host (Claude)
{host_deepen}"
    ));
    conversation_history.push(("Host (Claude)".to_string(), host_deepen));

    // Phase 4: DEEPENING (Attackers - sequential)
    for (name, model, _) in panelists {
        let other_names: Vec<String> = panelists
            .iter()
            .filter(|(n, _, _)| n != name)
            .map(|(n, _, _)| n.to_string())
            .collect();

        let other1 = other_names
            .get(0)
            .cloned()
            .unwrap_or_else(|| "another attacker".to_string());
        let other2 = other_names
            .get(1)
            .cloned()
            .unwrap_or_else(|| "another attacker".to_string());

        let attacker_deepen_system = redteam_attacker_deepen(name, &other1, &other2);

        let full_history = conversation_history
            .iter()
            .map(|(speaker, text)| {
                format!(
                    "**{speaker}**: {text}",
                    text = sanitize_speaker_content(text)
                )
            })
            .collect::<Vec<_>>()
            .join("

");

        let deepen_attacker_messages = vec![
            Message::system(attacker_deepen_system),
            Message::user(format!(
                "Plan/decision:
{question}

Discussion so far:

{full_history}

Find cascading/compound failures."
            )),
        ];

        if verbose {
            println!("### {name}");
        }
        let response = query_model(
            &client,
            api_key,
            model,
            &deepen_attacker_messages,
            600,
            timeout,
            2,
            Some(&cost_tracker),
        )
        .await;

        if verbose {
            println!("{response}
");
        }
        transcript_parts.push(format!("### {name}
{response}"));
        conversation_history.push((name.to_string(), response));
    }

    // Phase 5: TRIAGE
    if verbose {
        println!("## Triage
### Host (Claude)");
    }
    let full_history = conversation_history
        .iter()
        .map(|(speaker, text)| {
            format!(
                "**{speaker}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("

");

    let triage_messages = vec![
        Message::system(redteam_host_triage()),
        Message::user(format!(
            "Plan/decision:
{question}

Full red team discussion:

{full_history}"
        )),
    ];
    let host_triage = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &triage_messages,
        800,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    if verbose {
        println!("{host_triage}
");
    }
    transcript_parts.push(format!("## Triage

### Host (Claude)
{host_triage}"));

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
