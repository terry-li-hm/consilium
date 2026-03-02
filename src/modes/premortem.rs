//! Pre-mortem mode: assume failure happened and reason backward to causes.

use crate::api::{query_model, run_parallel_with_different_messages};
use crate::config::{
    sanitize_speaker_content, CostTracker, Message, ModelEntry, SessionResult, DISCUSS_HOST,
};
use crate::prompts::{
    premortem_host_framing, premortem_host_mitigation, premortem_host_synthesis,
    premortem_panelist_system,
};
use crate::session::Output;
use reqwest::Client;
use std::time::Instant;

pub async fn run_premortem(
    question: &str,
    panelists: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    _context: Option<String>,
    _format: &str,
    timeout: f64,
    output: &mut dyn Output,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();

    let mut transcript_parts = Vec::new();
    let mut conversation_history: Vec<(String, String)> = Vec::new();

    let _ = output.write_str("============================================================\n");
    let _ = output.write_str("PRE-MORTEM\n");
    let _ = output.write_str("============================================================\n\n");

    // Phase 1: SETUP
    let _ = output.begin_phase("SETUP");
    let setup_t0 = Instant::now();
    let _ = output.begin_participant("Host (Claude)");
    let _ = output.write_str("## Setup\n### Host (Claude)\n");
    let setup_messages = vec![
        Message::system(premortem_host_framing()),
        Message::user(question),
    ];
    let host_setup = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &setup_messages,
        500,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    let _ = output.write_str(&format!("{}\n\n", host_setup));
    let _ = output.end_participant(
        "Host (Claude)",
        &host_setup,
        setup_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!("## Setup\n\n### Host (Claude)\n{host_setup}"));
    conversation_history.push(("Host (Claude)".to_string(), host_setup.clone()));

    // Phase 2: FAILURE SCENARIOS (parallel)
    let _ = output.begin_phase("FAILURE SCENARIOS");
    let _ = output.write_str(&format!(
        "## Failure Scenarios\n(querying {} panelists in parallel...)\n",
        panelists.len()
    ));
    let mut scenario_messages_list = Vec::with_capacity(panelists.len());
    for (name, _, _) in panelists {
        scenario_messages_list.push(vec![
            Message::system(premortem_panelist_system(name)),
            Message::user(format!(
                "Plan/decision for pre-mortem:\n\n{question}\n\nHost framing:\n\n{}",
                sanitize_speaker_content(&host_setup)
            )),
        ]);
    }

    let scenario_results = run_parallel_with_different_messages(
        panelists,
        &scenario_messages_list,
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        xai_api_key,
        700,
        Some(&cost_tracker),
        None,
    )
    .await;

    transcript_parts.push("## Failure Scenarios".to_string());
    for (name, _, response) in scenario_results {
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(&name, &response, 0);
        transcript_parts.push(format!("### {name}\n{response}"));
        conversation_history.push((name, response));
    }

    let _ = output.write_str("\n");

    // Phase 3: HOST SYNTHESIS
    let _ = output.begin_phase("HOST SYNTHESIS");
    let synthesis_t0 = Instant::now();
    let _ = output.begin_participant("Host (Claude)");
    let _ = output.write_str("## Host Synthesis\n### Host (Claude)\n");
    let narratives_text = conversation_history[1..]
        .iter()
        .map(|(speaker, text)| {
            format!(
                "**{speaker}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let synthesis_messages = vec![
        Message::system(premortem_host_synthesis()),
        Message::user(format!(
            "Plan/decision:\n{question}\n\nFailure narratives:\n\n{narratives_text}"
        )),
    ];
    let host_synthesis = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &synthesis_messages,
        700,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    let _ = output.write_str(&format!("{}\n\n", host_synthesis));
    let _ = output.end_participant(
        "Host (Claude)",
        &host_synthesis,
        synthesis_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!(
        "## Host Synthesis\n\n### Host (Claude)\n{host_synthesis}"
    ));
    conversation_history.push(("Host (Claude)".to_string(), host_synthesis));

    // Phase 4: MITIGATION MAP
    let _ = output.begin_phase("MITIGATION MAP");
    let mitigation_t0 = Instant::now();
    let _ = output.begin_participant("Host (Claude)");
    let _ = output.write_str("## Mitigation Map\n### Host (Claude)\n");
    let full_history = conversation_history
        .iter()
        .map(|(speaker, text)| {
            format!(
                "**{speaker}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let mitigation_messages = vec![
        Message::system(premortem_host_mitigation()),
        Message::user(format!(
            "Plan/decision:\n{question}\n\nPre-mortem discussion so far:\n\n{full_history}"
        )),
    ];
    let host_mitigation = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &mitigation_messages,
        800,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    let _ = output.write_str(&format!("{}\n\n", host_mitigation));
    let _ = output.end_participant(
        "Host (Claude)",
        &host_mitigation,
        mitigation_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!(
        "## Mitigation Map\n\n### Host (Claude)\n{host_mitigation}"
    ));

    let duration = start.elapsed().as_secs_f64();
    let cost = (cost_tracker.total() * 10000.0).round() / 10000.0;

    let _ = output.write_str(&format!("({:.1}s, ~${:.2})\n", duration, cost));

    SessionResult {
        transcript: transcript_parts.join("\n\n"),
        cost,
        duration,
        failures: None,
    }
}
