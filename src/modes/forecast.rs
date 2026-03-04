//! Forecast mode: independent probabilities, divergence analysis, and reconciliation.

use crate::api::{query_model, run_parallel_with_different_messages};
use crate::config::{
    sanitize_speaker_content, CostTracker, Message, ModelEntry, SessionResult, DISCUSS_HOST,
};
use crate::prompts::{
    forecast_blind_system, forecast_host_divergence, forecast_host_synthesis,
    forecast_reconcile_system,
};
use crate::session::Output;
use reqwest::Client;
use std::time::Instant;

pub async fn run_forecast(
    question: &str,
    panelists: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    anthropic_api_key: Option<&str>,
    _context: Option<String>,
    _format: &str,
    timeout: f64,
    output: &mut dyn Output,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();

    let mut transcript_parts = Vec::new();

    let _ = output.write_str("============================================================\n");
    let _ = output.write_str("FORECAST\n");
    let _ = output.write_str("============================================================\n\n");

    // Phase 1: BLIND ESTIMATES (parallel)
    let _ = output.begin_phase("BLIND ESTIMATES");
    let _ = output.write_str(&format!(
        "## Blind Estimates\n(querying {} panelists in parallel...)\n",
        panelists.len()
    ));
    let mut blind_messages_list = Vec::with_capacity(panelists.len());
    for (name, _, _) in panelists {
        blind_messages_list.push(vec![
            Message::system(forecast_blind_system(name)),
            Message::user(format!("Forecast question:\n\n{question}")),
        ]);
    }

    let blind_results = run_parallel_with_different_messages(
        panelists,
        &blind_messages_list,
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        openai_api_key,
        xai_api_key,
        anthropic_api_key,
        500,
        timeout,
        Some(&cost_tracker),
        None,
        None,
    )
    .await;

    transcript_parts.push("## Blind Estimates".to_string());
    let mut blind_estimates = Vec::new();
    for (name, _, response) in blind_results {
        let _ = output.begin_participant(&name);
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(&name, &response, 0);
        transcript_parts.push(format!("### {name}\n{response}"));
        blind_estimates.push((name, response));
    }

    let all_blind_estimates = blind_estimates
        .iter()
        .map(|(speaker, text)| format!("**{speaker}**:\n{}", sanitize_speaker_content(text).trim()))
        .collect::<Vec<_>>()
        .join("\n\n");

    // Phase 2: DIVERGENCE ANALYSIS (host)
    let _ = output.begin_phase("DIVERGENCE ANALYSIS");
    let divergence_t0 = Instant::now();
    let _ = output.begin_participant("Host (Claude)");
    let _ = output.write_str("## Divergence Analysis\n### Host (Claude)\n");
    let divergence_messages = vec![
        Message::system(forecast_host_divergence(&all_blind_estimates)),
        Message::user(format!(
            "Question:\n{question}\n\nBlind estimates:\n\n{all_blind_estimates}"
        )),
    ];
    let host_divergence = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &divergence_messages,
        500,
        timeout,
        2,
        Some(&cost_tracker),
        None,
    )
    .await;
    let _ = output.write_str(&format!("{}\n\n", host_divergence));
    let _ = output.end_participant(
        "Host (Claude)",
        &host_divergence,
        divergence_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!(
        "## Divergence Analysis\n\n### Host (Claude)\n{host_divergence}"
    ));

    // Phase 3: RECONCILIATION (parallel)
    let _ = output.begin_phase("RECONCILIATION");
    let _ = output.write_str(&format!(
        "## Reconciliation\n(querying {} panelists in parallel...)\n",
        panelists.len()
    ));
    let sanitized_divergence = sanitize_speaker_content(&host_divergence);
    let mut reconcile_messages_list = Vec::with_capacity(panelists.len());
    for (name, _, _) in panelists {
        reconcile_messages_list.push(vec![
            Message::system(forecast_reconcile_system(
                name,
                &all_blind_estimates,
                &sanitized_divergence,
            )),
            Message::user(format!("Forecast question:\n\n{question}")),
        ]);
    }

    let reconcile_results = run_parallel_with_different_messages(
        panelists,
        &reconcile_messages_list,
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        openai_api_key,
        xai_api_key,
        anthropic_api_key,
        600,
        timeout,
        Some(&cost_tracker),
        None,
        None,
    )
    .await;

    transcript_parts.push("## Reconciliation".to_string());
    let mut final_estimates = Vec::new();
    for (name, _, response) in reconcile_results {
        let _ = output.begin_participant(&name);
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(&name, &response, 0);
        transcript_parts.push(format!("### {name}\n{response}"));
        final_estimates.push((name, response));
    }

    let final_estimates_text = final_estimates
        .iter()
        .map(|(speaker, text)| format!("**{speaker}**:\n{}", sanitize_speaker_content(text).trim()))
        .collect::<Vec<_>>()
        .join("\n\n");

    // Phase 4: FINAL DISTRIBUTION (host)
    let _ = output.begin_phase("FINAL DISTRIBUTION");
    let distribution_t0 = Instant::now();
    let _ = output.begin_participant("Host (Claude)");
    let _ = output.write_str("## Final Distribution\n### Host (Claude)\n");
    let synthesis_messages = vec![
        Message::system(forecast_host_synthesis()),
        Message::user(format!(
            "Question:\n{question}\n\nFinal reconciled estimates:\n\n{final_estimates_text}"
        )),
    ];
    let host_distribution = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &synthesis_messages,
        500,
        timeout,
        2,
        Some(&cost_tracker),
        None,
    )
    .await;
    let _ = output.write_str(&format!("{}\n\n", host_distribution));
    let _ = output.end_participant(
        "Host (Claude)",
        &host_distribution,
        distribution_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!(
        "## Final Distribution\n\n### Host (Claude)\n{host_distribution}"
    ));

    let duration = start.elapsed().as_secs_f64();
    let cost = (cost_tracker.total() * 10000.0).round() / 10000.0;
    let _ = output.write_str(&format!("({:.1}s, ~${:.2})\n", duration, cost));

    SessionResult {
        transcript: transcript_parts.join("\n\n"),
        cost,
        duration,
        failures: None,
        extra: None,
    }
}
