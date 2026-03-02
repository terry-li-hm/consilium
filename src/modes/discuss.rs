//! Discussion mode: hosted roundtable exploration.

use crate::api::{query_model, run_parallel};
use crate::config::{
    sanitize_speaker_content, CostTracker, Message, ModelEntry, SessionResult, DISCUSS_HOST,
};
use crate::prompts::{
    discuss_host_closing, discuss_host_framing, discuss_host_steer, discuss_panelist_closing,
    discuss_panelist_system, socratic_host_opening, socratic_host_probe, socratic_host_synthesis,
    socratic_panelist_closing, socratic_panelist_system,
};
use crate::session::Output;
use reqwest::Client;
use std::time::Instant;

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
        "Summarize this roundtable round. For each speaker (host and panelists), capture:\n1. Core position or steering direction (1 sentence)\n2. Key new argument, rebuttal, or probe (1 sentence)\n3. Whether they agree/disagree with the emerging consensus\n\nKeep exact quotes only if they contain specific data points or citations.\n\nTopic: {question}\n\nRound responses:\n{round_summary}"
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

pub async fn run_discuss(
    question: &str,
    panelists: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    style: &str,
    rounds: u32,
    _context: Option<String>,
    _format: &str,
    timeout: f64,
    output: &mut dyn Output,
    thorough: bool,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();

    let is_socratic = style == "socratic";
    let host_name = if is_socratic {
        "Examiner (Claude)"
    } else {
        "Host (Claude)"
    };

    let mut transcript_parts = Vec::new();
    let mut conversation_history: Vec<(String, String)> = Vec::new();
    let mut compressed_summaries: Vec<String> = Vec::new();

    let _ = output.write_str("============================================================\n");
    if is_socratic {
        let _ = output.write_str("SOCRATIC EXAMINATION\n");
    } else {
        let _ = output.write_str("ROUNDTABLE DISCUSSION\n");
    }
    let _ = output.write_str("============================================================\n\n");

    // Phase 1: FRAMING / QUESTIONS
    let opening_label = if is_socratic { "Questions" } else { "Opening" };
    let _ = output.begin_phase(&opening_label.to_ascii_uppercase());
    let framing_t0 = Instant::now();
    let _ = output.begin_participant(host_name);
    let _ = output.write_str(&format!("## {}\n### {}\n", opening_label, host_name));

    let framing_system = if is_socratic {
        socratic_host_opening()
    } else {
        discuss_host_framing()
    };

    let framing_messages = vec![Message::system(framing_system), Message::user(question)];

    let host_framing = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &framing_messages,
        500,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;

    let _ = output.write_str(&format!("{}\n\n", host_framing));
    let _ = output.end_participant(
        host_name,
        &host_framing,
        framing_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!(
        "## {}\n\n### {}\n{}",
        opening_label, host_name, host_framing
    ));
    conversation_history.push((host_name.to_string(), host_framing.clone()));

    // Panelist opening takes / answers (parallel)
    let _ = output.write_str(&format!(
        "(querying {} panelists in parallel...)\n",
        panelists.len()
    ));

    let opening_system = if is_socratic {
        socratic_panelist_system("a panelist", 200)
    } else {
        discuss_panelist_system("a panelist", "the host", 150)
    };

    let opening_user = if is_socratic {
        format!(
            "Topic: {question}\n\nThe examiner asks:\n{host_framing}\n\nAnswer each question directly.",
            host_framing = sanitize_speaker_content(&host_framing)
        )
    } else {
        format!(
            "Topic: {question}\n\nThe host opened with:\n{host_framing}\n\nGive your opening take.",
            host_framing = sanitize_speaker_content(&host_framing)
        )
    };

    let opening_messages = vec![Message::system(opening_system), Message::user(opening_user)];

    let opening_results = run_parallel(
        panelists,
        &opening_messages,
        api_key,
        google_api_key,
        zhipu_api_key,
        500,
        Some(&cost_tracker),
        None,
    )
    .await;

    if is_socratic {
        transcript_parts.push("## Answers".to_string());
    }

    for (name, _, response) in opening_results {
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(&name, &response, 0);
        transcript_parts.push(format!("### {name}\n{response}"));
        conversation_history.push((name, response));
    }

    let _ = output.write_str("\n");

    // Phase 2: DISCUSSION / PROBING
    let round_label = if is_socratic {
        "Probing Round"
    } else {
        "Round"
    };
    let history_label = if is_socratic {
        "Examination"
    } else {
        "Discussion"
    };
    let mut round_num = 0;

    loop {
        if rounds > 0 && round_num >= rounds {
            break;
        }

        round_num += 1;
        let _ = output.begin_phase(&format!(
            "{} {}",
            round_label.to_ascii_uppercase(),
            round_num
        ));

        let _ = output.write_str(&format!("## {} {}\n\n", round_label, round_num));
        transcript_parts.push(format!("## {} {}", round_label, round_num));

        // Host steering / probing
        let host_steer_t0 = Instant::now();
        let _ = output.begin_participant(host_name);
        let _ = output.write_str(&format!("### {}\n", host_name));

        let steer_system = if is_socratic {
            socratic_host_probe()
        } else {
            discuss_host_steer()
        };

        let history_text = if !thorough && !compressed_summaries.is_empty() {
            let mut parts = Vec::new();
            let opening_len = 1 + panelists.len();
            for (speaker, text) in conversation_history.iter().take(opening_len) {
                parts.push(format!(
                    "**{speaker}**: {text}",
                    text = sanitize_speaker_content(text)
                ));
            }
            for (i, summary) in compressed_summaries.iter().enumerate() {
                parts.push(format!("Summary of Round {}:\n{}", i + 1, summary));
            }
            let current_round_start_idx =
                opening_len + compressed_summaries.len() * (1 + panelists.len());
            for (speaker, text) in conversation_history.iter().skip(current_round_start_idx) {
                parts.push(format!(
                    "**{speaker}**: {text}",
                    text = sanitize_speaker_content(text)
                ));
            }
            parts.join("\n\n")
        } else {
            conversation_history
                .iter()
                .map(|(speaker, text)| {
                    format!(
                        "**{speaker}**: {text}",
                        text = sanitize_speaker_content(text)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        let steer_messages = vec![
            Message::system(steer_system),
            Message::user(format!(
                "Topic: {question}\n\n{history_label} so far:\n\n{history_text}"
            )),
        ];

        let host_steer = query_model(
            &client,
            api_key,
            DISCUSS_HOST,
            &steer_messages,
            300,
            timeout,
            2,
            Some(&cost_tracker),
        )
        .await;

        let _ = output.write_str(&format!("{}\n\n", host_steer));
        let _ = output.end_participant(
            host_name,
            &host_steer,
            host_steer_t0.elapsed().as_millis() as u64,
        );
        transcript_parts.push(format!("### {}\n{}", host_name, host_steer));
        conversation_history.push((host_name.to_string(), host_steer));

        // Panelists respond sequentially
        for (name, model, _) in panelists {
            let panelist_system = if is_socratic {
                socratic_panelist_system(name, 150)
            } else {
                discuss_panelist_system(name, "the others", 150)
            };

            let follow_up_cue = if is_socratic {
                "The examiner just asked a follow-up. Answer directly."
            } else {
                "The host just asked a follow-up. Give your response."
            };

            let history_text = if !thorough && !compressed_summaries.is_empty() {
                let mut parts = Vec::new();
                let opening_len = 1 + panelists.len();
                for (speaker, text) in conversation_history.iter().take(opening_len) {
                    parts.push(format!(
                        "**{speaker}**: {text}",
                        text = sanitize_speaker_content(text)
                    ));
                }
                for (i, summary) in compressed_summaries.iter().enumerate() {
                    parts.push(format!("Summary of Round {}:\n{}", i + 1, summary));
                }
                let current_round_start_idx =
                    opening_len + compressed_summaries.len() * (1 + panelists.len());
                for (speaker, text) in conversation_history.iter().skip(current_round_start_idx) {
                    parts.push(format!(
                        "**{speaker}**: {text}",
                        text = sanitize_speaker_content(text)
                    ));
                }
                parts.join("\n\n")
            } else {
                conversation_history
                    .iter()
                    .map(|(speaker, text)| {
                        format!(
                            "**{speaker}**: {text}",
                            text = sanitize_speaker_content(text)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            };

            let panelist_messages = vec![
                Message::system(panelist_system),
                Message::user(format!("Topic: {question}\n\n{history_label} so far:\n\n{history_text}\n\n{follow_up_cue}")),
            ];

            let panelist_t0 = Instant::now();
            let _ = output.begin_participant(name);
            let _ = output.write_str(&format!("### {name}\n"));
            let response = query_model(
                &client,
                api_key,
                model,
                &panelist_messages,
                150,
                timeout,
                2,
                Some(&cost_tracker),
            )
            .await;

            let _ = output.write_str(&format!("{}\n\n", response));
            let _ =
                output.end_participant(name, &response, panelist_t0.elapsed().as_millis() as u64);
            transcript_parts.push(format!("### {name}\n{response}"));
            conversation_history.push((name.to_string(), response));
        }

        if !thorough && rounds > 1 && round_num < rounds {
            let opening_len = 1 + panelists.len();
            let round_size = 1 + panelists.len();
            let round_start = opening_len + (round_num - 1) as usize * round_size;
            let round_responses = &conversation_history[round_start..];
            let _ = output.write_str(&format!("(compressing round {} context...)\n", round_num));
            let summary =
                compress_round_context(round_responses, question, &client, api_key, &cost_tracker)
                    .await;
            compressed_summaries.push(summary);
        }
    }

    // Phase 3: CLOSING
    let _ = output.begin_phase("CLOSING");
    let _ = output.write_str("## Closing\n\n");
    transcript_parts.push("## Closing".to_string());

    let closing_prompt = if is_socratic {
        socratic_panelist_closing()
    } else {
        discuss_panelist_closing()
    };

    let history_text = conversation_history
        .iter()
        .map(|(speaker, text)| {
            format!(
                "**{speaker}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let closing_messages = vec![
        Message::system(closing_prompt),
        Message::user(format!(
            "Topic: {question}\n\nFull {history_label}:\n\n{history_text}"
        )),
    ];

    let _ = output.write_str(&format!(
        "(querying {} panelists in parallel...)\n",
        panelists.len()
    ));

    let closing_results = run_parallel(
        panelists,
        &closing_messages,
        api_key,
        google_api_key,
        zhipu_api_key,
        300,
        Some(&cost_tracker),
        None,
    )
    .await;

    for (name, _, response) in closing_results {
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(&name, &response, 0);
        transcript_parts.push(format!("### {name}\n{response}"));
        conversation_history.push((name, response));
    }

    let _ = output.write_str("\n");

    // Host closing / synthesis
    let closing_host_prompt = if is_socratic {
        socratic_host_synthesis()
    } else {
        discuss_host_closing()
    };

    let history_text = conversation_history
        .iter()
        .map(|(speaker, text)| {
            format!(
                "**{speaker}**: {text}",
                text = sanitize_speaker_content(text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let closing_host_messages = vec![
        Message::system(closing_host_prompt),
        Message::user(format!(
            "Topic: {question}\n\nFull {history_label}:\n\n{history_text}"
        )),
    ];

    if is_socratic {
        let _ = output.begin_phase("SYNTHESIS");
        let _ = output.write_str("## Synthesis\n\n");
        transcript_parts.push("## Synthesis".to_string());
    }

    let host_closing_t0 = Instant::now();
    let _ = output.begin_participant(host_name);
    let _ = output.write_str(&format!("### {}\n", host_name));

    let host_closing = query_model(
        &client,
        api_key,
        DISCUSS_HOST,
        &closing_host_messages,
        if is_socratic { 400 } else { 300 },
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;

    let _ = output.write_str(&format!("{}\n\n", host_closing));
    let _ = output.end_participant(
        host_name,
        &host_closing,
        host_closing_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!("### {}\n{}", host_name, host_closing));

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
