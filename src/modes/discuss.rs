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
use reqwest::Client;
use std::time::Instant;
use tokio::signal;

pub async fn run_discuss(
    question: &str,
    panelists: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    style: &str,
    rounds: u32,
    _context: Option<String>,
    _format: &str,
    timeout: f64,
    quiet: bool,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();
    let verbose = !quiet;

    let is_socratic = style == "socratic";
    let host_name = if is_socratic {
        "Examiner (Claude)"
    } else {
        "Host (Claude)"
    };

    let mut transcript_parts = Vec::new();
    let mut conversation_history: Vec<(String, String)> = Vec::new();

    if verbose {
        println!("============================================================");
        if is_socratic {
            println!("SOCRATIC EXAMINATION");
        } else {
            println!("ROUNDTABLE DISCUSSION");
        }
        println!("============================================================");
        println!();
    }

    // Phase 1: FRAMING / QUESTIONS
    let opening_label = if is_socratic { "Questions" } else { "Opening" };
    if verbose {
        println!("## {opening_label}
### {host_name}");
    }

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

    if verbose {
        println!("{host_framing}
");
    }
    transcript_parts.push(format!(
        "## {opening_label}

### {host_name}
{host_framing}"
    ));
    conversation_history.push((host_name.to_string(), host_framing.clone()));

    // Panelist opening takes / answers (parallel)
    if verbose {
        println!("(querying {} panelists in parallel...)", panelists.len());
    }

    let opening_system = if is_socratic {
        socratic_panelist_system("a panelist", 200)
    } else {
        discuss_panelist_system("a panelist", "the host", 150)
    };

    let opening_user = if is_socratic {
        format!(
            "Topic: {question}

The examiner asks:
{host_framing}

Answer each question directly.",
            host_framing = sanitize_speaker_content(&host_framing)
        )
    } else {
        format!(
            "Topic: {question}

The host opened with:
{host_framing}

Give your opening take.",
            host_framing = sanitize_speaker_content(&host_framing)
        )
    };

    let opening_messages = vec![Message::system(opening_system), Message::user(opening_user)];

    let opening_results = run_parallel(
        panelists,
        &opening_messages,
        api_key,
        google_api_key,
        500,
        Some(&cost_tracker),
        verbose,
    )
    .await;

    if is_socratic {
        transcript_parts.push("## Answers".to_string());
    }

    for (name, _, response) in opening_results {
        transcript_parts.push(format!("### {name}
{response}"));
        conversation_history.push((name, response));
    }

    if verbose {
        println!();
    }

    // Phase 2: DISCUSSION / PROBING
    let round_label = if is_socratic { "Probing Round" } else { "Round" };
    let history_label = if is_socratic {
        "Examination"
    } else {
        "Discussion"
    };
    let mut round_num = 0;

    let ctrl_c = signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        if rounds > 0 && round_num >= rounds {
            break;
        }

        // For rounds=0, check if Ctrl+C was pressed
        if rounds == 0 {
            tokio::select! {
                _ = &mut ctrl_c => {
                    if verbose {
                        println!("
(interrupted, wrapping up...)
");
                    }
                    break;
                }
                else => {}
            }
        }

        round_num += 1;

        if verbose {
            println!("## {round_label} {round_num}
");
        }
        transcript_parts.push(format!("## {round_label} {round_num}"));

        // Host steering / probing
        if verbose {
            println!("### {host_name}");
        }

        let steer_system = if is_socratic {
            socratic_host_probe()
        } else {
            discuss_host_steer()
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
            .join("

");

        let steer_messages = vec![
            Message::system(steer_system),
            Message::user(format!(
                "Topic: {question}

{history_label} so far:

{history_text}"
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

        if verbose {
            println!("{host_steer}
");
        }
        transcript_parts.push(format!("### {host_name}
{host_steer}"));
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

            let history_text = conversation_history
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

            let panelist_messages = vec![
                Message::system(panelist_system),
                Message::user(format!(
                    "Topic: {question}

{history_label} so far:

{history_text}

{follow_up_cue}"
                )),
            ];

            if verbose {
                println!("### {name}");
            }
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

            if verbose {
                println!("{response}
");
            }
            transcript_parts.push(format!("### {name}
{response}"));
            conversation_history.push((name.to_string(), response));
        }
    }

    // Phase 3: CLOSING
    if verbose {
        println!("## Closing
");
    }
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
        .join("

");

    let closing_messages = vec![
        Message::system(closing_prompt),
        Message::user(format!(
            "Topic: {question}

Full {history_label}:

{history_text}"
        )),
    ];

    if verbose {
        println!("(querying {} panelists in parallel...)", panelists.len());
    }

    let closing_results = run_parallel(
        panelists,
        &closing_messages,
        api_key,
        google_api_key,
        300,
        Some(&cost_tracker),
        verbose,
    )
    .await;

    for (name, _, response) in closing_results {
        transcript_parts.push(format!("### {name}
{response}"));
        conversation_history.push((name, response));
    }

    if verbose {
        println!();
    }

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
        .join("

");

    let closing_host_messages = vec![
        Message::system(closing_host_prompt),
        Message::user(format!(
            "Topic: {question}

Full {history_label}:

{history_text}"
        )),
    ];

    if is_socratic && verbose {
        println!("## Synthesis
");
        transcript_parts.push("## Synthesis".to_string());
    }

    if verbose {
        println!("### {host_name}");
    }

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

    if verbose {
        println!("{host_closing}
");
    }
    transcript_parts.push(format!("### {host_name}
{host_closing}"));

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
