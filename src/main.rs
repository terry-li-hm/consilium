use clap::Parser;
use consilium::admin;
use consilium::api::classify_mode;
use consilium::cli::Cli;
use consilium::config::{
    discuss_models, oxford_models, quick_models, redteam_models, CostTracker, COUNCIL, JUDGE_MODEL,
};
use consilium::modes::{council, discuss, oxford, quick, redteam, solo};
use consilium::session::{append_feedback_to_history, finish_session, setup_live_output};
use serde_json::Value;
use std::io::IsTerminal;
use std::path::PathBuf;

fn supports_json_format(mode: &str) -> bool {
    matches!(mode, "council" | "quick")
}

fn parse_structured_payload(transcript: &str) -> Option<Value> {
    let trimmed = transcript.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }

    let parts: Vec<&str> = trimmed.split("\n\n---\n\n").collect();
    for part in parts.iter().rev() {
        let candidate = part.trim();
        if candidate.starts_with('{') && candidate.ends_with('}') {
            if let Ok(value) = serde_json::from_str::<Value>(candidate) {
                return Some(value);
            }
        }
    }

    None
}

fn compact_home_path(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home_str = home.to_string_lossy();
        if path == home_str {
            return "~".to_string();
        }
        let prefix = format!("{home_str}/");
        if path.starts_with(&prefix) {
            return path.replacen(&prefix, "~/", 1);
        }
    }
    path.to_string()
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();
    let cc_mode = args.cc;
    let effective_quiet = args.quiet || cc_mode;

    if args.version_flag {
        println!("consilium {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }

    if args.stats {
        admin::show_stats();
        std::process::exit(0);
    }

    if args.sessions {
        admin::list_sessions();
        std::process::exit(0);
    }

    if let Some(term) = args.view.as_deref() {
        admin::view_session(Some(term));
        std::process::exit(0);
    } else if args.view.is_some() {
        // Handle --view with no arg if allowed by parser, but Option<String> usually needs arg or another flag
        admin::view_session(None);
        std::process::exit(0);
    }

    if let Some(term) = args.search.as_deref() {
        admin::search_sessions(term);
        std::process::exit(0);
    }

    if args.list_roles {
        admin::list_roles();
        std::process::exit(0);
    }

    if args.tui {
        if let Err(e) = consilium::tui::run_tui() {
            eprintln!("TUI error: {e}");
            std::process::exit(1);
        }
        std::process::exit(0);
    }

    if args.watch {
        if let Err(e) = consilium::watch::watch_live() {
            eprintln!("Watch error: {e}");
            std::process::exit(1);
        }
        std::process::exit(0);
    }

    if args.doctor {
        let doctor_client = reqwest::Client::new();
        admin::run_doctor(&doctor_client).await;
        std::process::exit(0);
    }

    let question = match &args.question {
        Some(q) => q.clone(),
        None => {
            eprintln!("Error: question is required (or use an admin command like --stats)");
            std::process::exit(1);
        }
    };

    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            eprintln!("Error: OPENROUTER_API_KEY environment variable not set");
            std::process::exit(1);
        }
    };
    let google_api_key = std::env::var("GOOGLE_API_KEY").ok();

    let client = reqwest::Client::new();
    let mode = if let Some(explicit) = args.explicit_mode() {
        explicit.to_string()
    } else {
        if !effective_quiet {
            println!("Classifying question...");
        }
        let auto_mode = classify_mode(&client, &api_key, &question, None).await;
        if !effective_quiet {
            println!("Mode: {auto_mode}");
            println!();
        }
        auto_mode
    };

    let effective_format = if cc_mode && supports_json_format(&mode) {
        "json".to_string()
    } else {
        args.format.clone()
    };

    let effective_output: Option<String> = if let Some(path) = args.output.clone() {
        Some(path)
    } else if cc_mode {
        let default_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".consilium")
            .join("cc-latest.md");
        if let Some(parent) = default_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Some(default_path.to_string_lossy().to_string())
    } else {
        None
    };

    let color = !args.no_color
        && std::env::var("NO_COLOR").is_err()
        && std::io::stdout().is_terminal();
    let mut output = setup_live_output(effective_quiet, color);

    let result = match mode.as_str() {
        "quick" => {
            quick::run_quick(
                &question,
                &quick_models(),
                &api_key,
                google_api_key.as_deref(),
                &mut *output,
                &effective_format,
                args.timeout,
            )
            .await
        }
        "oxford" => {
            oxford::run_oxford(
                &question,
                &oxford_models(),
                &api_key,
                google_api_key.as_deref(),
                args.motion.clone(),
                &effective_format,
                args.timeout,
                &mut *output,
            )
            .await
        }
        "redteam" => {
            redteam::run_redteam(
                &question,
                &redteam_models(),
                &api_key,
                google_api_key.as_deref(),
                args.context.clone(),
                &effective_format,
                args.timeout,
                &mut *output,
            )
            .await
        }
        "discuss" | "socratic" => {
            discuss::run_discuss(
                &question,
                &discuss_models(),
                &api_key,
                google_api_key.as_deref(),
                &mode,
                args.rounds as u32,
                args.context.clone(),
                &effective_format,
                args.timeout,
                &mut *output,
                args.thorough,
            )
            .await
        }
        "solo" => {
            solo::run_solo(
                &question,
                JUDGE_MODEL,
                &api_key,
                google_api_key.as_deref(),
                args.roles.clone(),
                &effective_format,
                args.timeout,
                &mut *output,
            )
            .await
        }
        "council" => {
            let rounds = if args.deep {
                args.rounds.max(2)
            } else {
                args.rounds
            };
            let should_decompose = args.decompose || args.deep;

            let sub_questions = if should_decompose {
                let decompose_cost = CostTracker::new();
                Some(
                    council::decompose_question(&question, &api_key, &mut *output, &decompose_cost)
                        .await,
                )
            } else {
                None
            };

            let challenger_idx = args.challenger.as_ref().and_then(|target| {
                let target = target.to_lowercase();
                COUNCIL.iter().position(|(name, model, _)| {
                    name.to_lowercase() == target
                        || model.split('/').next_back().unwrap_or(model).to_lowercase() == target
                        || model.to_lowercase() == target
                })
            });

            council::run_council(
                &question,
                COUNCIL,
                &api_key,
                google_api_key.as_deref(),
                rounds,
                &mut *output,
                true,
                true,
                args.context.as_deref(),
                args.persona.as_deref(),
                args.domain.as_deref(),
                challenger_idx,
                &effective_format,
                !args.no_judge,
                !args.no_judge,
                sub_questions,
                args.xpol,
                args.followup,
                args.thorough,
            )
            .await
        }
        _ => {
            eprintln!("Mode '{mode}' not implemented.");
            std::process::exit(1);
        }
    };

    let session_path = finish_session(
        &question,
        &result,
        &mode,
        "",
        args.no_save,
        effective_output.as_deref(),
        args.share,
        effective_quiet,
        None,
    );

    if cc_mode {
        if supports_json_format(&mode) {
            let structured = parse_structured_payload(&result.transcript);
            let decision = structured
                .as_ref()
                .and_then(|v| v.get("decision"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("See saved JSON output for details");
            let confidence = structured
                .as_ref()
                .and_then(|v| v.get("confidence"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("n/a");
            println!(
                "[DECISION] {} (confidence: {}, cost: ${:.2})",
                decision, confidence, result.cost
            );
        } else {
            let output_path = session_path
                .as_ref()
                .map(|p| p.display().to_string())
                .or_else(|| effective_output.clone())
                .unwrap_or_else(|| "<not-saved>".to_string());
            println!(
                "[DONE] Session saved to {} (${:.2})",
                compact_home_path(&output_path),
                result.cost
            );
        }
    }

    if args.feedback {
        eprint!("Rate this session (1-5): ");
        let mut input = String::new();
        use std::io::BufRead;
        if std::io::stdin().lock().read_line(&mut input).is_ok() {
            if let Ok(rating) = input.trim().parse::<u8>() {
                if (1..=5).contains(&rating) {
                    append_feedback_to_history(rating);
                }
            }
        }
    }

    std::process::exit(0);
}
