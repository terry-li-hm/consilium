use clap::Parser;
use consilium::admin;
use consilium::api::classify_mode;
use consilium::cli::Cli;
use consilium::config::{
    discuss_models, oxford_models, quick_models, redteam_models, CostTracker, COUNCIL,
};
use consilium::modes::{council, discuss, oxford, quick, redteam};
use consilium::session::{finish_session, setup_live_output};
use std::io::IsTerminal;

#[tokio::main]
async fn main() {
    let args = Cli::parse();
    let effective_quiet = args.quiet || !std::io::stdout().is_terminal();

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

    let color = std::env::var("NO_COLOR").is_err() && std::io::stdout().is_terminal();
    let mut output = setup_live_output(effective_quiet, color);

    let result = match mode.as_str() {
        "quick" => {
            quick::run_quick(
                &question,
                &quick_models(),
                &api_key,
                google_api_key.as_deref(),
                &mut *output,
                &args.format,
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
                None,
                &args.format,
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
                &args.format,
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
                &args.format,
                args.timeout,
                &mut *output,
                args.thorough,
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
                &args.format,
                true,
                true,
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

    finish_session(
        &question,
        &result,
        &mode,
        "",
        args.no_save,
        args.output.as_deref(),
        args.share,
        effective_quiet,
        None,
    );

    std::process::exit(0);
}
