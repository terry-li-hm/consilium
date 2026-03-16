use clap::Parser;
use consilium::admin;
use consilium::api::classify_mode;
use consilium::cli::Cli;
use consilium::config::{
    discuss_models, oxford_models, quick_models, redteam_models, resolved_council, CostTracker,
    QueryOptions, ReasoningEffort, WebSearchConfig,
};
use consilium::modes::{council, discuss, forecast, oxford, premortem, quick, redteam};
use consilium::session::{
    finish_session, prepare_live_session_path, push_to_web, setup_live_output, AgentOutput,
    HumanOutput,
};
use std::io::IsTerminal;
use std::os::unix::io::AsRawFd;

/// Returns true when stdout is redirected to a regular file (background capture).
/// Distinguishes file redirect (wants full output) from pipe (scripting, compact fine).
fn stdout_is_file_redirect() -> bool {
    let fd = std::io::stdout().as_raw_fd();
    unsafe {
        let mut stat: libc::stat = std::mem::zeroed();
        libc::fstat(fd, &mut stat) == 0 && (stat.st_mode & libc::S_IFMT) == libc::S_IFREG
    }
}

/// Returns true when stdout is a pipe (e.g. CC background task).
/// Pipe output can overflow or truncate — suppress stdout, rely on session auto-save.
fn stdout_is_pipe() -> bool {
    let fd = std::io::stdout().as_raw_fd();
    unsafe {
        let mut stat: libc::stat = std::mem::zeroed();
        libc::fstat(fd, &mut stat) == 0 && (stat.st_mode & libc::S_IFMT) == libc::S_IFIFO
    }
}

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
        admin::doctor();
        std::process::exit(0);
    }

    let question = match (&args.question, &args.prompt_file) {
        (Some(q), _) => q.clone(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .unwrap_or_else(|e| {
                eprintln!(
                    "Error: could not read prompt file {}: {}",
                    path.display(),
                    e
                );
                std::process::exit(1);
            })
            .trim()
            .to_string(),
        (None, None) => {
            eprintln!(
                "Error: question is required (or use --prompt-file FILE, or use an admin command like --stats)"
            );
            std::process::exit(1);
        }
    };

    if question.is_empty() {
        eprintln!("Error: question is empty (check --prompt-file contents or positional arg)");
        std::process::exit(1);
    }

    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            eprintln!("Error: OPENROUTER_API_KEY environment variable not set");
            std::process::exit(1);
        }
    };
    let google_api_key = std::env::var("GOOGLE_API_KEY").ok();
    let zhipu_api_key = std::env::var("ZHIPU_API_KEY").ok();
    let moonshot_api_key = std::env::var("MOONSHOT_API_KEY").ok();
    let xai_api_key = std::env::var("XAI_API_KEY").ok();
    let openai_api_key = std::env::var("OPENAI_API_KEY").ok();
    let anthropic_api_key = std::env::var("ANTHROPIC_API_KEY").ok();

    // Apply --xai-model / --grok shortcut before resolved_council() is called
    if let Some(slug) = args.resolve_xai_model() {
        std::env::set_var("CONSILIUM_XAI_MODEL", slug);
    }

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
    let effort = args.effort.as_deref().and_then(ReasoningEffort::from_str);
    let web_search = args.web_search.map(|max_results| WebSearchConfig {
        max_results,
        engine: args.web_engine.unwrap_or_default(),
    });
    let opts = QueryOptions::new(effort, web_search);

    let color = std::env::var("NO_COLOR").is_err() && std::io::stdout().is_terminal();
    let piped = stdout_is_pipe();
    let mut output: Box<dyn consilium::session::Output> = if stdout_is_file_redirect() {
        // Stdout redirected to a file (background capture) — always stream full output
        // regardless of --quiet so TaskOutput shows progress and the task doesn't look stuck.
        Box::new(AgentOutput::new(None, false))
    } else if piped {
        // Stdout is a pipe (e.g. CC background task) — pipe buffers overflow on long
        // deliberations. Suppress stdout; session auto-saves to disk. Print path at end.
        Box::new(consilium::session::SilentOutput)
    } else if args.stream || args.quiet {
        setup_live_output(effective_quiet, color)
    } else {
        let session_file_path = prepare_live_session_path();
        Box::new(HumanOutput::new(&session_file_path, color))
    };

    let result = match mode.as_str() {
        "quick" => {
            quick::run_quick(
                &question,
                args.context.as_deref(),
                &quick_models(),
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                &mut *output,
                &args.format,
                args.timeout,
                &opts,
            )
            .await
        }
        "oxford" => {
            oxford::run_oxford(
                &question,
                &oxford_models(),
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                None,
                &args.format,
                args.timeout,
                &mut *output,
                &opts,
            )
            .await
        }
        "redteam" => {
            redteam::run_redteam(
                &question,
                &redteam_models(),
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                args.context.clone(),
                &args.format,
                args.timeout,
                &mut *output,
                &opts,
            )
            .await
        }
        "premortem" => {
            premortem::run_premortem(
                &question,
                &redteam_models(),
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                args.context.clone(),
                &args.format,
                args.timeout,
                &mut *output,
                &opts,
            )
            .await
        }
        "forecast" => {
            forecast::run_forecast(
                &question,
                &redteam_models(),
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                args.context.clone(),
                &args.format,
                args.timeout,
                &mut *output,
                &opts,
            )
            .await
        }
        "discuss" | "socratic" => {
            discuss::run_discuss(
                &question,
                &discuss_models(),
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                &mode,
                args.rounds as u32,
                args.context.clone(),
                &args.format,
                args.timeout,
                &mut *output,
                args.thorough,
                &opts,
            )
            .await
        }
        "council" => {
            let council = resolved_council();
            let rounds = if args.deep {
                args.rounds.max(2)
            } else {
                args.rounds
            };
            let should_decompose = args.decompose || args.deep;

            let sub_questions = if should_decompose {
                let decompose_cost = CostTracker::new();
                Some(
                    council::decompose_question(
                        &question,
                        &api_key,
                        args.judge_model.as_deref(),
                        &mut *output,
                        &decompose_cost,
                    )
                    .await,
                )
            } else {
                None
            };

            let challenger_idx = args.challenger.as_ref().and_then(|target| {
                let target = target.to_lowercase();
                council.iter().position(|(name, model, _)| {
                    name.to_lowercase() == target
                        || model.split('/').next_back().unwrap_or(model).to_lowercase() == target
                        || model.to_lowercase() == target
                })
            });

            council::run_council(
                &question,
                &council,
                &api_key,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                rounds,
                args.timeout,
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
                &opts,
                args.judge_model.as_deref(),
                args.critic_model.as_deref(),
                args.no_critic,
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
        args.vault || args.deep || mode == "premortem" || mode == "deep",
        args.output.as_deref(),
        args.share && !args.push, // gist share only when not using web push
        effective_quiet,
        result.extra.clone(),
    );

    // When piped (e.g. CC background task), stdout was suppressed to avoid pipe
    // buffer overflow. Print just the session path so the caller can `cat` it.
    if piped {
        if let Some(path) = &session_path {
            println!("{}", path.display());
        }
    }

    if args.push {
        // Build a minimal run payload compatible with the consilium.sh CLI API
        let run_id = uuid::Uuid::new_v4().to_string();
        let run_json = serde_json::json!({
            "id": run_id,
            "question": question,
            "mode": mode,
            "phase": "done",
            "transcript": result.transcript,
            "cost": result.cost,
            "duration": result.duration,
        });
        match push_to_web(&run_json, args.share).await {
            Ok(Some(url)) => println!("\nShared: {url}"),
            Ok(None) => {
                if !effective_quiet {
                    println!("\nPushed to consilium.sh");
                }
            }
            Err(e) => eprintln!("Push failed: {e}"),
        }
    }

    std::process::exit(0);
}
