use clap::Parser;
use consilium::cli::Cli;
use consilium::config::{
    discuss_models, oxford_models, quick_models, redteam_models, CostTracker, COUNCIL, JUDGE_MODEL,
};
use consilium::modes::{council, discuss, oxford, quick, redteam, solo};
use consilium::session::finish_session;

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    if args.is_admin_command() {
        // TODO: admin command dispatch
        eprintln!("Admin commands not yet implemented");
        std::process::exit(1);
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

    let mode = args.explicit_mode().unwrap_or("council");

    let result = match mode {
        "quick" => {
            quick::run_quick(
                &question,
                &quick_models(),
                &api_key,
                google_api_key.as_deref(),
                !args.quiet,
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
                args.motion.clone(),
                &args.format,
                args.timeout,
                args.quiet,
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
                args.quiet,
            )
            .await
        }
        "discuss" | "socratic" => {
            discuss::run_discuss(
                &question,
                &discuss_models(),
                &api_key,
                google_api_key.as_deref(),
                mode,
                args.rounds as u32,
                args.context.clone(),
                &args.format,
                args.timeout,
                args.quiet,
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
                &args.format,
                args.timeout,
                args.quiet,
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
                    council::decompose_question(&question, &api_key, !args.quiet, &decompose_cost)
                        .await,
                )
            } else {
                None
            };

            let challenger_idx = args.challenger.as_ref().and_then(|target| {
                let target = target.to_lowercase();
                COUNCIL.iter().position(|(name, model, _)| {
                    name.to_lowercase() == target
                        || model.split('/').last().unwrap_or(model).to_lowercase() == target
                        || model.to_lowercase() == target
                })
            });

            council::run_council(
                &question,
                COUNCIL,
                &api_key,
                google_api_key.as_deref(),
                rounds,
                !args.quiet,
                true,
                true,
                args.context.as_deref(),
                args.persona.as_deref(),
                args.domain.as_deref(),
                challenger_idx,
                &args.format,
                !args.no_judge,
                !args.no_judge,
                sub_questions,
                args.xpol,
                args.followup,
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
        mode,
        "",
        args.no_save,
        args.output.as_deref(),
        args.share,
        args.quiet,
        None,
    );

    std::process::exit(0);
}
