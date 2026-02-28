use clap::Parser;
use consilium::cli::Cli;
use consilium::config::{
    discuss_models, oxford_models, quick_models, redteam_models, JUDGE_MODEL,
};
use consilium::modes::{discuss, oxford, quick, redteam, solo};
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
            eprintln!("Council mode not yet implemented in this phase.");
            std::process::exit(1);
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
