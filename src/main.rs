use clap::Parser;
use consilium::cli::Cli;
use consilium::config::quick_models;
use consilium::modes::run_quick;
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

    if args.quick {
        let models = quick_models();
        let result = run_quick(
            &question,
            &models,
            &api_key,
            google_api_key.as_deref(),
            !args.quiet,
            &args.format,
            args.timeout,
        )
        .await;

        finish_session(
            &question,
            &result,
            "quick",
            "",
            args.no_save,
            args.output.as_deref(),
            args.share,
            args.quiet,
            None,
        );

        std::process::exit(0);
    }

    let mode = args.explicit_mode().unwrap_or("council");
    eprintln!("Mode '{mode}' for question: {question}");
    eprintln!("Not yet implemented — quick mode is available in this phase.");
}
