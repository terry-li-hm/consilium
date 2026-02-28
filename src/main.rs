use clap::Parser;
use consilium::cli::Cli;

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

    let mode = args.explicit_mode().unwrap_or("council");

    // TODO: mode dispatch
    eprintln!("Mode '{mode}' for question: {question}");
    eprintln!("Not yet implemented — see phases 2-6");
}
