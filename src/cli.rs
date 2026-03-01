//! Command-line argument parsing via clap derive.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "consilium",
    about = "Multi-model deliberation CLI",
    version,
    after_help = "Examples:\n  consilium \"Should I take this job offer?\"\n  consilium \"What could go wrong with our launch plan?\" --redteam\n  consilium \"Is Rust better than Go for CLIs?\" --oxford\n  consilium \"What is the best database for this use case?\" --quick"
)]
pub struct Cli {
    /// The question or topic to deliberate on
    pub question: Option<String>,

    // --- Mode flags (mutually exclusive by convention, validated in main) ---
    /// Quick parallel query (all models answer independently)
    #[arg(long, help_heading = "Core Modes")]
    pub quick: bool,

    /// Full council deliberation (blind → debate → judge)
    #[arg(long, help_heading = "Core Modes")]
    pub council: bool,

    /// Roundtable discussion (hosted exploration)
    #[arg(long, help_heading = "Workflow Presets")]
    pub discuss: bool,

    /// Adversarial red team stress test
    #[arg(long, help_heading = "Core Modes")]
    pub redteam: bool,

    /// Socratic probing (examiner mode)
    #[arg(long, help_heading = "Core Modes")]
    pub socratic: bool,

    /// Oxford-style binary debate
    #[arg(long, help_heading = "Workflow Presets")]
    pub oxford: bool,

    /// Deep mode (decompose + 2 rounds, falls through to council)
    #[arg(long, help_heading = "Workflow Presets")]
    pub deep: bool,

    // --- Content flags ---
    /// Additional context to include with the question
    #[arg(long)]
    pub context: Option<String>,

    /// Persona for the council (e.g., "startup founder")
    #[arg(long)]
    pub persona: Option<String>,

    /// Domain context (banking, healthcare, eu, fintech, bio)
    #[arg(long)]
    pub domain: Option<String>,

    /// Explicit challenger model name (council mode)
    #[arg(long)]
    pub challenger: Option<String>,

    /// Number of debate/discussion rounds
    #[arg(long, default_value = "1")]
    pub rounds: usize,

    // --- Session management ---
    /// Output file path (overrides auto-save)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Share session as secret GitHub gist
    #[arg(long)]
    pub share: bool,

    /// Don't auto-save session
    #[arg(long)]
    pub no_save: bool,

    /// Quiet mode (minimal output)
    #[arg(short, long)]
    pub quiet: bool,

    /// API timeout in seconds
    #[arg(long, default_value = "300")]
    pub timeout: f64,

    // --- Transformations ---
    /// Decompose into sub-questions before deliberation
    #[arg(long)]
    pub decompose: bool,

    /// Enable cross-pollination phase (council mode)
    #[arg(long)]
    pub xpol: bool,

    /// Enable followup discussion after judge synthesis
    #[arg(long)]
    pub followup: bool,

    /// Output format: prose, json, yaml
    #[arg(long, default_value = "prose")]
    pub format: String,

    // --- Admin commands ---
    /// Show session statistics
    #[arg(long)]
    pub stats: bool,

    /// List recent sessions
    #[arg(long)]
    pub sessions: bool,

    /// Watch live session output
    #[arg(long)]
    pub watch: bool,

    /// TUI for live session viewing
    #[arg(long)]
    pub tui: bool,

    /// View a session by name or search term
    #[arg(long)]
    pub view: Option<String>,

    /// Search sessions by content
    #[arg(long)]
    pub search: Option<String>,

    /// Print version
    #[arg(long)]
    pub version_flag: bool,

    /// Skip early consensus exit and context compression (full deliberation)
    #[arg(long)]
    pub thorough: bool,

    /// Run diagnostics (check API keys, connectivity, session directory)
    #[arg(long)]
    pub doctor: bool,
}

impl Cli {
    /// Returns the explicitly selected mode, or None for auto-routing.
    pub fn explicit_mode(&self) -> Option<&'static str> {
        if self.quick {
            Some("quick")
        } else if self.council {
            Some("council")
        } else if self.discuss {
            Some("discuss")
        } else if self.redteam {
            Some("redteam")
        } else if self.socratic {
            Some("socratic")
        } else if self.oxford {
            Some("oxford")
        } else if self.deep {
            Some("council") // deep falls through to council
        } else {
            None
        }
    }

    /// Check if any admin command is requested (no question needed).
    pub fn is_admin_command(&self) -> bool {
        self.stats
            || self.sessions
            || self.watch
            || self.tui
            || self.view.is_some()
            || self.search.is_some()
            || self.version_flag
            || self.doctor
    }
}
