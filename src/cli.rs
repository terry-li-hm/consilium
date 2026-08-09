//! Command-line argument parsing via clap derive.

use crate::config::SearchEngine;
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

    /// Read question from file instead of positional arg (avoids shell quoting issues)
    #[arg(long, value_name = "FILE", help_heading = "Context")]
    pub prompt_file: Option<std::path::PathBuf>,

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

    /// Pre-mortem: assume failure, work backward
    #[arg(long, help_heading = "Core Modes")]
    pub premortem: bool,

    /// Superforecasting: probability estimates + reconciliation
    #[arg(long, help_heading = "Core Modes")]
    pub forecast: bool,

    /// Socratic probing (examiner mode)
    #[arg(long, help_heading = "Core Modes")]
    pub socratic: bool,

    /// Oxford-style binary debate
    #[arg(long, help_heading = "Workflow Presets")]
    pub oxford: bool,

    /// Deep mode (decompose + 2 rounds, falls through to council)
    #[arg(long, help_heading = "Workflow Presets")]
    pub deep: bool,

    /// Additional context to include with the question
    #[arg(long, help_heading = "Context")]
    pub context: Option<String>,

    /// Persona for the council (e.g., "startup founder")
    #[arg(long, help_heading = "Context")]
    pub persona: Option<String>,

    /// Domain context (banking, healthcare, eu, fintech, bio)
    #[arg(long, help_heading = "Context")]
    pub domain: Option<String>,

    /// Explicit challenger model name (council mode)
    #[arg(long, help_heading = "Deliberation")]
    pub challenger: Option<String>,

    /// Number of debate/discussion rounds
    #[arg(long, default_value = "1", help_heading = "Deliberation")]
    pub rounds: usize,

    /// Decompose into sub-questions before deliberation
    #[arg(long, help_heading = "Deliberation")]
    pub decompose: bool,

    /// Enable cross-pollination phase (council mode)
    #[arg(long, help_heading = "Deliberation")]
    pub xpol: bool,

    /// Enable followup discussion after judge synthesis
    #[arg(long, help_heading = "Deliberation")]
    pub followup: bool,

    /// Reasoning effort for thinking models: low, medium, high
    #[arg(long, help_heading = "Deliberation")]
    pub effort: Option<String>,

    /// Enable web search grounding via OpenRouter [default: 5 results, engine: exa]
    #[arg(long, help_heading = "Context", value_name = "N", default_missing_value = "5", num_args = 0..=1, value_parser = clap::value_parser!(u8).range(1..=255))]
    pub web_search: Option<u8>,

    /// Search engine for --web-search: exa (~$0.02/req), native (provider pricing, may cost significantly more), firecrawl, parallel
    #[arg(long, help_heading = "Context", value_name = "ENGINE", requires = "web_search")]
    pub web_engine: Option<SearchEngine>,

    /// Skip early consensus exit and context compression (full deliberation)
    #[arg(long, help_heading = "Deliberation")]
    pub thorough: bool,

    /// Output file path (overrides auto-save)
    #[arg(short, long, help_heading = "Output")]
    pub output: Option<String>,

    /// Output format: prose, json, yaml
    #[arg(long, default_value = "prose", help_heading = "Output")]
    pub format: String,

    /// Share session as secret GitHub gist, or (with --push) make web run public and print URL
    #[arg(long, help_heading = "Output")]
    pub share: bool,

    /// Push completed run to consilium.sh (requires CONSILIUM_API_KEY)
    #[arg(long, help_heading = "Output")]
    pub push: bool,

    /// Don't auto-save session
    #[arg(long, help_heading = "Output")]
    pub no_save: bool,

    /// Save session to Obsidian vault (~/notes/Councils/)
    #[arg(long, help_heading = "Output")]
    pub vault: bool,

    /// Quiet mode (minimal output)
    #[arg(short, long, help_heading = "Output")]
    pub quiet: bool,

    /// Stream raw tokens instead of compact participant summaries
    #[arg(long, default_value_t = false, help_heading = "Output")]
    pub stream: bool,

    /// Override the xAI direct model slug (e.g. grok-4.20-experimental-beta-0304-reasoning)
    #[arg(long, value_name = "SLUG", help_heading = "Models")]
    pub xai_model: Option<String>,

    /// Named Grok variant shortcut: beta, fast, multi, stable
    ///   beta   → grok-4.20-experimental-beta-0304-reasoning (default)
    ///   fast   → grok-4.20-experimental-beta-0304-non-reasoning
    ///   multi  → grok-4.20-multi-agent-experimental-beta-0304
    ///   stable → grok-4
    #[arg(long, value_name = "VARIANT", help_heading = "Models")]
    pub grok: Option<String>,

    /// Override the judge model: sonnet, opus, gemini, or full model ID
    #[arg(short = 'J', long, value_name = "MODEL", help_heading = "Models")]
    pub judge_model: Option<String>,

    /// Override the critic model: sonnet, opus, gemini, or full model ID
    #[arg(short = 'C', long, value_name = "MODEL", help_heading = "Models")]
    pub critic_model: Option<String>,

    /// Skip the critique step in council mode
    #[arg(long, help_heading = "Models")]
    pub no_critic: bool,

    /// API timeout in seconds
    #[arg(long, default_value = "300", help_heading = "Output")]
    pub timeout: f64,

    /// Show session statistics
    #[arg(long, help_heading = "Admin")]
    pub stats: bool,

    /// List recent sessions
    #[arg(long, help_heading = "Admin")]
    pub sessions: bool,

    /// Watch live session output
    #[arg(long, help_heading = "Admin")]
    pub watch: bool,

    /// TUI for live session viewing
    #[arg(long, help_heading = "Admin")]
    pub tui: bool,

    /// View a session by name or search term
    #[arg(long, help_heading = "Admin")]
    pub view: Option<String>,

    /// Search sessions by content
    #[arg(long, help_heading = "Admin")]
    pub search: Option<String>,

    /// Print version
    #[arg(long, help_heading = "Admin")]
    pub version_flag: bool,

    /// Run diagnostics (check API keys, connectivity, session directory)
    #[arg(long, help_heading = "Admin")]
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
        } else if self.premortem {
            Some("premortem")
        } else if self.forecast {
            Some("forecast")
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

    /// Resolve the xAI model slug from --xai-model or --grok shortcut.
    /// Returns None if neither is set (env var / compiled default applies).
    pub fn resolve_xai_model(&self) -> Option<String> {
        if let Some(slug) = &self.xai_model {
            return Some(slug.clone());
        }
        self.grok.as_deref().map(|variant| {
            match variant.to_lowercase().as_str() {
                "fast" | "nr" | "non-reasoning" => {
                    "grok-4.20-experimental-beta-0304-non-reasoning".to_string()
                }
                "multi" | "research" => "grok-4.20-multi-agent-experimental-beta-0304".to_string(),
                "stable" | "4" => "grok-4".to_string(),
                _ => "grok-4.20-experimental-beta-0304-reasoning".to_string(), // beta / reasoning / default
            }
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_override_flags_parse() {
        let cli = Cli::parse_from([
            "consilium",
            "Test question",
            "--judge-model",
            "sonnet",
            "--critic-model",
            "gemini",
            "--no-critic",
        ]);

        assert_eq!(cli.judge_model.as_deref(), Some("sonnet"));
        assert_eq!(cli.critic_model.as_deref(), Some("gemini"));
        assert!(cli.no_critic);
    }
}
