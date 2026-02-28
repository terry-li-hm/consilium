//! Session save/share/history utilities.

use crate::config::SessionResult;
use chrono::Local;
use regex::Regex;
use serde_json::{Map, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

fn mode_title(mode: &str) -> &'static str {
    match mode {
        "quick" => "Quick Council",
        "discuss" => "Roundtable Discussion",
        "redteam" => "Red Team",
        "socratic" => "Socratic Examination",
        "oxford" => "Oxford Debate",
        "solo" => "Solo Council",
        "council" => "Council Deliberation",
        "deep" => "Deep Council",
        _ => "Session",
    }
}

pub fn get_sessions_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let sessions_dir = home.join(".consilium").join("sessions");
    let _ = fs::create_dir_all(&sessions_dir);
    sessions_dir
}

pub fn slugify(text: &str, max_len: usize) -> String {
    static NON_WORD: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"[^\w\s-]").expect("valid regex"));
    static SEPS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s_-]+").expect("valid regex"));

    let lower = text.to_lowercase();
    let cleaned = NON_WORD.replace_all(&lower, "");
    let collapsed = SEPS.replace_all(&cleaned, "-");
    let mut slug: String = collapsed.chars().take(max_len).collect();
    while slug.ends_with('-') {
        slug.pop();
    }
    slug
}

pub fn save_session(
    question: &str,
    transcript: &str,
    mode: &str,
    header_extra: &str,
    no_save: bool,
    output: Option<&str>,
    quiet: bool,
) -> Option<PathBuf> {
    if let Some(output_path) = output {
        if let Err(e) = fs::write(output_path, transcript) {
            eprintln!("Failed to save transcript to {output_path}: {e}");
        } else if !quiet {
            println!("Transcript saved to: {output_path}");
        }
    }

    if no_save {
        return None;
    }

    let now = Local::now();
    let timestamp = now.format("%Y%m%d-%H%M%S").to_string();
    let slug = slugify(question, 40);
    let filename = if mode == "council" {
        format!("{timestamp}-{slug}.md")
    } else {
        format!("{timestamp}-{mode}-{slug}.md")
    };
    let session_path = get_sessions_dir().join(filename);

    let session_content = format!(
        "# {}\n\n**Question:** {}\n**Date:** {}\n**Mode:** {}\n{}\n---\n\n{}\n",
        mode_title(mode),
        question,
        now.format("%Y-%m-%d %H:%M"),
        mode,
        header_extra,
        transcript
    );

    match fs::write(&session_path, session_content) {
        Ok(_) => {
            if !quiet {
                println!("Session saved to: {}", session_path.display());
            }
            Some(session_path)
        }
        Err(e) => {
            eprintln!("Failed to save session: {e}");
            None
        }
    }
}

pub fn share_gist(question: &str, transcript: &str, mode: &str, quiet: bool) -> Option<String> {
    let title = mode_title(mode);
    let now = Local::now();

    let temp_path = std::env::temp_dir().join(format!(
        "council-{mode}-{}-{}.md",
        std::process::id(),
        now.timestamp_millis()
    ));

    let gist_content = format!(
        "# {}\n\n**Question:** {}\n\n**Date:** {}\n\n---\n\n{}",
        title,
        question,
        now.format("%Y-%m-%d %H:%M"),
        transcript
    );

    if let Err(e) = fs::write(&temp_path, gist_content) {
        eprintln!("Gist temp file creation failed: {e}");
        return None;
    }

    let output = Command::new("gh")
        .arg("gist")
        .arg("create")
        .arg(&temp_path)
        .arg("--desc")
        .arg(format!("{title}: {}", question.chars().take(50).collect::<String>()))
        .output();

    let _ = fs::remove_file(&temp_path);

    match output {
        Ok(result) if result.status.success() => {
            let gist_url = String::from_utf8_lossy(&result.stdout).trim().to_string();
            if !quiet {
                println!("\nShared: {gist_url}");
            }
            Some(gist_url)
        }
        Ok(result) => {
            let err = String::from_utf8_lossy(&result.stderr);
            eprintln!("Gist creation failed: {err}");
            None
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                eprintln!("Error: 'gh' CLI not found. Install with: brew install gh");
            } else {
                eprintln!("Gist creation failed: {e}");
            }
            None
        }
    }
}

pub fn log_history(
    question: &str,
    mode: &str,
    session_path: Option<&Path>,
    gist_url: Option<&str>,
    extra: Option<Map<String, Value>>,
    cost: Option<f64>,
    duration: Option<f64>,
) {
    let history_file = get_sessions_dir()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("history.jsonl");

    let mut entry = Map::new();
    entry.insert(
        "timestamp".into(),
        Value::String(Local::now().format("%Y-%m-%dT%H:%M:%S%.f").to_string()),
    );
    entry.insert(
        "question".into(),
        Value::String(question.chars().take(200).collect()),
    );
    entry.insert("mode".into(), Value::String(mode.to_string()));
    entry.insert(
        "cost".into(),
        cost.map_or(Value::Null, |v| Value::from(((v * 10_000.0).round()) / 10_000.0)),
    );
    entry.insert(
        "duration".into(),
        duration.map_or(Value::Null, |v| Value::from(((v * 10.0).round()) / 10.0)),
    );
    entry.insert(
        "session".into(),
        session_path
            .map(|p| Value::String(p.display().to_string()))
            .unwrap_or(Value::Null),
    );
    entry.insert(
        "gist".into(),
        gist_url
            .map(|g| Value::String(g.to_string()))
            .unwrap_or(Value::Null),
    );

    if let Some(extra) = extra {
        for (k, v) in extra {
            entry.insert(k, v);
        }
    }

    let line = serde_json::to_string(&Value::Object(entry));
    match line {
        Ok(serialized) => {
            let mut file = match fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(history_file)
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to open history file: {e}");
                    return;
                }
            };
            let _ = writeln!(file, "{serialized}");
        }
        Err(e) => eprintln!("Failed to serialize history entry: {e}"),
    }
}

pub fn finish_session(
    question: &str,
    result: &SessionResult,
    mode: &str,
    header_extra: &str,
    no_save: bool,
    output: Option<&str>,
    share: bool,
    quiet: bool,
    history_extra: Option<Map<String, Value>>,
) -> Option<PathBuf> {
    let session_path = save_session(
        question,
        &result.transcript,
        mode,
        header_extra,
        no_save,
        output,
        quiet,
    );

    let gist_url = if share {
        share_gist(question, &result.transcript, mode, quiet)
    } else {
        None
    };

    let mut merged_extra = history_extra.unwrap_or_default();
    if let Some(failures) = &result.failures {
        merged_extra.insert(
            "failures".into(),
            Value::Array(failures.iter().map(|f| Value::String(f.clone())).collect()),
        );
    }

    log_history(
        question,
        mode,
        session_path.as_deref(),
        gist_url.as_deref(),
        if merged_extra.is_empty() {
            None
        } else {
            Some(merged_extra)
        },
        Some(result.cost),
        Some(result.duration),
    );

    session_path
}
