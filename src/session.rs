//! Session save/share/history utilities.

use crate::config::SessionResult;
use crate::watch::{classify, LineType};
use chrono::Local;
use crossterm::style::{Attribute, Color, Stylize};
use regex::Regex;
use serde_json::{Map, Value};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

#[cfg(unix)]
use libc;

pub trait Output: Send + Sync {
    fn write_str(&mut self, s: &str) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
}

pub struct NullOutput;

impl Output for NullOutput {
    fn write_str(&mut self, _s: &str) -> io::Result<()> {
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn render_colored_line(
    out: &mut impl Write,
    line: &str,
    prev_type: Option<LineType>,
) -> io::Result<()> {
    let line_type = classify(line, prev_type);
    let stripped = line.trim();

    match line_type {
        LineType::Separator => {
            writeln!(out, "{}", "─".repeat(60).with(Color::DarkGrey))?;
        }
        LineType::PhaseBanner => {
            writeln!(
                out,
                "{}",
                format!(" {stripped} ")
                    .with(Color::Cyan)
                    .attribute(Attribute::Bold)
            )?;
        }
        LineType::ModelHeader => {
            writeln!(
                out,
                "{}",
                stripped
                    .trim_start_matches("### ")
                    .with(Color::Yellow)
                    .attribute(Attribute::Bold)
            )?;
        }
        LineType::SectionHeader => {
            writeln!(
                out,
                "{}",
                stripped
                    .trim_start_matches("## ")
                    .with(Color::Blue)
                    .attribute(Attribute::Bold)
            )?;
        }
        LineType::Notice => {
            writeln!(
                out,
                "{}",
                stripped.with(Color::Green).attribute(Attribute::Bold)
            )?;
        }
        LineType::Status => {
            writeln!(
                out,
                "{}",
                stripped.with(Color::DarkGrey).attribute(Attribute::Italic)
            )?;
        }
        LineType::Confidence => {
            writeln!(
                out,
                "{}",
                stripped.with(Color::Magenta).attribute(Attribute::Bold)
            )?;
        }
        LineType::Stats => {
            writeln!(out, "{}", stripped.with(Color::DarkGrey))?;
        }
        LineType::Body => {
            writeln!(out, "{line}")?;
        }
    }

    out.flush()
}

pub struct StdoutOutput {
    color: bool,
    line_buf: String,
    prev_type: Option<LineType>,
    flushed_partial: bool,
}

impl StdoutOutput {
    pub fn new(color: bool) -> Self {
        Self {
            color,
            line_buf: String::new(),
            prev_type: None,
            flushed_partial: false,
        }
    }
}

impl Output for StdoutOutput {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        if !self.color {
            print!("{s}");
            return io::stdout().flush();
        }

        self.line_buf.push_str(s);
        let mut out = io::stdout();

        while let Some(newline_pos) = self.line_buf.find('\n') {
            let line = self.line_buf[..newline_pos].to_string();
            self.line_buf.drain(..=newline_pos);

            if self.flushed_partial {
                // Rest of a partial line — already printed prefix as plain text
                writeln!(out, "{line}")?;
                out.flush()?;
                self.flushed_partial = false;
            } else {
                let line_type = classify(&line, self.prev_type);
                render_colored_line(&mut out, &line, self.prev_type)?;
                self.prev_type = Some(line_type);
            }
        }

        // Flush any remaining partial line as plain text (streaming UX)
        if !self.line_buf.is_empty() && !self.flushed_partial {
            write!(out, "{}", self.line_buf)?;
            out.flush()?;
            self.flushed_partial = true;
        }

        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

pub struct TeeOutput {
    file: fs::File,
    color: bool,
    line_buf: String,
    prev_type: Option<LineType>,
    flushed_partial: bool,
}

impl TeeOutput {
    pub fn new(path: &Path, color: bool) -> io::Result<Self> {
        let file = fs::File::create(path)?;
        Ok(Self {
            file,
            color,
            line_buf: String::new(),
            prev_type: None,
            flushed_partial: false,
        })
    }
}

impl Output for TeeOutput {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        // File always gets plain text
        self.file.write_all(s.as_bytes())?;
        self.file.flush()?;

        if !self.color {
            print!("{s}");
            return io::stdout().flush();
        }

        self.line_buf.push_str(s);
        let mut out = io::stdout();

        while let Some(newline_pos) = self.line_buf.find('\n') {
            let line = self.line_buf[..newline_pos].to_string();
            self.line_buf.drain(..=newline_pos);

            if self.flushed_partial {
                writeln!(out, "{line}")?;
                out.flush()?;
                self.flushed_partial = false;
            } else {
                let line_type = classify(&line, self.prev_type);
                render_colored_line(&mut out, &line, self.prev_type)?;
                self.prev_type = Some(line_type);
            }
        }

        if !self.line_buf.is_empty() && !self.flushed_partial {
            write!(out, "{}", self.line_buf)?;
            out.flush()?;
            self.flushed_partial = true;
        }

        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        let _ = io::stdout().flush();
        self.file.flush()
    }
}

pub fn setup_live_output(quiet: bool, color: bool) -> Box<dyn Output> {
    if quiet {
        return Box::new(NullOutput);
    }

    let sessions_dir = get_sessions_dir();
    let live_dir = sessions_dir.parent().unwrap();
    let pid = std::process::id();
    let live_pid_path = live_dir.join(format!("live-{}.md", pid));
    let live_link = live_dir.join("live.md");

    // Clean up stale live files
    #[cfg(unix)]
    if let Ok(entries) = fs::read_dir(live_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("live-") && name.ends_with(".md") && path != live_pid_path {
                    // Try to parse PID
                    if let Some(old_pid_str) = name
                        .strip_prefix("live-")
                        .and_then(|s| s.strip_suffix(".md"))
                    {
                        if let Ok(old_pid) = old_pid_str.parse::<i32>() {
                            // Check if process is alive (signal 0)
                            let is_alive = unsafe { libc::kill(old_pid, 0) == 0 };
                            if !is_alive {
                                let _ = fs::remove_file(&path);
                            }
                        }
                    }
                }
            }
        }
    }

    match TeeOutput::new(&live_pid_path, color) {
        Ok(tee) => {
            // Update symlink
            let _ = fs::remove_file(&live_link);
            #[cfg(unix)]
            {
                let _ = std::os::unix::fs::symlink(live_pid_path.file_name().unwrap(), &live_link);
            }
            Box::new(tee)
        }
        Err(_) => Box::new(StdoutOutput::new(color)),
    }
}

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
        .arg(format!(
            "{title}: {}",
            question.chars().take(50).collect::<String>()
        ))
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
        cost.map_or(Value::Null, |v| {
            Value::from(((v * 10_000.0).round()) / 10_000.0)
        }),
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

pub fn append_feedback_to_history(rating: u8) {
    let history_file = get_sessions_dir()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("history.jsonl");

    let content = match fs::read_to_string(&history_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read history file: {e}");
            return;
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return;
    }

    let last_line = lines[lines.len() - 1];
    let mut entry = match serde_json::from_str::<Value>(last_line) {
        Ok(Value::Object(m)) => m,
        _ => {
            eprintln!("Failed to parse last history entry");
            return;
        }
    };

    entry.insert("feedback".to_string(), Value::Number(rating.into()));

    let updated_line = match serde_json::to_string(&Value::Object(entry)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to serialize updated entry: {e}");
            return;
        }
    };

    let new_content = if lines.len() == 1 {
        format!("{updated_line}\n")
    } else {
        let prefix = lines[..lines.len() - 1].join("\n");
        format!("{prefix}\n{updated_line}\n")
    };

    if let Err(e) = fs::write(&history_file, new_content) {
        eprintln!("Failed to write history file: {e}");
    }
}
