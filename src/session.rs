//! Session save/share/history utilities.

use crate::config::SessionResult;
use crate::watch::{classify, LineType};
use chrono::Local;
use crossterm::style::{Attribute, Color, Stylize};
use regex::Regex;
use serde_json::{Map, Value};
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::time::Instant;

#[cfg(unix)]
use libc;

pub trait Output: Send + Sync {
    fn write_str(&mut self, s: &str) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
    fn begin_participant(&mut self, _name: &str) -> io::Result<()> {
        Ok(())
    }
    fn end_participant(
        &mut self,
        _name: &str,
        _full_response: &str,
        _elapsed_ms: u64,
    ) -> io::Result<()> {
        Ok(())
    }
    fn begin_phase(&mut self, _phase_name: &str) -> io::Result<()> {
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

pub struct CompactTeeOutput {
    file: Option<BufWriter<fs::File>>,
    color: bool,
    buffer: String,
    current_model: String,
    start_time: Instant,
    spinner_idx: usize,
    token_count: usize,
    streaming_phase: bool,
}

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

impl CompactTeeOutput {
    pub fn new(path: &Path, color: bool) -> Self {
        let file = fs::File::create(path).ok().map(BufWriter::new);
        Self {
            file,
            color,
            buffer: String::new(),
            current_model: String::new(),
            start_time: Instant::now(),
            spinner_idx: 0,
            token_count: 0,
            streaming_phase: false,
        }
    }

    fn clear_spinner_line(&mut self) -> io::Result<()> {
        let mut out = io::stdout();
        write!(out, "\r{}\r", " ".repeat(80))?;
        out.flush()
    }

    fn write_phase_banner(&mut self, phase_name: &str) -> io::Result<()> {
        let mut out = io::stdout();
        if self.color {
            writeln!(
                out,
                "{}",
                format!(" {phase_name} ")
                    .with(Color::Cyan)
                    .attribute(Attribute::Bold)
            )?;
        } else {
            writeln!(out, " {phase_name} ")?;
        }
        out.flush()
    }
}

fn extract_summary(full_response: &str) -> String {
    static STRIP_PREFIX_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^\s*(?:[-*>]+|\d+[.)])\s*").expect("valid regex"));

    let mut lines = Vec::new();
    let mut in_code_fence = false;
    let mut char_count = 0usize;

    for raw_line in full_response.lines() {
        let trimmed = raw_line.trim();

        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if trimmed.len() >= 3 && trimmed.chars().all(|c| c == '-') {
            continue;
        }

        let cleaned = STRIP_PREFIX_RE.replace(trimmed, "").trim().to_string();
        if cleaned.is_empty() {
            continue;
        }

        lines.push(cleaned.clone());
        char_count += cleaned.chars().count();

        if lines.len() >= 2 || char_count >= 200 {
            break;
        }
    }

    let mut summary = lines.join(" ");
    if summary.is_empty() {
        summary = "No summary available.".to_string();
    }

    if summary.chars().count() > 220 {
        summary = format!("{}...", summary.chars().take(217).collect::<String>());
    }

    summary
}

impl Output for CompactTeeOutput {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        if let Some(file) = &mut self.file {
            file.write_all(s.as_bytes())?;
            file.flush()?;
        }

        if self.streaming_phase {
            print!("{s}");
            return io::stdout().flush();
        }

        self.buffer.push_str(s);
        self.token_count += s.split_whitespace().count();
        let steps = self.token_count / 5;
        if steps > self.spinner_idx {
            self.spinner_idx = steps;
            let elapsed = self.start_time.elapsed().as_secs();
            let spinner = SPINNER[self.spinner_idx % SPINNER.len()];
            let mut out = io::stdout();
            write!(
                out,
                "\r  {}  {} {}s",
                self.current_model.as_str(),
                spinner,
                elapsed
            )?;
            out.flush()?;
        }

        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        io::stdout().flush()
    }

    fn begin_participant(&mut self, name: &str) -> io::Result<()> {
        self.current_model = name.to_string();
        self.buffer.clear();
        self.start_time = Instant::now();
        self.spinner_idx = 0;
        self.token_count = 0;
        // streaming_phase is owned by begin_phase — do not reset here

        let mut out = io::stdout();
        if !self.streaming_phase {
            write!(out, "  {name}  ⠙ deliberating...")?;
        }
        out.flush()
    }

    fn end_participant(
        &mut self,
        name: &str,
        full_response: &str,
        elapsed_ms: u64,
    ) -> io::Result<()> {
        // Streaming phases (judgment) already show full output — skip summary card
        if self.streaming_phase {
            self.buffer.clear();
            return Ok(());
        }

        let summary = extract_summary(full_response);
        let elapsed_secs = elapsed_ms / 1000;
        let mut out = io::stdout();
        if elapsed_ms == 0 {
            // Parallel-collected result — no spinner was shown, no time to display
            writeln!(out, "  {name}  ✓")?;
            writeln!(out, "  {summary}\n")?;
        } else {
            self.clear_spinner_line()?;
            writeln!(out, "  {name}  ✓  ({elapsed_secs}s)")?;
            writeln!(out, "  {summary}\n")?;
        }
        out.flush()?;

        self.buffer.clear();
        Ok(())
    }

    fn begin_phase(&mut self, phase_name: &str) -> io::Result<()> {
        self.clear_spinner_line()?;
        if let Some(file) = &mut self.file {
            writeln!(file, "{phase_name}")?;
            file.flush()?;
        }

        self.write_phase_banner(phase_name)?;
        let upper = phase_name.to_ascii_uppercase();
        self.streaming_phase = upper.contains("JUDGMENT") || upper.contains("JUDGE");
        Ok(())
    }
}

pub fn prepare_live_session_path() -> PathBuf {
    let sessions_dir = get_sessions_dir();
    let live_dir = sessions_dir.parent().unwrap();
    let pid = std::process::id();
    let live_pid_path = live_dir.join(format!("live-{}.md", pid));
    let live_link = live_dir.join("live.md");

    #[cfg(unix)]
    if let Ok(entries) = fs::read_dir(live_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("live-") && name.ends_with(".md") && path != live_pid_path {
                    if let Some(old_pid_str) = name
                        .strip_prefix("live-")
                        .and_then(|s| s.strip_suffix(".md"))
                    {
                        if let Ok(old_pid) = old_pid_str.parse::<i32>() {
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

    let _ = fs::remove_file(&live_link);
    #[cfg(unix)]
    {
        let _ = std::os::unix::fs::symlink(live_pid_path.file_name().unwrap(), &live_link);
    }

    live_pid_path
}

pub fn setup_live_output(quiet: bool, color: bool) -> Box<dyn Output> {
    if quiet {
        return Box::new(StdoutOutput::new(false));
    }

    let live_pid_path = prepare_live_session_path();

    match TeeOutput::new(&live_pid_path, color) {
        Ok(tee) => Box::new(tee),
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
