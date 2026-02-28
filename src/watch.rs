//! Live markdown watcher with crossterm styling.

use anyhow::Result;
use crossterm::style::{Attribute, Color, Stylize};
use regex::Regex;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineType {
    Separator,
    PhaseBanner,
    ModelHeader,
    SectionHeader,
    Notice,
    Status,
    Confidence,
    Stats,
    Body,
}

static SEPARATOR_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^={50,}$").unwrap());
static MODEL_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^### (.+)").unwrap());
static SECTION_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^## ([^#].*)").unwrap());
static NOTICE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^>>> (.+)").unwrap());
static STATS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\(\d+\.\d+s,\s*~?\$[\d.]+\)$").unwrap());
static CONFIDENCE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:\*\*)?Confidence:\s*(.+?)(?:\*\*)?$").unwrap());
static STATUS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\((?:thinking|querying|generating|round \d+|\
          querying \d+ (?:models|panelists|attackers) in parallel|\
          round \d+ done|\
          Models see each other|\
          Fallback enabled|\
          Persona context|\
          Challenger|\
          Contrarian challenger|\
          Auto-detected)",
    )
    .unwrap()
});

pub fn classify(line: &str, prev_type: Option<LineType>) -> LineType {
    let stripped = line.trim();
    if stripped.is_empty() {
        return LineType::Body;
    }

    if prev_type == Some(LineType::Separator) && stripped.len() > 3 {
        let core = stripped
            .split(':')
            .next()
            .unwrap_or(stripped)
            .split(" (")
            .next()
            .unwrap_or(stripped)
            .trim();
        if !core.is_empty() && core == core.to_uppercase() {
            return LineType::PhaseBanner;
        }
    }

    if SEPARATOR_RE.is_match(stripped) {
        LineType::Separator
    } else if MODEL_HEADER_RE.is_match(stripped) {
        LineType::ModelHeader
    } else if SECTION_HEADER_RE.is_match(stripped) {
        LineType::SectionHeader
    } else if NOTICE_RE.is_match(stripped) {
        LineType::Notice
    } else if STATS_RE.is_match(stripped) {
        LineType::Stats
    } else if CONFIDENCE_RE.is_match(stripped) {
        LineType::Confidence
    } else if STATUS_RE.is_match(stripped) {
        LineType::Status
    } else {
        LineType::Body
    }
}

pub fn resolve_target(link_path: &Path) -> Option<PathBuf> {
    if let Ok(meta) = std::fs::symlink_metadata(link_path) {
        if meta.file_type().is_symlink() {
            if let Ok(link) = std::fs::read_link(link_path) {
                let target = if link.is_absolute() {
                    link
                } else {
                    link_path.parent().unwrap_or_else(|| Path::new(".")).join(link)
                };
                if target.exists() {
                    return Some(target);
                }
            }
            return None;
        }
    }

    if link_path.exists() {
        return Some(link_path.to_path_buf());
    }
    None
}

struct Renderer {
    prev_type: Option<LineType>,
}

impl Renderer {
    fn new() -> Self {
        Self { prev_type: None }
    }

    fn render(&mut self, out: &mut impl Write, line: &str) -> io::Result<()> {
        let line_type = classify(line, self.prev_type);
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

        out.flush()?;
        self.prev_type = Some(line_type);
        Ok(())
    }
}

fn live_link_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".consilium")
        .join("live.md")
}

pub fn watch_live() -> Result<()> {
    let live_link = live_link_path();
    let mut renderer = Renderer::new();
    let mut out = io::stdout();
    let mut current_target: Option<PathBuf> = None;
    let mut file: Option<File> = None;
    let mut partial_buf = String::new();
    let mut flushed_partial = false;

    loop {
        let new_target = resolve_target(&live_link);

        if new_target != current_target {
            file = None;
            partial_buf.clear();
            flushed_partial = false;

            if let Some(target) = new_target.as_ref() {
                file = File::open(target).ok();
                if renderer.prev_type.is_some() {
                    writeln!(out)?;
                    writeln!(
                        out,
                        "{}",
                        " new session "
                            .with(Color::Blue)
                            .attribute(Attribute::Bold)
                    )?;
                    writeln!(out)?;
                    out.flush()?;
                    renderer.prev_type = None;
                }
            }

            current_target = new_target;
        }

        let Some(fh) = file.as_mut() else {
            thread::sleep(Duration::from_millis(100));
            continue;
        };

        let mut buf = [0_u8; 4096];
        let n = fh.read(&mut buf)?;
        if n > 0 {
            partial_buf.push_str(&String::from_utf8_lossy(&buf[..n]));

            while let Some(newline_pos) = partial_buf.find('\n') {
                let line = partial_buf[..newline_pos].to_string();
                partial_buf.drain(..=newline_pos);

                if flushed_partial {
                    writeln!(out, "{line}")?;
                    out.flush()?;
                    flushed_partial = false;
                } else {
                    renderer.render(&mut out, &line)?;
                }
            }

            if !partial_buf.is_empty() && !flushed_partial {
                write!(out, "{partial_buf}")?;
                out.flush()?;
                flushed_partial = true;
            }
        } else {
            if current_target.as_ref().is_some_and(|p| !p.exists()) {
                file = None;
                current_target = None;
                partial_buf.clear();
                flushed_partial = false;
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
}
