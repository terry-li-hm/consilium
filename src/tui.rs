//! Ratatui live viewer for consilium sessions.

use crate::watch::{classify, resolve_target, LineType};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use regex::Regex;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::LazyLock;
use std::thread;
use std::time::{Duration, Instant};

static STATS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\([\d.]+s,\s*~?\$([\d.]+)\)").unwrap());

const FLEXOKI_BLACK: Color = Color::Rgb(0x10, 0x0F, 0x0F);
const FLEXOKI_SURFACE: Color = Color::Rgb(0x1C, 0x1B, 0x1A);
const FLEXOKI_PANEL: Color = Color::Rgb(0x28, 0x27, 0x26);
const FLEXOKI_TEXT: Color = Color::Rgb(0xCE, 0xCD, 0xC3);
const FLEXOKI_ORANGE: Color = Color::Rgb(0xDA, 0x70, 0x2C);
const FLEXOKI_YELLOW: Color = Color::Rgb(0xD0, 0xA2, 0x15);
const FLEXOKI_GREEN: Color = Color::Rgb(0x87, 0x9A, 0x39);
const FLEXOKI_MAGENTA: Color = Color::Rgb(0x8B, 0x7E, 0xC8);
const FLEXOKI_CYAN: Color = Color::Rgb(0x43, 0x85, 0xBE);
const FLEXOKI_DIM: Color = Color::Rgb(0x87, 0x86, 0x80);

struct TuiApp {
    live_link: PathBuf,
    current_target: Option<PathBuf>,
    file: Option<File>,
    partial_buf: String,
    prev_type: Option<LineType>,
    log_lines: Vec<Line<'static>>,
    stream_preview: String,
    body_buffer: Vec<String>,
    in_model_block: bool,
    phase: String,
    cost: f64,
    start_time: Option<Instant>,
    session_active: bool,
    scroll_offset: usize,
    auto_scroll: bool,
    viewport_height: usize,
}

impl TuiApp {
    fn new(live_link: PathBuf) -> Self {
        Self {
            live_link,
            current_target: None,
            file: None,
            partial_buf: String::new(),
            prev_type: None,
            log_lines: Vec::new(),
            stream_preview: String::new(),
            body_buffer: Vec::new(),
            in_model_block: false,
            phase: "waiting…".to_string(),
            cost: 0.0,
            start_time: None,
            session_active: false,
            scroll_offset: 0,
            auto_scroll: true,
            viewport_height: 1,
        }
    }

    fn elapsed_seconds(&self) -> u64 {
        if self.session_active {
            self.start_time
                .map(|t| t.elapsed().as_secs())
                .unwrap_or_default()
        } else {
            0
        }
    }

    fn max_scroll(&self) -> usize {
        self.log_lines.len().saturating_sub(self.viewport_height.max(1))
    }

    fn push_line(&mut self, line: Line<'static>) {
        self.log_lines.push(line);
        if self.auto_scroll {
            self.scroll_offset = self.max_scroll();
        }
    }

    fn flush_body(&mut self) {
        if self.body_buffer.is_empty() {
            self.in_model_block = false;
            return;
        }
        let drained: Vec<String> = self.body_buffer.drain(..).collect();
        for raw in drained {
            self.push_line(Line::from(Span::raw(raw)));
        }
        self.in_model_block = false;
    }

    fn render_line(&mut self, line: &str) {
        let stripped = line.trim();
        let line_type = classify(line, self.prev_type);

        if line_type == LineType::PhaseBanner {
            self.phase = stripped.to_string();
        }

        if let Some(caps) = STATS_RE.captures(stripped) {
            if let Ok(inc) = caps[1].parse::<f64>() {
                self.cost += inc;
                self.phase = "done".to_string();
                self.session_active = false;
            }
        }

        let is_bodyish = matches!(line_type, LineType::Body | LineType::Confidence);
        if !is_bodyish && self.in_model_block {
            self.flush_body();
        }

        match line_type {
            LineType::Separator => {
                self.push_line(Line::from(Span::styled(
                    "─".repeat(60),
                    Style::default().fg(FLEXOKI_DIM),
                )));
            }
            LineType::PhaseBanner => {
                self.push_line(Line::from(Span::styled(
                    format!(" {stripped} "),
                    Style::default()
                        .fg(FLEXOKI_CYAN)
                        .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                )));
            }
            LineType::ModelHeader => {
                self.in_model_block = true;
                self.push_line(Line::from(""));
                self.push_line(Line::from(Span::styled(
                    stripped.trim_start_matches("### ").to_string(),
                    Style::default().fg(FLEXOKI_YELLOW).add_modifier(Modifier::BOLD),
                )));
            }
            LineType::SectionHeader => {
                self.push_line(Line::from(Span::styled(
                    stripped.trim_start_matches("## ").to_string(),
                    Style::default().fg(FLEXOKI_ORANGE).add_modifier(Modifier::BOLD),
                )));
            }
            LineType::Notice => {
                self.push_line(Line::from(Span::styled(
                    stripped.to_string(),
                    Style::default().fg(FLEXOKI_GREEN).add_modifier(Modifier::BOLD),
                )));
            }
            LineType::Status => {
                self.push_line(Line::from(Span::styled(
                    stripped.to_string(),
                    Style::default()
                        .fg(FLEXOKI_DIM)
                        .add_modifier(Modifier::ITALIC),
                )));
            }
            LineType::Confidence => {
                if self.in_model_block {
                    self.body_buffer.push(line.to_string());
                } else {
                    self.push_line(Line::from(Span::styled(
                        line.to_string(),
                        Style::default().fg(FLEXOKI_MAGENTA).add_modifier(Modifier::BOLD),
                    )));
                }
            }
            LineType::Stats => {
                self.push_line(Line::from(Span::styled(
                    stripped.to_string(),
                    Style::default().fg(FLEXOKI_DIM),
                )));
            }
            LineType::Body => {
                if self.in_model_block {
                    self.body_buffer.push(line.to_string());
                } else {
                    self.push_line(Line::from(Span::raw(line.to_string())));
                }
            }
        }

        self.prev_type = Some(line_type);
    }

    fn update_stream_preview(&mut self) {
        if self.in_model_block && (!self.body_buffer.is_empty() || !self.partial_buf.is_empty()) {
            let tail_start = self.body_buffer.len().saturating_sub(7);
            let mut preview = self.body_buffer[tail_start..].join("\n");
            if !self.partial_buf.is_empty() {
                if !preview.is_empty() {
                    preview.push('\n');
                }
                preview.push_str(&self.partial_buf);
            }
            self.stream_preview = preview;
        } else if !self.partial_buf.is_empty() {
            self.stream_preview = self.partial_buf.clone();
        } else {
            self.stream_preview.clear();
        }
    }

    fn poll_file(&mut self) -> Result<()> {
        let new_target = resolve_target(&self.live_link);
        if new_target != self.current_target {
            if self.in_model_block {
                self.flush_body();
            }
            self.file = None;
            self.partial_buf.clear();

            if let Some(target) = new_target.as_ref() {
                self.file = Some(File::open(target)?);
                self.start_time = Some(Instant::now());
                self.session_active = true;
                self.phase = "starting…".to_string();
                self.cost = 0.0;
                self.scroll_offset = 0;
                self.auto_scroll = true;

                if self.prev_type.is_some() {
                    self.push_line(Line::from(""));
                    self.push_line(Line::from(Span::styled(
                        " new session ",
                        Style::default()
                            .fg(FLEXOKI_CYAN)
                            .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                    )));
                    self.push_line(Line::from(""));
                    self.prev_type = None;
                }
            } else {
                self.phase = "waiting…".to_string();
                self.session_active = false;
            }

            self.current_target = new_target;
        }

        let Some(fh) = self.file.as_mut() else {
            self.update_stream_preview();
            return Ok(());
        };

        let mut buf = [0_u8; 4096];
        let n = fh.read(&mut buf)?;
        if n > 0 {
            self.partial_buf.push_str(&String::from_utf8_lossy(&buf[..n]));
            while let Some(pos) = self.partial_buf.find('\n') {
                let line = self.partial_buf[..pos].to_string();
                self.partial_buf.drain(..=pos);
                self.render_line(&line);
            }
            self.update_stream_preview();
        } else {
            if self.current_target.as_ref().is_some_and(|p| !p.exists()) {
                if self.in_model_block {
                    self.flush_body();
                }
                self.file = None;
                self.current_target = None;
                self.partial_buf.clear();
                self.stream_preview.clear();
            }
        }
        Ok(())
    }
}

fn default_live_link() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".consilium")
        .join("live.md")
}

pub fn run_tui() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = TuiApp::new(default_live_link());
    let mut should_quit = false;
    let loop_result = (|| -> Result<()> {
        while !should_quit {
            app.poll_file()?;

            terminal.draw(|f| {
                let size = f.area();
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(1),
                        Constraint::Min(1),
                        Constraint::Length(4),
                    ])
                    .split(size);

                let elapsed = app.elapsed_seconds();
                let mins = elapsed / 60;
                let secs = elapsed % 60;

                let bar = Line::from(vec![
                    Span::styled(
                        format!(" {} ", app.phase),
                        Style::default()
                            .fg(FLEXOKI_TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" | ", Style::default().fg(FLEXOKI_DIM)),
                    Span::styled(
                        format!("{mins}:{secs:02}"),
                        Style::default().fg(FLEXOKI_TEXT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" | ", Style::default().fg(FLEXOKI_DIM)),
                    Span::styled(
                        format!("${:.3}", app.cost),
                        Style::default()
                            .fg(FLEXOKI_YELLOW)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]);
                let bar_widget =
                    Paragraph::new(bar).style(Style::default().bg(FLEXOKI_SURFACE).fg(FLEXOKI_TEXT));
                f.render_widget(bar_widget, chunks[0]);

                let log_block = Block::default()
                    .title("Transcript")
                    .borders(Borders::ALL)
                    .style(Style::default().bg(FLEXOKI_BLACK).fg(FLEXOKI_TEXT))
                    .border_style(Style::default().fg(FLEXOKI_PANEL));
                let log_height = chunks[1].height.saturating_sub(2) as usize;
                app.viewport_height = log_height.max(1);
                if app.auto_scroll {
                    app.scroll_offset = app.max_scroll();
                }
                let log = Paragraph::new(app.log_lines.clone())
                    .block(log_block)
                    .scroll((app.scroll_offset as u16, 0))
                    .wrap(Wrap { trim: false });
                f.render_widget(log, chunks[1]);

                let stream = Paragraph::new(app.stream_preview.clone())
                    .block(
                        Block::default()
                            .title("Stream")
                            .borders(Borders::ALL)
                            .style(Style::default().bg(FLEXOKI_SURFACE))
                            .border_style(Style::default().fg(FLEXOKI_PANEL)),
                    )
                    .style(Style::default().fg(FLEXOKI_DIM))
                    .wrap(Wrap { trim: false });
                f.render_widget(stream, chunks[2]);
            })?;

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => should_quit = true,
                        KeyCode::Up => {
                            app.auto_scroll = false;
                            app.scroll_offset = app.scroll_offset.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            let max = app.max_scroll();
                            app.scroll_offset = (app.scroll_offset + 1).min(max);
                            app.auto_scroll = app.scroll_offset >= max;
                        }
                        KeyCode::PageUp => {
                            app.auto_scroll = false;
                            let step = app.viewport_height.max(1);
                            app.scroll_offset = app.scroll_offset.saturating_sub(step);
                        }
                        KeyCode::PageDown => {
                            let step = app.viewport_height.max(1);
                            let max = app.max_scroll();
                            app.scroll_offset = (app.scroll_offset + step).min(max);
                            app.auto_scroll = app.scroll_offset >= max;
                        }
                        _ => {}
                    }
                }
            } else {
                thread::sleep(Duration::from_millis(5));
            }
        }
        Ok(())
    })();

    let cleanup_result = (|| -> Result<()> {
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        Ok(())
    })();

    loop_result?;
    cleanup_result
}
