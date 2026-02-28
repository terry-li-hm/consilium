use crate::prompts;
use crate::session::get_sessions_dir;
use chrono::{DateTime, Local};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::process::Command;

pub fn list_roles() {
    println!("Available predefined roles for --solo --roles:\n");
    let roles = prompts::role_library();
    let mut keys: Vec<_> = roles.keys().collect();
    keys.sort();

    for name in keys {
        let desc = roles.get(name).unwrap();
        let short = if desc.len() > 80 {
            format!("{}...", &desc[..77])
        } else {
            desc.to_string()
        };
        println!("  {:<20} {}", name, short.replace('\n', " "));
    }
    println!("\nDefault: Advocate, Skeptic, Pragmatist");
    println!("Unknown roles use a generic prompt. Any name works.");
}

pub fn list_sessions() {
    let sessions_dir = get_sessions_dir();
    let mut entries = match fs::read_dir(&sessions_dir) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("Failed to read sessions directory: {e}");
            return;
        }
    };

    entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).ok()));

    if entries.is_empty() {
        println!("No sessions found.");
    } else {
        println!("Sessions in {}:\n", sessions_dir.display());
        for entry in entries.iter().take(20) {
            let metadata = entry.metadata().ok();
            let mtime: String = metadata
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    let dt: DateTime<Local> = t.into();
                    dt.format("%Y-%m-%d %H:%M").to_string()
                })
                .unwrap_or_else(|| "unknown".to_string());
            println!("  {}  {}", mtime, entry.file_name().to_string_lossy());
        }
        if entries.len() > 20 {
            println!("\n  ... and {} more", entries.len() - 20);
        }
    }
}

pub fn show_stats() {
    let history_file = get_sessions_dir()
        .parent()
        .expect("sessions dir should have a parent")
        .join("history.jsonl");

    if !history_file.exists() {
        println!("No history found.");
        return;
    }

    let content = match fs::read_to_string(&history_file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read history file: {e}");
            return;
        }
    };

    let mut entries = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(line) {
            entries.push(val);
        }
    }

    if entries.is_empty() {
        println!("No sessions recorded.");
        return;
    }

    let mut by_mode: HashMap<String, Vec<&Value>> = HashMap::new();
    for entry in &entries {
        let mode = entry
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        by_mode.entry(mode).or_default().push(entry);
    }

    let first = entries[0]
        .get("timestamp")
        .and_then(|v| v.as_str())
        .map(|s| &s[..10])
        .unwrap_or("?");
    let last = entries
        .last()
        .and_then(|e| e.get("timestamp"))
        .and_then(|v| v.as_str())
        .map(|s| &s[..10])
        .unwrap_or("?");

    println!("Consilium Stats ({} sessions, {} — {})
", entries.len(), first, last);
    println!(
        "{:<14} {:>8} {:>10} {:>12} {:>10}",
        "Mode", "Sessions", "Avg Cost", "Total Cost", "Avg Time"
    );

    let mut total_cost = 0.0;
    let mut sorted_modes: Vec<_> = by_mode.keys().collect();
    sorted_modes.sort();

    for mode in sorted_modes {
        let mode_entries = by_mode.get(mode).unwrap();
        let count = mode_entries.len();
        let costs: Vec<f64> = mode_entries
            .iter()
            .filter_map(|e| e.get("cost").and_then(|v| v.as_f64()))
            .collect();
        let durations: Vec<f64> = mode_entries
            .iter()
            .filter_map(|e| e.get("duration").and_then(|v| v.as_f64()))
            .collect();

        let avg_cost_str = if !costs.is_empty() {
            format!("${:.2}", costs.iter().sum::<f64>() / costs.len() as f64)
        } else {
            "—".to_string()
        };
        let sum_cost = costs.iter().sum::<f64>();
        total_cost += sum_cost;
        let total_cost_str = if !costs.is_empty() {
            format!("${:.2}", sum_cost)
        } else {
            "—".to_string()
        };
        let avg_dur_str = if !durations.is_empty() {
            format!("{:.0}s", durations.iter().sum::<f64>() / durations.len() as f64)
        } else {
            "—".to_string()
        };

        println!(
            "{:<14} {:>8} {:>10} {:>12} {:>10}",
            mode, count, avg_cost_str, total_cost_str, avg_dur_str
        );
    }

    println!(
        "
{:<14} {:>8} {:>10} ${:>11.2}",
        "Total",
        entries.len(),
        "",
        total_cost
    );

    let now = Local::now();
    let cutoff = now - chrono::Duration::days(7);
    let recent: Vec<_> = entries
        .iter()
        .filter(|e| {
            e.get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Local) >= cutoff)
                .unwrap_or(false)
        })
        .collect();
    let recent_cost: f64 = recent
        .iter()
        .filter_map(|e| e.get("cost").and_then(|v| v.as_f64()))
        .sum();
    println!(
        "
Last 7 days: {} sessions, ${:.2}",
        recent.len(),
        recent_cost
    );
}

pub fn view_session(term: Option<&str>) {
    let sessions_dir = get_sessions_dir();
    let mut entries = match fs::read_dir(&sessions_dir) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("Failed to read sessions directory: {e}");
            std::process::exit(1);
        }
    };

    if entries.is_empty() {
        println!("No sessions found.");
        std::process::exit(1);
    }

    entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).ok()));

    let target = if let Some(t) = term {
        let t_lower = t.to_lowercase();
        let matches: Vec<_> = entries
            .iter()
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .to_lowercase()
                    .contains(&t_lower)
            })
            .collect();

        if !matches.is_empty() {
            if matches.len() > 1 {
                println!("({} matches, showing most recent)
", matches.len());
            }
            matches[0].path()
        } else {
            // Search content
            let mut content_matches = Vec::new();
            for entry in &entries {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if content.to_lowercase().contains(&t_lower) {
                        content_matches.push(entry.path());
                    }
                }
                if content_matches.len() > 10 {
                    break;
                }
            }
            if !content_matches.is_empty() {
                if content_matches.len() > 1 {
                    println!("({} matches, showing most recent)
", content_matches.len());
                }
                content_matches[0].clone()
            } else {
                println!("No sessions matching '{}'.", t);
                std::process::exit(1);
            }
        }
    } else {
        entries[0].path()
    };

    let pager = std::env::var("PAGER").unwrap_or_else(|_| "less".to_string());
    let mut child = Command::new(&pager)
        .arg(target)
        .spawn()
        .expect("Failed to spawn pager");
    let _ = child.wait();
}

pub fn search_sessions(term: &str) {
    let sessions_dir = get_sessions_dir();
    let mut entries = match fs::read_dir(&sessions_dir) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("Failed to read sessions directory: {e}");
            return;
        }
    };

    entries.sort_by_key(|e| std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).ok()));

    let t_lower = term.to_lowercase();
    let mut matches = Vec::new();

    for entry in entries {
        if let Ok(mut file) = fs::File::open(entry.path()) {
            let mut content = String::new();
            // Read up to 10KB for searching
            let _ = std::io::Read::by_ref(&mut file)
                .take(10240)
                .read_to_string(&mut content);
            if content.to_lowercase().contains(&t_lower) {
                matches.push((entry, content));
            }
        }
    }

    if matches.is_empty() {
        println!("No sessions matching '{}'.", term);
    } else {
        println!("Sessions matching '{}':
", term);
        for (entry, content) in matches.iter().take(20) {
            let metadata = entry.metadata().ok();
            let mtime: String = metadata
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    let dt: DateTime<Local> = t.into();
                    dt.format("%Y-%m-%d %H:%M").to_string()
                })
                .unwrap_or_else(|| "unknown".to_string());

            let question = content
                .lines()
                .find(|l| l.contains("**Question:**"))
                .and_then(|l| l.split("**Question:**").nth(1))
                .map(|s| s.trim())
                .map(|s| {
                    if s.len() > 60 {
                        format!("{}...", &s[..57])
                    } else {
                        s.to_string()
                    }
                })
                .unwrap_or_else(|| entry.file_name().to_string_lossy().to_string());

            println!("  {}  {}", mtime, question);
        }
        if matches.len() > 20 {
            println!("
  ... and {} more", matches.len() - 20);
        }
    }
}
