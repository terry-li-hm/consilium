use crate::config::{
    resolved_council, resolved_judge_model, ModelEntry, CONSILIUM_MODEL_M1_ENV,
    CONSILIUM_MODEL_M2_ENV, CONSILIUM_MODEL_M3_ENV, CONSILIUM_MODEL_M4_ENV,
    CONSILIUM_MODEL_M5_ENV, CONSILIUM_MODEL_JUDGE_ENV,
};
use crate::session::get_sessions_dir;
use chrono::{DateTime, Local};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::process::Command;

pub fn list_sessions() {
    let sessions_dir = get_sessions_dir();
    let mut entries = match fs::read_dir(&sessions_dir) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
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

fn calculate_percentile(mut values: Vec<f64>, percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() as f64 * percentile / 100.0).floor() as usize).min(values.len() - 1);
    Some(values[idx])
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

    let total_cost: f64 = entries
        .iter()
        .filter_map(|e| e.get("cost").and_then(|v| v.as_f64()))
        .sum();

    println!(
        "consilium stats — {} sessions, ${:.2} total\n",
        entries.len(),
        total_cost
    );

    let mut mode_stats: Vec<(&String, &Vec<&Value>)> = by_mode.iter().collect();
    mode_stats.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    println!("By mode (sorted by usage):");

    let mut all_feedbacks: Vec<u8> = Vec::new();

    for (mode, mode_entries) in &mode_stats {
        let count = mode_entries.len();
        let costs: Vec<f64> = mode_entries
            .iter()
            .filter_map(|e| e.get("cost").and_then(|v| v.as_f64()))
            .collect();
        let durations: Vec<f64> = mode_entries
            .iter()
            .filter_map(|e| e.get("duration").and_then(|v| v.as_f64()))
            .collect();

        let avg_cost = if !costs.is_empty() {
            costs.iter().sum::<f64>() / costs.len() as f64
        } else {
            0.0
        };
        let sum_cost = costs.iter().sum::<f64>();

        let p50 = calculate_percentile(durations.clone(), 50.0);
        let p95 = calculate_percentile(durations.clone(), 95.0);

        let mode_feedbacks: Vec<u8> = mode_entries
            .iter()
            .filter_map(|e| e.get("feedback").and_then(|v| v.as_u64()).map(|v| v as u8))
            .filter(|&v| (1..=5).contains(&v))
            .collect();
        all_feedbacks.extend(&mode_feedbacks);

        let feedback_str = if !mode_feedbacks.is_empty() {
            let avg: f64 = mode_feedbacks.iter().sum::<u8>() as f64 / mode_feedbacks.len() as f64;
            format!("  ★{:.1}", avg)
        } else {
            String::new()
        };

        let p50_str = p50.map_or("—".to_string(), |v| format!("{:.0}s p50", v));
        let p95_str = p95.map_or("—".to_string(), |v| format!("{:.0}s p95", v));

        println!(
            "  {:<12} {:>3} sessions  ${:.2} avg  ${:.2} total  {:>10}  {:>10}{}",
            mode, count, avg_cost, sum_cost, p50_str, p95_str, feedback_str
        );
    }

    let now = Local::now();
    let cutoff_7d = now - chrono::Duration::days(7);
    let cutoff_30d = now - chrono::Duration::days(30);

    let recent_7d: Vec<_> = entries
        .iter()
        .filter(|e| {
            e.get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Local) >= cutoff_7d)
                .unwrap_or(false)
        })
        .collect();

    let recent_30d: Vec<_> = entries
        .iter()
        .filter(|e| {
            e.get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Local) >= cutoff_30d)
                .unwrap_or(false)
        })
        .collect();

    let cost_7d: f64 = recent_7d
        .iter()
        .filter_map(|e| e.get("cost").and_then(|v| v.as_f64()))
        .sum();

    let cost_30d: f64 = recent_30d
        .iter()
        .filter_map(|e| e.get("cost").and_then(|v| v.as_f64()))
        .sum();

    println!();
    println!("Last 7 days: {} sessions, ${:.2}", recent_7d.len(), cost_7d);
    println!(
        "Last 30 days: {} sessions, ${:.2}",
        recent_30d.len(),
        cost_30d
    );

    if !all_feedbacks.is_empty() {
        let avg_feedback: f64 =
            all_feedbacks.iter().sum::<u8>() as f64 / all_feedbacks.len() as f64;
        println!(
            "\nFeedback: {:.1} avg ({} rated sessions)",
            avg_feedback,
            all_feedbacks.len()
        );
    }
}

pub fn view_session(term: Option<&str>) {
    let sessions_dir = get_sessions_dir();
    let mut entries = match fs::read_dir(&sessions_dir) {
        Ok(dir) => dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
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
                println!(
                    "({} matches, showing most recent)
",
                    matches.len()
                );
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
                    println!(
                        "({} matches, showing most recent)
",
                        content_matches.len()
                    );
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
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
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
        println!(
            "Sessions matching '{}':
",
            term
        );
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
            println!(
                "
  ... and {} more",
                matches.len() - 20
            );
        }
    }
}

pub fn doctor() {
    let council = resolved_council();
    let judge = resolved_judge_model();

    let m1_source = env_source(CONSILIUM_MODEL_M1_ENV);
    let m2_source = env_source(CONSILIUM_MODEL_M2_ENV);
    let m3_source = env_source(CONSILIUM_MODEL_M3_ENV);
    let m4_source = env_source(CONSILIUM_MODEL_M4_ENV);
    let m5_source = env_source(CONSILIUM_MODEL_M5_ENV);
    let judge_source = env_source(CONSILIUM_MODEL_JUDGE_ENV);

    println!("consilium doctor");
    println!("═══════════════════════════════════════");
    println!();
    println!("Council models:");
    println!("  {:<14} {:<35} ({})", council[0].0, council[0].1, m1_source);
    println!("  {:<14} {:<35} ({})", council[1].0, council[1].1, m2_source);
    println!("  {:<14} {:<35} ({})", council[2].0, council[2].1, m3_source);
    println!("  {:<14} {:<35} ({})", council[3].0, council[3].1, m4_source);
    println!("  {:<14} {:<35} ({})", council[4].0, glm_model_name(&council), m5_source);
    println!("  {:<14} {:<35} ({})", "Judge", judge, judge_source);
    println!();
    println!("API keys:");
    println!(
        "  {:<20} {}",
        "OPENROUTER_API_KEY",
        key_marker("OPENROUTER_API_KEY")
    );
    println!("  {:<20} {}", "OPENAI_API_KEY", key_marker("OPENAI_API_KEY"));
    println!("  {:<20} {}", "ANTHROPIC_API_KEY", key_marker("ANTHROPIC_API_KEY"));
    println!("  {:<20} {}", "GOOGLE_API_KEY", key_marker("GOOGLE_API_KEY"));
    println!("  {:<20} {}", "XAI_API_KEY", key_marker("XAI_API_KEY"));
    println!("  {:<20} {}", "ZHIPU_API_KEY", key_marker("ZHIPU_API_KEY"));
    println!("  {:<20} {}", "MOONSHOT_API_KEY", key_marker("MOONSHOT_API_KEY"));
    println!();
    println!("Routing (benchmarked from HK, 2026-03-04):");
    println!("  {}  → OpenAI Responses API direct 1.6s (OR 4.0s)", council[0].0);
    println!("  {}  → OpenRouter only (OR 5.0s vs Google direct 8.3s)", council[1].0);
    println!("  {}    → xAI direct 5.8s  (OR 13.0s)", council[2].0);
    println!("  {} → Moonshot direct 2.7s (OR 2.6s, tied)", council[3].0);
    println!("  {}     → z.ai direct 2.6s  (OR 9.8s)", council[4].0);
    println!("  Judge         → Anthropic direct");
}

fn glm_model_name(council: &[ModelEntry]) -> &str {
    council
        .iter()
        .find(|(name, _, _)| *name == "GLM")
        .and_then(|(_, _, fallback)| *fallback)
        .map(|(_, model)| model)
        .unwrap_or("glm-5")
}

fn env_source(var: &'static str) -> &'static str {
    match std::env::var(var) {
        Ok(value) if !value.trim().is_empty() => var,
        _ => "default",
    }
}

fn key_marker(var: &str) -> &'static str {
    match std::env::var(var) {
        Ok(value) if !value.trim().is_empty() => "✓ set",
        _ => "✗ not set",
    }
}
