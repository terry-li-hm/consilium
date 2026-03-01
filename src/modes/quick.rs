//! Quick mode: parallel queries, no debate, no judge.

use crate::api::{query_model_async, query_model_streaming, run_parallel};
use crate::config::{is_error_response, CostTracker, Message, ModelEntry, SessionResult};
use crate::session::Output;
use chrono::Local;
use reqwest::Client;
use serde_json::json;
use std::time::Instant;

async fn run_quick_streaming(
    question: &str,
    context: Option<&str>,
    models: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    max_tokens: u32,
    timeout: f64,
    cost_tracker: &CostTracker,
    output: &mut dyn Output,
) -> Vec<(String, String, String)> {
    let client = Client::new();
    let _ = output.begin_phase("QUICK RESPONSES");
    let full_question = match context {
        Some(ctx) => format!("{ctx}\n\n{question}"),
        None => question.to_string(),
    };
    let messages = vec![Message::user(&full_question)];

    let mut out = Vec::with_capacity(models.len());

    for &(name, model, fallback) in models {
        let model_name = model.split('/').next_back().unwrap_or(model).to_string();
        let participant_start = Instant::now();
        let _ = output.begin_participant(&model_name);

        let _ = output.write_str(&format!("### {model_name}\n"));

        let mut response = query_model_streaming(
            &client,
            api_key,
            model,
            &messages,
            max_tokens,
            timeout,
            Some(cost_tracker),
            output,
        )
        .await;

        let mut used_model_name = model_name.clone();

        // Fallback only if streaming failed.
        if is_error_response(&response) {
            let (_, fb_model_name, fb_response) = query_model_async(
                &client,
                api_key,
                model,
                &messages,
                name,
                fallback,
                google_api_key,
                max_tokens,
                2,
                Some(cost_tracker),
            )
            .await;

            used_model_name = fb_model_name;
            response = fb_response;

            if !is_error_response(&response) {
                let _ = output.write_str(&format!("{}\n", response));
            }
        }

        let _ = output.write_str("\n");
        let _ = output.end_participant(
            &model_name,
            &response,
            participant_start.elapsed().as_millis() as u64,
        );

        out.push((
            name.to_string(),
            used_model_name,
            response.trim().to_string(),
        ));
    }

    out
}

pub async fn run_quick(
    question: &str,
    context: Option<&str>,
    models: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    output: &mut dyn Output,
    format: &str,
    timeout: f64,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();

    let _ = output.write_str(&format!("(querying {} models in parallel...)\n\n", models.len()));

    let full_question = match context {
        Some(ctx) => format!("{ctx}\n\n{question}"),
        None => question.to_string(),
    };
    let messages = vec![Message::user(&full_question)];

    // For quick mode, if we are not quiet (meaning we have a real output), we do streaming.
    // If we are quiet, main.rs would have passed a dummy or we wouldn't be here.
    // But actually we should check if it's prose format or not.
    
    let results = if format == "prose" {
        run_quick_streaming(
            question,
            context,
            models,
            api_key,
            google_api_key,
            4000,
            timeout,
            &cost_tracker,
            output,
        )
        .await
    } else {
        run_parallel(
            models,
            &messages,
            api_key,
            google_api_key,
            4000,
            Some(&cost_tracker),
            None,
        )
        .await
    };

    let duration = start.elapsed().as_secs_f64();
    let total_cost = (cost_tracker.total() * 10_000.0).round() / 10_000.0;

    let failures: Vec<String> = results
        .iter()
        .filter_map(|(_, model_name, response)| {
            if is_error_response(response) {
                Some(format!("{model_name}: {response}"))
            } else {
                None
            }
        })
        .collect();

    if !failures.is_empty() {
        let _ = output.write_str("Failures:\n");
        for f in &failures {
            let _ = output.write_str(&format!("  - {f}\n"));
        }
        let _ = output.write_str("\n");
    }

    let _ = output.write_str(&format!("({:.1}s, ~${:.2})\n", duration, total_cost));

    let transcript = match format {
        "json" | "yaml" => {
            let responses: Vec<_> = results
                .iter()
                .filter(|(_, _, response)| !is_error_response(response))
                .map(|(_, model_name, response)| {
                    json!({
                        "model": model_name,
                        "content": response,
                    })
                })
                .collect();

            let errors: Vec<_> = results
                .iter()
                .filter(|(_, _, response)| is_error_response(response))
                .map(|(_, model_name, response)| {
                    json!({
                        "model": model_name,
                        "error": response,
                    })
                })
                .collect();

            let mut structured = json!({
                "schema_version": "1.0",
                "question": question,
                "mode": "quick",
                "responses": responses,
                "errors": errors,
                "meta": {
                    "timestamp": Local::now().to_rfc3339(),
                    "models_used": models
                        .iter()
                        .map(|(_, model, _)| model.split('/').next_back().unwrap_or(model).to_string())
                        .collect::<Vec<_>>(),
                    "duration_seconds": (duration * 10.0).round() / 10.0,
                    "estimated_cost_usd": total_cost,
                }
            });

            if errors.is_empty() {
                if let Some(obj) = structured.as_object_mut() {
                    obj.remove("errors");
                }
            }

            if format == "json" {
                serde_json::to_string_pretty(&structured).unwrap_or_else(|_| "{}".to_string())
            } else {
                serde_yaml::to_string(&structured).unwrap_or_else(|_| "{}\n".to_string())
            }
        }
        _ => {
            let parts: Vec<String> = results
                .iter()
                .map(|(_, model_name, response)| format!("### {model_name}\n{response}"))
                .collect();
            parts.join("\n\n")
        }
    };

    SessionResult {
        transcript,
        cost: total_cost,
        duration,
        failures: if failures.is_empty() {
            None
        } else {
            Some(failures)
        },
    }
}
