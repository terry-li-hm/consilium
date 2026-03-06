//! Quick mode: parallel queries, no debate, no judge.

use crate::api::{query_model_async, query_model_streaming};
use crate::config::{
    fallback_also_failed_message, is_error_response, model_max_output_tokens,
    per_model_max_tokens, CostTracker, Message, ModelEntry, ReasoningEffort, SessionResult,
};
use crate::session::Output;
use chrono::Local;
use futures_util::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;
use serde_json::json;
use std::time::Instant;

struct NullOutput;

impl Output for NullOutput {
    fn write_str(&mut self, _s: &str) -> std::io::Result<()> {
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn run_quick_streaming(
    question: &str,
    context: Option<&str>,
    models: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    anthropic_api_key: Option<&str>,
    max_tokens: u32,
    timeout: f64,
    cost_tracker: &CostTracker,
    effort: Option<ReasoningEffort>,
    output: &mut dyn Output,
) -> Vec<(String, String, String, u64)> {
    let client = Client::new();
    let _ = output.begin_phase("QUICK RESPONSES");
    let full_question = match context {
        Some(ctx) => format!("{ctx}\n\n{question}"),
        None => question.to_string(),
    };
    let messages = vec![Message::user(&full_question)];

    let mut out = Vec::with_capacity(models.len());
    let mut pending = FuturesUnordered::new();

    for &(name, model, fallback) in models {
        let client = client.clone();
        let api_key = api_key.to_string();
        let messages = messages.clone();
        let name = name.to_string();
        let model = model.to_string();
        let model_name = model.split('/').next_back().unwrap_or(&model).to_string();
        let model_max_tokens =
            model_max_output_tokens(&model).min(per_model_max_tokens(&model, max_tokens));
        let fallback_owned: Option<(String, String)> =
            fallback.map(|(provider, fallback_model)| {
                (provider.to_string(), fallback_model.to_string())
            });
        let google_api_key = google_api_key.map(|s| s.to_string());
        let zhipu_api_key = zhipu_api_key.map(|s| s.to_string());
        let moonshot_api_key = moonshot_api_key.map(|s| s.to_string());
        let openai_api_key = openai_api_key.map(|s| s.to_string());
        let xai_api_key = xai_api_key.map(|s| s.to_string());
        let anthropic_api_key = anthropic_api_key.map(|s| s.to_string());
        let cost_tracker = cost_tracker.clone();

        pending.push(async move {
            let participant_start = Instant::now();
            let mut null_output = NullOutput;

            let mut response = query_model_streaming(
                &client,
                &api_key,
                &model,
                &messages,
                model_max_tokens,
                timeout,
                Some(&cost_tracker),
                effort,
                &mut null_output,
            )
            .await;

            let mut used_model_name = model_name.clone();
            let fallback_ref = fallback_owned.as_ref().map(|(p, m)| (p.as_str(), m.as_str()));

            // Fallback only if streaming failed.
            if is_error_response(&response) {
                let primary_response = response;
                let (_, fb_model_name, fb_response) = query_model_async(
                    &client,
                    &api_key,
                    &model,
                    &messages,
                    &name,
                    fallback_ref,
                    google_api_key.as_deref(),
                    zhipu_api_key.as_deref(),
                    moonshot_api_key.as_deref(),
                    openai_api_key.as_deref(),
                    xai_api_key.as_deref(),
                    anthropic_api_key.as_deref(),
                    model_max_tokens,
                    timeout,
                    2,
                    Some(&cost_tracker),
                    effort,
                )
                .await;

                if is_error_response(&fb_response) {
                    response =
                        fallback_also_failed_message(&name, &primary_response, &fb_response);
                } else {
                    used_model_name = fb_model_name;
                    response = fb_response;
                }
            }

            (
                name,
                model_name,
                used_model_name,
                response.trim().to_string(),
                participant_start.elapsed().as_millis() as u64,
            )
        });
    }

    while let Some((name, model_name, used_model_name, response, elapsed_ms)) = pending.next().await
    {
        let _ = output.begin_participant(&model_name);
        let _ = output.write_str(&format!("### {model_name}\n{response}\n\n"));
        let _ = output.end_participant(&model_name, &response, elapsed_ms);
        out.push((name, used_model_name, response, elapsed_ms));
    }

    out
}

async fn run_quick_parallel(
    messages: &[Message],
    models: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    anthropic_api_key: Option<&str>,
    max_tokens: u32,
    timeout: f64,
    cost_tracker: &CostTracker,
    effort: Option<ReasoningEffort>,
) -> Vec<(String, String, String, u64)> {
    let client = Client::new();
    let mut pending = FuturesUnordered::new();

    for (idx, &(name, model, fallback)) in models.iter().enumerate() {
        let client = client.clone();
        let api_key = api_key.to_string();
        let messages = messages.to_vec();
        let name = name.to_string();
        let model = model.to_string();
        let model_max_tokens =
            model_max_output_tokens(&model).min(per_model_max_tokens(&model, max_tokens));
        let fallback_owned: Option<(String, String)> =
            fallback.map(|(provider, fallback_model)| {
                (provider.to_string(), fallback_model.to_string())
            });
        let google_api_key = google_api_key.map(|s| s.to_string());
        let zhipu_api_key = zhipu_api_key.map(|s| s.to_string());
        let moonshot_api_key = moonshot_api_key.map(|s| s.to_string());
        let openai_api_key = openai_api_key.map(|s| s.to_string());
        let xai_api_key = xai_api_key.map(|s| s.to_string());
        let anthropic_api_key = anthropic_api_key.map(|s| s.to_string());
        let cost_tracker = cost_tracker.clone();

        pending.push(async move {
            let start = Instant::now();
            let fallback_ref = fallback_owned.as_ref().map(|(p, m)| (p.as_str(), m.as_str()));
            let (speaker_name, used_model_name, response) = query_model_async(
                &client,
                &api_key,
                &model,
                &messages,
                &name,
                fallback_ref,
                google_api_key.as_deref(),
                zhipu_api_key.as_deref(),
                moonshot_api_key.as_deref(),
                openai_api_key.as_deref(),
                xai_api_key.as_deref(),
                anthropic_api_key.as_deref(),
                model_max_tokens,
                timeout,
                2,
                Some(&cost_tracker),
                effort,
            )
            .await;
            (
                idx,
                speaker_name,
                used_model_name,
                response.trim().to_string(),
                start.elapsed().as_millis() as u64,
            )
        });
    }

    let mut out = Vec::with_capacity(models.len());
    while let Some(result) = pending.next().await {
        out.push(result);
    }
    out.sort_by_key(|(idx, _, _, _, _)| *idx);
    out.into_iter()
        .map(|(_, speaker_name, used_model_name, response, elapsed_ms)| {
            (speaker_name, used_model_name, response, elapsed_ms)
        })
        .collect()
}

pub async fn run_quick(
    question: &str,
    context: Option<&str>,
    models: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    openai_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    anthropic_api_key: Option<&str>,
    output: &mut dyn Output,
    format: &str,
    timeout: f64,
    effort: Option<ReasoningEffort>,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();

    let _ = output.write_str(&format!(
        "(querying {} models in parallel...)\n\n",
        models.len()
    ));

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
            zhipu_api_key,
            moonshot_api_key,
            openai_api_key,
            xai_api_key,
            anthropic_api_key,
            2048,
            timeout,
            &cost_tracker,
            effort,
            output,
        )
        .await
    } else {
        run_quick_parallel(
            &messages,
            models,
            api_key,
            google_api_key,
            zhipu_api_key,
            moonshot_api_key,
            openai_api_key,
            xai_api_key,
            anthropic_api_key,
            2048,
            timeout,
            &cost_tracker,
            effort,
        )
        .await
    };

    let duration = start.elapsed().as_secs_f64();
    let total_cost = (cost_tracker.total() * 10_000.0).round() / 10_000.0;

    let failures: Vec<String> = results
        .iter()
        .filter_map(|(_, model_name, response, _)| {
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
                .filter(|(_, _, response, _)| !is_error_response(response))
                .map(|(_, model_name, response, elapsed_ms)| {
                    json!({
                        "model": model_name,
                        "content": response,
                        "duration_ms": elapsed_ms,
                    })
                })
                .collect();

            let errors: Vec<_> = results
                .iter()
                .filter(|(_, _, response, _)| is_error_response(response))
                .map(|(_, model_name, response, elapsed_ms)| {
                    json!({
                        "model": model_name,
                        "error": response,
                        "duration_ms": elapsed_ms,
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
                .map(|(_, model_name, response, _)| format!("### {model_name}\n{response}"))
                .collect();
            parts.join("\n\n")
        }
    };

    let mut model_timings = serde_json::Map::new();
    for (_, model_name, _, elapsed_ms) in &results {
        model_timings.insert(model_name.clone(), serde_json::Value::from(*elapsed_ms));
    }
    let mut extra = serde_json::Map::new();
    extra.insert("model_timings".into(), serde_json::Value::Object(model_timings));

    SessionResult {
        transcript,
        cost: total_cost,
        duration,
        failures: if failures.is_empty() {
            None
        } else {
            Some(failures)
        },
        extra: Some(extra),
    }
}
