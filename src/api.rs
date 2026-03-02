//! HTTP clients, SSE streaming, parallel queries, retry, and fallback.

use crate::config::{
    is_thinking_model, CostTracker, Message, ModelEntry, BIGMODEL_URL, GOOGLE_AI_STUDIO_URL,
    MOONSHOT_URL, OPENROUTER_URL, XAI_URL,
};
use crate::session::Output;
use futures_util::StreamExt;
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::time::sleep;

/// Query a model via OpenRouter with retry logic.
pub async fn query_model(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    timeout_secs: f64,
    retries: u32,
    cost_tracker: Option<&CostTracker>,
) -> String {
    let mut max_tokens = max_tokens;
    let mut timeout_secs = timeout_secs;

    if is_thinking_model(model) {
        max_tokens = max_tokens.max(2500);
        timeout_secs = timeout_secs.max(300.0);
    }

    for attempt in 0..=retries {
        if attempt > 0 {
            let backoff = (2u64.pow(attempt)) as f64 + rand::random::<f64>();
            sleep(Duration::from_secs_f64(backoff)).await;
        }

        let result = client
            .post(OPENROUTER_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(Duration::from_secs_f64(timeout_secs))
            .json(&serde_json::json!({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }))
            .send()
            .await;

        let response = match result {
            Ok(r) => r,
            Err(_) if attempt < retries => continue,
            Err(e) => return format!("[Error: Connection failed for {model}: {e}]"),
        };

        if response.status() != 200 {
            if attempt < retries {
                continue;
            }
            return format!("[Error: HTTP {} from {model}]", response.status());
        }

        let data: Value = match response.json().await {
            Ok(d) => d,
            Err(_) if attempt < retries => continue,
            Err(_) => return format!("[Error: Invalid JSON response from {model}]"),
        };

        if let Some(err) = data.get("error") {
            if attempt < retries {
                continue;
            }
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return format!("[Error: {msg}]");
        }

        let choices = match data.get("choices").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ if attempt < retries => continue,
            _ => return format!("[Error: No response from {model}]"),
        };

        let content = choices[0]
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        if content.trim().is_empty() {
            // Check for reasoning-only response
            let reasoning = choices[0]
                .get("message")
                .and_then(|m| m.get("reasoning"))
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if !reasoning.trim().is_empty() {
                if attempt < retries {
                    continue;
                }
                let preview = &reasoning[..reasoning.len().min(150)];
                return format!(
                    "[Model still thinking - needs more tokens. Partial reasoning: {preview}...]"
                );
            }
            if attempt < retries {
                continue;
            }
            return format!("[No response from {model} after {} attempts]", retries + 1);
        }

        let content = strip_think_blocks(content);

        if let Some(tracker) = cost_tracker {
            if let Some(cost) = data
                .get("usage")
                .and_then(|u| u.get("cost"))
                .and_then(|c| c.as_f64())
            {
                tracker.add(cost);
            }
        }

        return content;
    }

    format!("[Error: Failed to get response from {model}]")
}

/// Query Google AI Studio directly (fallback for Gemini models).
pub async fn query_google_ai_studio(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    timeout_secs: f64,
    retries: u32,
) -> String {
    let mut contents = Vec::new();
    let mut system_instruction: Option<String> = None;

    for msg in messages {
        match msg.role.as_str() {
            "system" => system_instruction = Some(msg.content.clone()),
            "user" => contents.push(serde_json::json!({
                "role": "user",
                "parts": [{"text": &msg.content}]
            })),
            "assistant" => contents.push(serde_json::json!({
                "role": "model",
                "parts": [{"text": &msg.content}]
            })),
            _ => {}
        }
    }

    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        }
    });

    if let Some(ref sys) = system_instruction {
        body["systemInstruction"] = serde_json::json!({"parts": [{"text": sys}]});
    }

    let url = format!("{GOOGLE_AI_STUDIO_URL}/{model}:generateContent?key={api_key}");

    for attempt in 0..=retries {
        if attempt > 0 {
            let backoff = (2u64.pow(attempt)) as f64 + rand::random::<f64>();
            sleep(Duration::from_secs_f64(backoff)).await;
        }

        let result = client
            .post(&url)
            .timeout(Duration::from_secs_f64(timeout_secs))
            .json(&body)
            .send()
            .await;

        let response = match result {
            Ok(r) => r,
            Err(_) if attempt < retries => continue,
            Err(_) => return format!("[Error: Request failed for AI Studio {model}]"),
        };

        if response.status() != 200 {
            if attempt < retries {
                continue;
            }
            return format!("[Error: HTTP {} from AI Studio {model}]", response.status());
        }

        let data: Value = match response.json().await {
            Ok(d) => d,
            Err(_) if attempt < retries => continue,
            Err(_) => return format!("[Error: Invalid JSON from AI Studio {model}]"),
        };

        if data.get("error").is_some() {
            if attempt < retries {
                continue;
            }
            let msg = data["error"]
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return format!("[Error: {msg}]");
        }

        let candidates = match data.get("candidates").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ if attempt < retries => continue,
            _ => return format!("[Error: No candidates from AI Studio {model}]"),
        };

        let text = candidates[0]
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|p| p.first())
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        if text.trim().is_empty() {
            if attempt < retries {
                continue;
            }
            return format!(
                "[No response from AI Studio {model} after {} attempts]",
                retries + 1
            );
        }

        return text.to_string();
    }

    format!("[Error: Failed to get response from AI Studio {model}]")
}

/// Query Zhipu bigmodel.cn directly (fallback for GLM models).
pub async fn query_bigmodel(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    timeout_secs: f64,
    retries: u32,
) -> String {
    let mut max_tokens = max_tokens;
    let mut timeout_secs = timeout_secs;

    if is_thinking_model(model) {
        max_tokens = max_tokens.max(2500);
        timeout_secs = timeout_secs.max(300.0);
    }

    for attempt in 0..=retries {
        if attempt > 0 {
            let backoff = (2u64.pow(attempt)) as f64 + rand::random::<f64>();
            sleep(Duration::from_secs_f64(backoff)).await;
        }

        let result = client
            .post(BIGMODEL_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(Duration::from_secs_f64(timeout_secs))
            .json(&serde_json::json!({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }))
            .send()
            .await;

        let response = match result {
            Ok(r) => r,
            Err(_) if attempt < retries => continue,
            Err(e) => return format!("[Error: Connection failed for bigmodel {model}: {e}]"),
        };

        if response.status() != 200 {
            if attempt < retries {
                continue;
            }
            return format!("[Error: HTTP {} from bigmodel {model}]", response.status());
        }

        let data: Value = match response.json().await {
            Ok(d) => d,
            Err(_) if attempt < retries => continue,
            Err(_) => return format!("[Error: Invalid JSON response from bigmodel {model}]"),
        };

        if let Some(err) = data.get("error") {
            if attempt < retries {
                continue;
            }
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return format!("[Error: {msg}]");
        }

        let choices = match data.get("choices").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ if attempt < retries => continue,
            _ => return format!("[Error: No response from bigmodel {model}]"),
        };

        let content = choices[0]
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        if content.trim().is_empty() {
            let reasoning = choices[0]
                .get("message")
                .and_then(|m| m.get("reasoning"))
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if !reasoning.trim().is_empty() {
                if attempt < retries {
                    continue;
                }
                let preview = &reasoning[..reasoning.len().min(150)];
                return format!(
                    "[Model still thinking - needs more tokens. Partial reasoning: {preview}...]"
                );
            }
            if attempt < retries {
                continue;
            }
            return format!(
                "[No response from bigmodel {model} after {} attempts]",
                retries + 1
            );
        }

        return strip_think_blocks(content);
    }

    format!("[Error: Failed to get response from bigmodel {model}]")
}

/// Query Moonshot directly (fallback for Kimi models).
pub async fn query_moonshot(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    timeout_secs: f64,
    retries: u32,
) -> String {
    let mut max_tokens = max_tokens;
    let mut timeout_secs = timeout_secs;

    if is_thinking_model(model) {
        max_tokens = max_tokens.max(2500);
        timeout_secs = timeout_secs.max(300.0);
    }

    for attempt in 0..=retries {
        if attempt > 0 {
            let backoff = (2u64.pow(attempt)) as f64 + rand::random::<f64>();
            sleep(Duration::from_secs_f64(backoff)).await;
        }

        let result = client
            .post(MOONSHOT_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(Duration::from_secs_f64(timeout_secs))
            .json(&serde_json::json!({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }))
            .send()
            .await;

        let response = match result {
            Ok(r) => r,
            Err(_) if attempt < retries => continue,
            Err(e) => return format!("[Error: Connection failed for moonshot {model}: {e}]"),
        };

        if response.status() != 200 {
            if attempt < retries {
                continue;
            }
            return format!("[Error: HTTP {} from moonshot {model}]", response.status());
        }

        let data: Value = match response.json().await {
            Ok(d) => d,
            Err(_) if attempt < retries => continue,
            Err(_) => return format!("[Error: Invalid JSON response from moonshot {model}]"),
        };

        if let Some(err) = data.get("error") {
            if attempt < retries {
                continue;
            }
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return format!("[Error: {msg}]");
        }

        let choices = match data.get("choices").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ if attempt < retries => continue,
            _ => return format!("[Error: No response from moonshot {model}]"),
        };

        let content = choices[0]
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        if content.trim().is_empty() {
            let reasoning = choices[0]
                .get("message")
                .and_then(|m| m.get("reasoning"))
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if !reasoning.trim().is_empty() {
                if attempt < retries {
                    continue;
                }
                let preview = &reasoning[..reasoning.len().min(150)];
                return format!(
                    "[Model still thinking - needs more tokens. Partial reasoning: {preview}...]"
                );
            }
            if attempt < retries {
                continue;
            }
            return format!(
                "[No response from moonshot {model} after {} attempts]",
                retries + 1
            );
        }

        return strip_think_blocks(content);
    }

    format!("[Error: Failed to get response from moonshot {model}]")
}

/// Query xAI directly (primary for Grok models).
pub async fn query_xai(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    timeout_secs: f64,
    retries: u32,
) -> String {
    let mut max_tokens = max_tokens;
    let mut timeout_secs = timeout_secs;

    if is_thinking_model(model) {
        max_tokens = max_tokens.max(2500);
        timeout_secs = timeout_secs.max(300.0);
    }

    for attempt in 0..=retries {
        if attempt > 0 {
            let backoff = (2u64.pow(attempt)) as f64 + rand::random::<f64>();
            sleep(Duration::from_secs_f64(backoff)).await;
        }

        let result = client
            .post(XAI_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(Duration::from_secs_f64(timeout_secs))
            .json(&serde_json::json!({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }))
            .send()
            .await;

        let response = match result {
            Ok(r) => r,
            Err(_) if attempt < retries => continue,
            Err(e) => return format!("[Error: Connection failed for xai {model}: {e}]"),
        };

        if response.status() != 200 {
            if attempt < retries {
                continue;
            }
            return format!("[Error: HTTP {} from xai {model}]", response.status());
        }

        let data: Value = match response.json().await {
            Ok(d) => d,
            Err(_) if attempt < retries => continue,
            Err(_) => return format!("[Error: Invalid JSON response from xai {model}]"),
        };

        if let Some(err) = data.get("error") {
            if attempt < retries {
                continue;
            }
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            return format!("[Error: {msg}]");
        }

        let choices = match data.get("choices").and_then(|c| c.as_array()) {
            Some(c) if !c.is_empty() => c,
            _ if attempt < retries => continue,
            _ => return format!("[Error: No response from xai {model}]"),
        };

        let content = choices[0]
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        if content.trim().is_empty() {
            if attempt < retries {
                continue;
            }
            return format!("[No response from xai {model} after {} attempts]", retries + 1);
        }

        return strip_think_blocks(content);
    }

    format!("[Error: Failed to get response from xai {model}]")
}

/// Query a model with streaming output — prints tokens as they arrive.
pub async fn query_model_streaming(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    timeout_secs: f64,
    cost_tracker: Option<&CostTracker>,
    output: &mut dyn Output,
) -> String {
    let mut full_content = Vec::new();
    let mut in_think_block = false;

    let result = client
        .post(OPENROUTER_URL)
        .header("Authorization", format!("Bearer {api_key}"))
        .timeout(Duration::from_secs_f64(timeout_secs))
        .json(&serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": true,
        }))
        .send()
        .await;

    let response = match result {
        Ok(r) if r.status() == 200 => r,
        Ok(r) => {
            let msg = format!("[Error: HTTP {} from {model}]", r.status());
            let _ = output.write_str(&format!("{}\n", msg));
            return msg;
        }
        Err(e) => {
            let msg = format!("[Error: Connection failed for {model}: {e}]");
            let _ = output.write_str(&format!("{}\n", msg));
            return msg;
        }
    };

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                let msg = format!("[Error: Stream read failed for {model}: {e}]");
                if full_content.is_empty() {
                    let _ = output.write_str(&format!("{}\n", msg));
                    return msg;
                }
                break;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&bytes));

        // Process complete lines from buffer
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim().to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data_str) = line.strip_prefix("data: ") {
                if data_str.trim() == "[DONE]" {
                    let _ = output.write_str("\n");
                    return full_content.join("");
                }

                if let Ok(data) = serde_json::from_str::<Value>(data_str) {
                    // Check for error
                    if let Some(err) = data.get("error") {
                        let msg = err
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown error");
                        let err_str = format!("[Error: {msg}]");
                        let _ = output.write_str(&format!("{}\n", err_str));
                        return err_str;
                    }

                    // Track cost from final usage chunk
                    if let Some(tracker) = cost_tracker {
                        if let Some(cost) = data
                            .get("usage")
                            .and_then(|u| u.get("cost"))
                            .and_then(|c| c.as_f64())
                        {
                            tracker.add(cost);
                        }
                    }

                    // Extract content delta
                    if let Some(content) = data
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|c| c.first())
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        if !content.is_empty() {
                            // Handle <think> blocks inline (DeepSeek-R1 format)
                            let mut delta_output = content.to_string();

                            if delta_output.contains("<think>") {
                                in_think_block = true;
                            }

                            if in_think_block {
                                if delta_output.contains("</think>") {
                                    in_think_block = false;
                                    delta_output = delta_output
                                        .split("</think>")
                                        .last()
                                        .unwrap_or("")
                                        .to_string();
                                } else {
                                    continue;
                                }
                            }

                            if !delta_output.is_empty() {
                                let _ = output.write_str(&delta_output);
                                full_content.push(delta_output);
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = output.write_str("\n");

    if full_content.is_empty() {
        let msg = format!("[No response from {model}]");
        let _ = output.write_str(&format!("{}\n", msg));
        return msg;
    }

    full_content.join("")
}

/// Async query for parallel phases.
pub async fn query_model_with_fallback(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    name: &str,
    fallback: Option<(&str, &str)>,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    max_tokens: u32,
    retries: u32,
    cost_tracker: Option<&CostTracker>,
) -> (String, String, String) {
    let model_name = model.split('/').next_back().unwrap_or(model).to_string();

    // For GLM: try bigmodel.cn directly first (more reliable than OpenRouter z-ai)
    if let Some(("zhipu", zhipu_model)) = fallback {
        if let Some(zapi_key) = zhipu_api_key {
            let primary_response = query_bigmodel(
                client,
                zapi_key,
                zhipu_model,
                messages,
                max_tokens,
                300.0,
                retries,
            )
            .await;
            if !primary_response.starts_with('[') {
                return (name.to_string(), zhipu_model.to_string(), primary_response);
            }
            // bigmodel.cn failed — fall through to OpenRouter
        }
    }

    // For Grok: try xAI directly first
    if let Some(("xai", xai_model)) = fallback {
        if let Some(xapi_key) = xai_api_key {
            let primary_response = query_xai(
                client,
                xapi_key,
                xai_model,
                messages,
                max_tokens,
                300.0,
                retries,
            )
            .await;
            if !primary_response.starts_with('[') {
                return (name.to_string(), xai_model.to_string(), primary_response);
            }
            // xAI failed — fall through to OpenRouter
        }
    }

    // For Gemini: try Google AI Studio directly first
    if let Some(("google", google_model)) = fallback {
        if let Some(gapi_key) = google_api_key {
            let primary_response = query_google_ai_studio(
                client,
                gapi_key,
                google_model,
                messages,
                max_tokens,
                300.0,
                retries,
            )
            .await;
            if !primary_response.starts_with('[') {
                return (name.to_string(), google_model.to_string(), primary_response);
            }
            // Google AI Studio failed — fall through to OpenRouter
        }
    }

    let response = query_model(
        client,
        api_key,
        model,
        messages,
        max_tokens,
        300.0,
        retries,
        cost_tracker,
    )
    .await;

    // If OpenRouter succeeded, return
    if !response.starts_with('[') {
        return (name.to_string(), model_name, response);
    }

    // Try fallback (Moonshot.cn direct API)
    if let Some(("moonshot", moonshot_model)) = fallback {
        if let Some(mapi_key) = moonshot_api_key {
            let fb_response = query_moonshot(
                client,
                mapi_key,
                moonshot_model,
                messages,
                max_tokens,
                300.0,
                retries,
            )
            .await;
            return (name.to_string(), moonshot_model.to_string(), fb_response);
        }
    }

    (name.to_string(), model_name, response)
}

/// Async query for parallel phases.
pub async fn query_model_async(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
    name: &str,
    fallback: Option<(&str, &str)>,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    max_tokens: u32,
    retries: u32,
    cost_tracker: Option<&CostTracker>,
) -> (String, String, String) {
    query_model_with_fallback(
        client,
        api_key,
        model,
        messages,
        name,
        fallback,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        xai_api_key,
        max_tokens,
        retries,
        cost_tracker,
    )
    .await
}

/// Parallel query panelists with shared messages.
pub async fn run_parallel(
    panelists: &[ModelEntry],
    messages: &[Message],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    max_tokens: u32,
    cost_tracker: Option<&CostTracker>,
    output: Option<&mut dyn Output>,
) -> Vec<(String, String, String)> {
    let client = Client::new();

    // We can't easily pass a &mut dyn Output into the tokio::spawn tasks because it's not 'static
    // and not thread-safe for parallel access without a Mutex.
    // However, the Python implementation seemed to just rely on sys.stdout.
    // In our case, run_parallel only uses verbose to print if requested.
    // If output is provided, we'll collect results and then print them if we want it to be serialized.
    // But usually verbose=true means print-as-they-arrive.
    // Since they arrive in arbitrary order, and we are using tokio::spawn,
    // we should use a shared Mutex if we want multiple tasks to write to the same output.

    // Given the constraints and typical CLI usage, it's simpler to collect results and print
    // them at the end if we want them ordered, OR use a Mutex for immediate output.

    let handles: Vec<_> = panelists
        .iter()
        .enumerate()
        .map(|(idx, &(name, model, fallback))| {
            let client = client.clone();
            let api_key = api_key.to_string();
            let google_api_key = google_api_key.map(|s| s.to_string());
            let zhipu_api_key = zhipu_api_key.map(|s| s.to_string());
            let moonshot_api_key = moonshot_api_key.map(|s| s.to_string());
            let xai_api_key = xai_api_key.map(|s| s.to_string());
            let messages = messages.to_vec();
            let name = name.to_string();
            let model = model.to_string();
            let fallback_owned: Option<(String, String)> =
                fallback.map(|(p, m)| (p.to_string(), m.to_string()));
            let cost_tracker = cost_tracker.cloned();

            tokio::spawn(async move {
                let fallback_ref: Option<(&str, &str)> = fallback_owned
                    .as_ref()
                    .map(|(p, m)| (p.as_str(), m.as_str()));
                let result = query_model_with_fallback(
                    &client,
                    &api_key,
                    &model,
                    &messages,
                    &name,
                    fallback_ref,
                    google_api_key.as_deref(),
                    zhipu_api_key.as_deref(),
                    moonshot_api_key.as_deref(),
                    xai_api_key.as_deref(),
                    max_tokens,
                    2,
                    cost_tracker.as_ref(),
                )
                .await;

                (idx, result)
            })
        })
        .collect();

    let mut indexed_results: Vec<(usize, (String, String, String))> = Vec::new();
    for handle in handles {
        match handle.await {
            Ok((idx, result)) => indexed_results.push((idx, result)),
            Err(e) => {
                eprintln!("[Error: Task join failed: {e}]");
            }
        }
    }

    // Sort by original index to preserve order
    indexed_results.sort_by_key(|(idx, _)| *idx);
    let final_results: Vec<_> = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect();

    if let Some(out) = output {
        for (name, _, response) in &final_results {
            if !response.starts_with('[') {
                let _ = out.write_str(&format!("\n### {}\n", name));
                let _ = out.write_str(&format!("{}\n\n", response));
            }
        }
    }

    final_results
}

/// Parallel query panelists with different messages per panelist.
pub async fn run_parallel_with_different_messages(
    panelists: &[ModelEntry],
    messages_list: &[Vec<Message>],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    max_tokens: u32,
    cost_tracker: Option<&CostTracker>,
    output: Option<&mut dyn Output>,
) -> Vec<(String, String, String)> {
    assert_eq!(
        panelists.len(),
        messages_list.len(),
        "panelists and messages_list must have the same length"
    );
    let client = Client::new();

    let handles: Vec<_> = panelists
        .iter()
        .zip(messages_list.iter())
        .enumerate()
        .map(|(idx, (&(name, model, fallback), messages))| {
            let client = client.clone();
            let api_key = api_key.to_string();
            let google_api_key = google_api_key.map(|s| s.to_string());
            let zhipu_api_key = zhipu_api_key.map(|s| s.to_string());
            let moonshot_api_key = moonshot_api_key.map(|s| s.to_string());
            let xai_api_key = xai_api_key.map(|s| s.to_string());
            let messages = messages.to_vec();
            let name = name.to_string();
            let model = model.to_string();
            let fallback_owned: Option<(String, String)> =
                fallback.map(|(p, m)| (p.to_string(), m.to_string()));
            let cost_tracker = cost_tracker.cloned();

            tokio::spawn(async move {
                let fallback_ref: Option<(&str, &str)> = fallback_owned
                    .as_ref()
                    .map(|(p, m)| (p.as_str(), m.as_str()));
                let result = query_model_with_fallback(
                    &client,
                    &api_key,
                    &model,
                    &messages,
                    &name,
                    fallback_ref,
                    google_api_key.as_deref(),
                    zhipu_api_key.as_deref(),
                    moonshot_api_key.as_deref(),
                    xai_api_key.as_deref(),
                    max_tokens,
                    2,
                    cost_tracker.as_ref(),
                )
                .await;

                (idx, result)
            })
        })
        .collect();

    let mut indexed_results: Vec<(usize, (String, String, String))> = Vec::new();
    for handle in handles {
        match handle.await {
            Ok((idx, result)) => indexed_results.push((idx, result)),
            Err(e) => {
                eprintln!("[Error: Task join failed: {e}]");
            }
        }
    }

    // Sort by original index to preserve order
    indexed_results.sort_by_key(|(idx, _)| *idx);
    let final_results: Vec<_> = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect();

    if let Some(out) = output {
        for (name, _, response) in &final_results {
            if !response.starts_with('[') {
                let _ = out.write_str(&format!("\n### {}\n", name));
                let _ = out.write_str(&format!("{}\n\n", response));
            }
        }
    }

    final_results
}

/// Strip `<think>...</think>` blocks from content.
pub fn strip_think_blocks(content: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").unwrap());
    RE.replace_all(content, "").trim().to_string()
}

/// Classify question into best deliberation mode via LLM.
pub async fn classify_mode(
    client: &Client,
    api_key: &str,
    question: &str,
    cost_tracker: Option<&CostTracker>,
) -> String {
    let messages = vec![
        Message::system(
            "Pick the best deliberation mode for this question. Respond with ONLY the mode name.\n\n\
             quick: Factual questions, straightforward comparisons, single-dimension — just need parallel opinions\n\
             council: Complex trade-offs, multi-stakeholder, strategic decisions with many interacting variables\n\
             oxford: Binary decisions with clear for/against framing — \"should I X or Y?\"\n\
             redteam: Stress-testing a plan, decision, or strategy — \"what could go wrong with X?\"\n\
             socratic: Exposing hidden assumptions, probing beliefs — \"what am I not seeing about X?\"\n\
             discuss: Open-ended exploration, no clear decision needed — \"let's think about X\"\n\
             solo: Niche — only when the user explicitly wants one model in multiple roles\n\n\
             Default to council when unsure."
        ),
        Message::user(question),
    ];

    let response = query_model(
        client,
        api_key,
        crate::config::CLASSIFIER_MODEL,
        &messages,
        10,
        15.0,
        2,
        cost_tracker,
    )
    .await;

    let result = response
        .trim()
        .to_lowercase()
        .trim_end_matches('.')
        .to_string();

    let valid_modes = [
        "quick", "council", "oxford", "redteam", "socratic", "discuss", "solo",
    ];
    if valid_modes.contains(&result.as_str()) {
        result
    } else {
        "council".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_think_blocks_basic() {
        let input = "Hello <think>internal reasoning</think> World";
        assert_eq!(strip_think_blocks(input), "Hello  World");
    }

    #[test]
    fn test_strip_think_blocks_multiline() {
        let input = "Before\n<think>\nline1\nline2\n</think>\nAfter";
        assert_eq!(strip_think_blocks(input), "Before\n\nAfter");
    }

    #[test]
    fn test_strip_think_blocks_no_think() {
        let input = "Normal content without think blocks";
        assert_eq!(strip_think_blocks(input), input);
    }

    #[test]
    fn test_strip_think_blocks_multiple() {
        let input = "<think>first</think>middle<think>second</think>end";
        assert_eq!(strip_think_blocks(input), "middleend");
    }

    // SSE line parsing tests
    #[test]
    fn test_sse_data_line_parse() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        let data_str = line.strip_prefix("data: ").unwrap();
        let data: Value = serde_json::from_str(data_str).unwrap();
        let content = data["choices"][0]["delta"]["content"].as_str().unwrap();
        assert_eq!(content, "Hello");
    }

    #[test]
    fn test_sse_done_line() {
        let line = "data: [DONE]";
        let data_str = line.strip_prefix("data: ").unwrap();
        assert_eq!(data_str.trim(), "[DONE]");
    }

    #[test]
    fn test_sse_error_in_stream() {
        let line = r#"data: {"error":{"message":"Rate limited"}}"#;
        let data_str = line.strip_prefix("data: ").unwrap();
        let data: Value = serde_json::from_str(data_str).unwrap();
        assert!(data.get("error").is_some());
        assert_eq!(data["error"]["message"].as_str().unwrap(), "Rate limited");
    }

    #[test]
    fn test_sse_cost_tracking() {
        let line = r#"data: {"usage":{"cost":0.05}}"#;
        let data_str = line.strip_prefix("data: ").unwrap();
        let data: Value = serde_json::from_str(data_str).unwrap();
        let cost = data["usage"]["cost"].as_f64().unwrap();
        assert!((cost - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_sse_reasoning_field_ignored() {
        // OpenAI reasoning goes in delta.reasoning_details, not delta.content
        // Our parser only reads delta.content, so reasoning is auto-invisible
        let line = r#"data: {"choices":[{"delta":{"reasoning_details":"thinking...","content":"visible"}}]}"#;
        let data_str = line.strip_prefix("data: ").unwrap();
        let data: Value = serde_json::from_str(data_str).unwrap();
        // We only read content
        let content = data["choices"][0]["delta"]["content"].as_str().unwrap();
        assert_eq!(content, "visible");
        // reasoning_details is present but we don't use it
        assert!(data["choices"][0]["delta"]["reasoning_details"].is_string());
    }
}
