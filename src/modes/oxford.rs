//! Oxford debate mode: structured for/against with rebuttals and verdict.

use crate::api::{query_model, run_parallel_with_different_messages};
use crate::config::{
    resolved_judge_model, sanitize_speaker_content, CostTracker, Message, ModelEntry, SessionResult,
};
use crate::prompts::{
    oxford_closing_system, oxford_constructive_system, oxford_judge_prior, oxford_judge_verdict,
    oxford_motion_transform, oxford_rebuttal_system,
};
use crate::session::Output;
use rand::seq::SliceRandom;
use reqwest::Client;
use std::time::Instant;

pub async fn run_oxford(
    question: &str,
    models: &[ModelEntry],
    api_key: &str,
    google_api_key: Option<&str>,
    zhipu_api_key: Option<&str>,
    moonshot_api_key: Option<&str>,
    xai_api_key: Option<&str>,
    motion_override: Option<String>,
    _format: &str,
    timeout: f64,
    output: &mut dyn Output,
) -> SessionResult {
    let start = Instant::now();
    let cost_tracker = CostTracker::new();
    let client = Client::new();
    let judge_model = resolved_judge_model();

    let mut transcript_parts = Vec::new();

    let _ = output.write_str("============================================================\n");
    let _ = output.write_str("OXFORD DEBATE\n");
    let _ = output.write_str("============================================================\n\n");

    // Phase 1: MOTION
    let _ = output.begin_phase("MOTION");
    let motion = if let Some(m) = motion_override {
        m
    } else {
        let motion_t0 = Instant::now();
        let _ = output.begin_participant("Judge (Claude)");
        let _ = output.write_str("## Motion\n\n");
        let m_messages = vec![
            Message::system(oxford_motion_transform(question)),
            Message::user(question),
        ];
        let m = query_model(
            &client,
            api_key,
            judge_model.as_str(),
            &m_messages,
            100,
            timeout,
            2,
            Some(&cost_tracker),
        )
        .await;
        let m = m.trim().trim_matches('"').to_string();
        let _ = output.write_str(&format!("{}\n\n", m));
        let _ =
            output.end_participant("Judge (Claude)", &m, motion_t0.elapsed().as_millis() as u64);
        m
    };
    transcript_parts.push(format!("## Motion\n\n{motion}"));

    // Random side assignment
    let mut sides = models.to_vec();
    let mut rng = rand::thread_rng();
    sides.shuffle(&mut rng);

    let (prop_name, prop_model, prop_fallback) = sides[0];
    let (opp_name, opp_model, opp_fallback) = sides[1];

    let _ = output.write_str(&format!("Proposition (FOR): {prop_name}\n"));
    let _ = output.write_str(&format!("Opposition (AGAINST): {opp_name}\n\n"));
    transcript_parts.push(format!(
        "**Proposition:** {prop_name} | **Opposition:** {opp_name}"
    ));

    // Phase 2: PRIOR
    let _ = output.begin_phase("PRIOR");
    let prior_t0 = Instant::now();
    let _ = output.begin_participant("Judge (Claude)");
    let _ = output.write_str("## Prior\n### Judge (Claude)\n");
    let prior_messages = vec![
        Message::system(oxford_judge_prior(&motion)),
        Message::user(format!("Motion: {motion}")),
    ];
    let prior_response = query_model(
        &client,
        api_key,
        judge_model.as_str(),
        &prior_messages,
        200,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;
    let _ = output.write_str(&format!("{}\n\n", prior_response));
    let _ = output.end_participant(
        "Judge (Claude)",
        &prior_response,
        prior_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!("## Prior\n\n### Judge (Claude)\n{prior_response}"));

    // Phase 3: CONSTRUCTIVE (parallel)
    let _ = output.begin_phase("CONSTRUCTIVE SPEECHES");
    let _ = output.write_str("## Constructive Speeches\n(both sides arguing in parallel...)\n");
    let prop_system = oxford_constructive_system(prop_name, "FOR", &motion);
    let opp_system = oxford_constructive_system(opp_name, "AGAINST", &motion);

    let constructive_results = run_parallel_with_different_messages(
        &[
            (prop_name, prop_model, prop_fallback),
            (opp_name, opp_model, opp_fallback),
        ],
        &[
            vec![
                Message::system(prop_system),
                Message::user(format!("Argue FOR the motion: {motion}")),
            ],
            vec![
                Message::system(opp_system),
                Message::user(format!("Argue AGAINST the motion: {motion}")),
            ],
        ],
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        xai_api_key,
        800,
        Some(&cost_tracker),
        None,
    )
    .await;

    let prop_constructive = constructive_results[0].2.clone();
    let opp_constructive = constructive_results[1].2.clone();
    for (name, _, response) in &constructive_results {
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(name, response, 0);
    }

    transcript_parts.push(format!(
        "## Constructive Speeches\n\n### {prop_name} (Proposition)\n{prop_constructive}"
    ));
    transcript_parts.push(format!("### {opp_name} (Opposition)\n{opp_constructive}"));

    // Phase 4: REBUTTAL (parallel)
    let _ = output.begin_phase("REBUTTALS");
    let _ = output.write_str("## Rebuttals\n(both sides rebutting in parallel...)\n");
    let prop_rebuttal_system = oxford_rebuttal_system(
        prop_name,
        "FOR",
        &motion,
        &sanitize_speaker_content(&opp_constructive),
    );
    let opp_rebuttal_system = oxford_rebuttal_system(
        opp_name,
        "AGAINST",
        &motion,
        &sanitize_speaker_content(&prop_constructive),
    );

    let rebuttal_results = run_parallel_with_different_messages(
        &[
            (prop_name, prop_model, prop_fallback),
            (opp_name, opp_model, opp_fallback),
        ],
        &[
            vec![
                Message::system(prop_rebuttal_system),
                Message::user("Rebut the opposition's arguments."),
            ],
            vec![
                Message::system(opp_rebuttal_system),
                Message::user("Rebut the proposition's arguments."),
            ],
        ],
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        xai_api_key,
        600,
        Some(&cost_tracker),
        None,
    )
    .await;

    let prop_rebuttal = rebuttal_results[0].2.clone();
    let opp_rebuttal = rebuttal_results[1].2.clone();
    for (name, _, response) in &rebuttal_results {
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(name, response, 0);
    }

    transcript_parts.push(format!(
        "## Rebuttals\n\n### {prop_name} (Proposition rebuttal)\n{prop_rebuttal}"
    ));
    transcript_parts.push(format!(
        "### {opp_name} (Opposition rebuttal)\n{opp_rebuttal}"
    ));

    // Phase 5: CLOSING (parallel)
    let _ = output.begin_phase("CLOSING STATEMENTS");
    let _ = output.write_str("## Closing Statements\n(both sides closing in parallel...)\n");
    let prop_closing_system = oxford_closing_system(prop_name, "FOR", &motion);
    let opp_closing_system = oxford_closing_system(opp_name, "AGAINST", &motion);

    let closing_results = run_parallel_with_different_messages(
        &[
            (prop_name, prop_model, prop_fallback),
            (opp_name, opp_model, opp_fallback),
        ],
        &[
            vec![
                Message::system(prop_closing_system),
                Message::user("Give your closing statement."),
            ],
            vec![
                Message::system(opp_closing_system),
                Message::user("Give your closing statement."),
            ],
        ],
        api_key,
        google_api_key,
        zhipu_api_key,
        moonshot_api_key,
        xai_api_key,
        400,
        Some(&cost_tracker),
        None,
    )
    .await;

    let prop_closing = closing_results[0].2.clone();
    let opp_closing = closing_results[1].2.clone();
    for (name, _, response) in &closing_results {
        let _ = output.write_str(&format!("### {name}\n{response}\n\n"));
        let _ = output.end_participant(name, response, 0);
    }

    transcript_parts.push(format!(
        "## Closing Statements\n\n### {prop_name} (Proposition closing)\n{prop_closing}"
    ));
    transcript_parts.push(format!(
        "## Closing Statements\n\n### {opp_name} (Opposition closing)\n{opp_closing}"
    ));

    // Phase 6: VERDICT
    let verdict_t0 = Instant::now();
    let _ = output.begin_participant("Judge (Claude)");
    let _ = output.begin_phase("JUDGMENT");
    let _ = output.write_str("## Verdict\n### Judge (Claude)\n");
    let debate_transcript = format!(
        "### Proposition ({prop_name}) — Constructive\n{prop_constructive}\n\n### Opposition ({opp_name}) — Constructive\n{opp_constructive}\n\n### Proposition ({prop_name}) — Rebuttal\n{prop_rebuttal}\n\n### Opposition ({opp_name}) — Rebuttal\n{opp_rebuttal}\n\n### Proposition ({prop_name}) — Closing\n{prop_closing}\n\n### Opposition ({opp_name}) — Closing\n{opp_closing}",
        prop_constructive = sanitize_speaker_content(&prop_constructive),
        opp_constructive = sanitize_speaker_content(&opp_constructive),
        prop_rebuttal = sanitize_speaker_content(&prop_rebuttal),
        opp_rebuttal = sanitize_speaker_content(&opp_rebuttal),
        prop_closing = sanitize_speaker_content(&prop_closing),
        opp_closing = sanitize_speaker_content(&opp_closing)
    );

    let verdict_system = oxford_judge_verdict(&motion, prop_name, opp_name, &debate_transcript);
    let verdict_messages = vec![
        Message::system(verdict_system),
        Message::user(format!(
            "Judge this debate on: {motion}\n\nYour prior assessment:\n{prior_response}",
            prior_response = sanitize_speaker_content(&prior_response)
        )),
    ];

    let verdict = query_model(
        &client,
        api_key,
        judge_model.as_str(),
        &verdict_messages,
        1000,
        timeout,
        2,
        Some(&cost_tracker),
    )
    .await;

    let _ = output.write_str(&format!("{}\n\n", verdict));
    let _ = output.end_participant(
        "Judge (Claude)",
        &verdict,
        verdict_t0.elapsed().as_millis() as u64,
    );
    transcript_parts.push(format!("## Verdict\n\n### Judge (Claude)\n{verdict}"));

    let duration = start.elapsed().as_secs_f64();
    let cost = (cost_tracker.total() * 10000.0).round() / 10000.0;

    let _ = output.write_str(&format!("({:.1}s, ~${:.2})\n", duration, cost));

    SessionResult {
        transcript: transcript_parts.join("\n\n"),
        cost,
        duration,
        failures: None,
    }
}
