"""Quick mode: parallel queries, no debate, no judge."""

import asyncio
import json
import time
import yaml
from datetime import datetime

from .models import SessionResult, run_parallel


def run_quick(
    question: str,
    models: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    verbose: bool = True,
    format: str = "prose",
) -> SessionResult:
    """Run quick mode: parallel queries, no debate, no judge. Returns SessionResult."""
    start_time = time.time()
    cost_accumulator: list[float] = []

    messages = [{"role": "user", "content": question}]

    if verbose:
        print(f"(querying {len(models)} models in parallel...)")

    results = asyncio.run(run_parallel(
        models, messages, api_key, google_api_key,
        max_tokens=2000, cost_accumulator=cost_accumulator,
    ))

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    if verbose:
        print()

    failed = []
    for name, model_name, response in results:
        if response.startswith("["):
            failed.append(f"{model_name}: {response}")
        elif verbose:
            print(f"### {model_name}")
            print(response)
            print()

    if failed and verbose:
        print("Failures:")
        for f in failed:
            print(f"  - {f}")
        print()

    if verbose:
        print(f"({duration:.1f}s, ~${total_cost:.2f})")

    # Build output
    if format in ("json", "yaml"):
        structured = {
            "schema_version": "1.0",
            "question": question,
            "mode": "quick",
            "responses": [
                {
                    "model": model_name,
                    "content": response,
                }
                for name, model_name, response in results
                if not response.startswith("[")
            ],
            "errors": [
                {
                    "model": model_name,
                    "error": response,
                }
                for name, model_name, response in results
                if response.startswith("[")
            ],
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "models_used": [m.split("/")[-1] for _, m, _ in models],
                "duration_seconds": round(duration, 1),
                "estimated_cost_usd": total_cost,
            },
        }
        if not structured["errors"]:
            del structured["errors"]
        if format == "json":
            transcript = json.dumps(structured, indent=2, ensure_ascii=False)
        else:
            transcript = yaml.dump(structured, allow_unicode=True, default_flow_style=False)
    else:
        # Prose format
        parts = []
        for name, model_name, response in results:
            parts.append(f"### {model_name}\n{response}")
        transcript = "\n\n".join(parts)

    return SessionResult(transcript=transcript, cost=total_cost, duration=duration)
