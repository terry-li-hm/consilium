"""Quick mode: parallel queries, no debate, no judge."""

import asyncio
import json
import time
import yaml
from datetime import datetime

from .models import SessionResult, run_parallel, query_model_async

import httpx


async def _run_quick_streaming(
    question: str,
    models: list[tuple[str, str, tuple[str, str] | None]],
    api_key: str,
    google_api_key: str | None = None,
    max_tokens: int = 2000,
    cost_accumulator: list[float] | None = None,
    verbose: bool = True,
) -> list[tuple[str, str, str]]:
    """Query models via parallel SSE streams and display sequentially."""
    messages = [{"role": "user", "content": question}]
    openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

    states: list[dict] = []
    for name, model, fallback in models:
        states.append(
            {
                "name": name,
                "model": model,
                "fallback": fallback,
                "model_name": model.split("/")[-1],
                "response_parts": [],
                "queue": asyncio.Queue(),
                "done": asyncio.Event(),
                "stream_error": None,
                "in_think_block": False,
            }
        )

    async def _stream_one(state: dict, client: httpx.AsyncClient) -> None:
        try:
            async with client.stream(
                "POST",
                openrouter_url,
                json={
                    "model": state["model"],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    state["stream_error"] = f"[Error: HTTP {response.status_code} from {state['model_name']}]"
                    return

                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        error = data["error"]
                        if isinstance(error, dict):
                            error = error.get("message", error)
                        state["stream_error"] = f"[Error: {error}]"
                        break

                    if cost_accumulator is not None and "usage" in data:
                        cost = data["usage"].get("cost")
                        if cost is not None:
                            cost_accumulator.append(float(cost))

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        continue

                    if "<think>" in content:
                        state["in_think_block"] = True
                    if state["in_think_block"]:
                        if "</think>" in content:
                            state["in_think_block"] = False
                            content = content.split("</think>", 1)[-1]
                        else:
                            continue

                    if content:
                        state["response_parts"].append(content)
                        await state["queue"].put(content)
        except (httpx.RequestError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            state["stream_error"] = f"[Error: Connection failed for {state['model_name']}: {e}]"
        except Exception as e:
            state["stream_error"] = f"[Error: {e}]"

    async def _stream_with_fallback(state: dict, client: httpx.AsyncClient) -> None:
        try:
            await _stream_one(state, client)

            # Fallback only if streaming failed before producing any content.
            if state["stream_error"] and not state["response_parts"]:
                name, model_name, response = await query_model_async(
                    client,
                    state["model"],
                    messages,
                    state["name"],
                    state["fallback"],
                    google_api_key,
                    max_tokens=max_tokens,
                    cost_accumulator=cost_accumulator,
                )
                state["name"] = name
                state["model_name"] = model_name
                state["response_parts"] = [response]
                await state["queue"].put(response)
        finally:
            state["done"].set()

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180.0,
    ) as client:
        tasks = [asyncio.create_task(_stream_with_fallback(state, client)) for state in states]

        if verbose:
            for state in states:
                print(f"### {state['model_name']}")
                while True:
                    printed_any = False
                    while True:
                        try:
                            token = state["queue"].get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        print(token, end="", flush=True)
                        printed_any = True

                    if state["done"].is_set() and state["queue"].empty():
                        break
                    if not printed_any:
                        await asyncio.sleep(0.02)
                print(flush=True)

        await asyncio.gather(*tasks, return_exceptions=True)

    out: list[tuple[str, str, str]] = []
    for i, state in enumerate(states):
        name, model, _ = models[i]
        response = "".join(state["response_parts"]).strip()
        if not response:
            if state["stream_error"]:
                response = state["stream_error"]
            else:
                response = f"[No response from {state['model_name']}]"
        out.append((name, state["model_name"], response))
    return out


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

    if verbose:
        print(f"(querying {len(models)} models in parallel...)")
        print(flush=True)

    results = asyncio.run(_run_quick_streaming(
        question, models, api_key, google_api_key,
        max_tokens=4000, cost_accumulator=cost_accumulator,
        verbose=verbose,
    ))

    duration = time.time() - start_time
    total_cost = round(sum(cost_accumulator), 4) if cost_accumulator else 0.0

    # Print failures
    failed = [f"{mn}: {r}" for _, mn, r in results if r.startswith("[")]
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
