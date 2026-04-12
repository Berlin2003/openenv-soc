"""
inference.py — OpenEnv-SOC Baseline Agent

Runs an OpenAI-compatible LLM agent against all 3 SOC tasks using tool calling.
Reads HF_TOKEN, API_BASE_URL, and MODEL_NAME from environment.
Emits structured [START]/[STEP]/[END] logs for the hackathon validator.

Usage:
    export HF_TOKEN="your_token"
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    python inference.py --env-url http://localhost:7860
"""
import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Tuple

import httpx
from openai import OpenAI

from client import SOCEnvClient
from models import SOCAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK = "soc-env"
TASKS = ["easy", "medium", "hard"]
MAX_AGENT_STEPS = 30  # safety cap per task (env has its own per-task limits)
SUCCESS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Structured logging — [START] / [STEP] / [END]
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] {json.dumps({"task": task, "env": env, "model": model})}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    print(
        f'[STEP] {json.dumps({"step": step, "action": action, "reward": reward, "done": done, "error": error})}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f'[END] {json.dumps({"success": success, "steps": steps, "score": score, "rewards": rewards})}',
        flush=True,
    )


# ---------------------------------------------------------------------------
# Server readiness check
# ---------------------------------------------------------------------------
def wait_for_server(url: str, timeout: int = 120, interval: int = 3) -> bool:
    """Poll the server /health endpoint until it responds or timeout."""
    deadline = time.time() + timeout
    health_url = url.rstrip("/") + "/health"
    print(f"[DEBUG] Waiting for server at {health_url} ...", flush=True)
    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                print("[DEBUG] Server is ready.", flush=True)
                return True
        except Exception:
            pass
        time.sleep(interval)
    print(f"[DEBUG] Server not reachable after {timeout}s — proceeding anyway.", flush=True)
    return False


# ---------------------------------------------------------------------------
# OpenAI Tool definitions
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_logs",
            "description": "Search corporate log sources (firewall, endpoint, activedirectory) for events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["firewall", "endpoint", "activedirectory"],
                        "description": "Log source to query",
                    },
                    "query": {"type": "string", "description": "Keyword to search in log messages"},
                },
                "required": ["source", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_threat_intel",
            "description": "Look up an IP, domain, or filename in threat intelligence databases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "IP, domain, or file hash"},
                },
                "required": ["indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block_ip",
            "description": "Block an external IP address on the perimeter firewall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip_address": {"type": "string"},
                },
                "required": ["ip_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "isolate_host",
            "description": "Quarantine an internal host from the corporate network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                },
                "required": ["hostname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revoke_session",
            "description": "Revoke all active sessions for a compromised user account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                },
                "required": ["username"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_alert",
            "description": (
                "TERMINAL ACTION — Close the incident as resolved or false positive. "
                "Call only when the incident is fully investigated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "alert_id": {"type": "string"},
                    "resolution_summary": {"type": "string"},
                    "is_false_positive": {"type": "boolean"},
                },
                "required": ["alert_id", "resolution_summary", "is_false_positive"],
            },
        },
    },
]


def _build_action(name: str, args: dict) -> SOCAction:
    """Map tool call name -> unified SOCAction model."""
    tool_to_action = {
        "query_logs":         lambda a: SOCAction(action_type="query_logs", source=a["source"], query=a["query"]),
        "query_threat_intel": lambda a: SOCAction(action_type="query_threat_intel", indicator=a["indicator"]),
        "block_ip":           lambda a: SOCAction(action_type="block_ip", ip_address=a["ip_address"]),
        "isolate_host":       lambda a: SOCAction(action_type="isolate_host", hostname=a["hostname"]),
        "revoke_session":     lambda a: SOCAction(action_type="revoke_session", username=a["username"]),
        "close_alert":        lambda a: SOCAction(
            action_type="close_alert",
            alert_id=a["alert_id"],
            resolution_summary=a["resolution_summary"],
            is_false_positive=a.get("is_false_positive", False),
        ),
    }
    fn = tool_to_action.get(name)
    if not fn:
        raise ValueError(f"Unknown tool: {name}")
    return fn(args)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def get_model_response(client: OpenAI, model: str, messages: list) -> object:
    """Call the LLM and return the response message."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
    )


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------
async def run_task(
    base_url: str, task: str, model: str, client: OpenAI
) -> Tuple[float, List[float]]:
    """Run one task with structured logging. Returns (score, rewards)."""

    system_prompt = (
        "You are an elite Tier-2 Security Operations Center (SOC) Analyst. "
        "You will be given a security alert. Investigate it by querying logs and threat intelligence. "
        "Take precise, targeted remediation actions (block IPs, isolate hosts, revoke sessions). "
        "Avoid blocking or isolating systems without evidence. "
        "When you are confident the incident is fully resolved, call close_alert with a clear summary. "
        "Be systematic: investigate first, remediate second, close last."
    )

    rewards: List[float] = []
    step_num = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=model)

    try:
        async with SOCEnvClient(base_url=base_url, task=task) as env:
            try:
                obs = await env.reset()
            except Exception as exc:
                print(f"[DEBUG] Failed to reset environment: {exc}", flush=True)
                return score, rewards

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Active Alerts:\n{obs.model_dump_json(indent=2)}\n\n"
                        "Investigate and remediate this incident using your available tools."
                    ),
                },
            ]

            done = False
            while not done and step_num < MAX_AGENT_STEPS:
                # --- LLM call ---
                try:
                    response = get_model_response(client, model, messages)
                except Exception as exc:
                    print(f"[DEBUG] Model request failed: {exc}", flush=True)
                    log_step(step=step_num + 1, action="llm_call", reward=0.0, done=False, error=str(exc))
                    break

                msg = response.choices[0].message
                messages.append(msg.model_dump(exclude_none=True))

                if not msg.tool_calls:
                    log_step(step=step_num + 1, action="no_tool_call", reward=0.0, done=False, error="LLM returned no tool call")
                    break

                tc = msg.tool_calls[0]
                name = tc.function.name

                # --- Parse arguments ---
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError) as exc:
                    step_num += 1
                    error_msg = f"Invalid JSON: {tc.function.arguments}"
                    log_step(step=step_num, action=name, reward=0.0, done=False, error=error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: {error_msg}",
                    })
                    continue

                action_str = f"{name}({json.dumps(args)})"

                # --- Build action and step ---
                try:
                    action = _build_action(name, args)
                    result = await env.step(action)
                except ValueError as exc:
                    step_num += 1
                    log_step(step=step_num, action=action_str, reward=0.0, done=False, error=str(exc))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: {exc}",
                    })
                    continue
                except Exception as exc:
                    step_num += 1
                    log_step(step=step_num, action=action_str, reward=0.0, done=False, error=str(exc))
                    break

                step_num += 1
                reward = result.reward if result.reward is not None else 0.0
                done = result.done
                rewards.append(reward)

                log_step(step=step_num, action=action_str, reward=reward, done=done, error=None)

                # On terminal close_alert step, reward IS the grader score
                if done and name == "close_alert":
                    score = max(0.0, min(1.0, reward))

                # Feed result back to LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.observation.last_action_result,
                })
                if not done:
                    messages.append({
                        "role": "user",
                        "content": f"Updated environment state:\n{result.observation.model_dump_json(indent=2)}",
                    })

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task} exception: {exc}", flush=True)

    finally:
        log_end(success=success, steps=step_num, score=score, rewards=rewards)

    return score, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main(env_url: str) -> None:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
    api_base_url = os.environ.get("API_BASE_URL") or ""
    model_name = os.environ.get("MODEL_NAME") or "unknown"

    if not api_key or model_name == "unknown":
        print("[DEBUG] Missing env vars: HF_TOKEN/OPENAI_API_KEY or MODEL_NAME", flush=True)
        for task in TASKS:
            log_start(task=task, env=BENCHMARK, model=model_name)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    print(f"[DEBUG] API: {api_base_url or 'default'} | Model: {model_name}", flush=True)

    # Wait for the environment server
    wait_for_server(env_url)

    # Initialize OpenAI client (synchronous — matches hackathon sample)
    try:
        oai = OpenAI(api_key=api_key, base_url=api_base_url if api_base_url else None)
    except Exception as exc:
        print(f"[DEBUG] Failed to init OpenAI client: {exc}", flush=True)
        for task in TASKS:
            log_start(task=task, env=BENCHMARK, model=model_name)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    all_scores = []
    for task in TASKS:
        try:
            task_score, _ = await run_task(env_url, task, model_name, oai)
            all_scores.append(task_score)
        except Exception as exc:
            print(f"[DEBUG] Task {task} outer exception: {exc}", flush=True)
            all_scores.append(0.0)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv-SOC Inference")
    parser.add_argument("--env-url", default="http://localhost:7860")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.env_url))
    except Exception as exc:
        print(f"[DEBUG] Fatal: {exc}", flush=True)
    sys.exit(0)
