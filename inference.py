"""
inference.py — OpenEnv-SOC Baseline Agent

Runs an OpenAI LLM agent against all 3 SOC tasks using tool calling.
Reads HF_TOKEN, API_BASE_URL, and MODEL_NAME from environment.
Prints a reproducible score table.

Usage:
    export HF_TOKEN="your_token"
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    # Start server: uvicorn server.app:app --host 0.0.0.0 --port 7860
    python inference.py --env-url http://localhost:7860
"""
import argparse
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

from client import SOCEnvClient
from models import SOCAction

# ---------------------------------------------------------------------------
# OpenAI Tool definitions (auto-derived from the action models' JSON schemas)
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
    """Map tool call name → unified SOCAction model."""
    tool_to_action = {
        "query_logs":       lambda a: SOCAction(action_type="query_logs",       source=a["source"],       query=a["query"]),
        "query_threat_intel": lambda a: SOCAction(action_type="query_threat_intel", indicator=a["indicator"]),
        "block_ip":         lambda a: SOCAction(action_type="block_ip",          ip_address=a["ip_address"]),
        "isolate_host":     lambda a: SOCAction(action_type="isolate_host",      hostname=a["hostname"]),
        "revoke_session":   lambda a: SOCAction(action_type="revoke_session",    username=a["username"]),
        "close_alert":      lambda a: SOCAction(
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


async def run_task(base_url: str, task: str, model: str, client: AsyncOpenAI) -> float:
    """Run one task, print step-by-step trace, return final episode score."""
    print(f"\n{'='*62}")
    print(f"  TASK: {task.upper()}")
    print(f"{'='*62}")

    system_prompt = (
        "You are an elite Tier-2 Security Operations Center (SOC) Analyst. "
        "You will be given a security alert. Investigate it by querying logs and threat intelligence. "
        "Take precise, targeted remediation actions (block IPs, isolate hosts, revoke sessions). "
        "Avoid blocking or isolating systems without evidence. "
        "When you are confident the incident is fully resolved, call close_alert with a clear summary. "
        "Be systematic: investigate first, remediate second, close last."
    )

    async with SOCEnvClient(base_url=base_url, task=task) as env:
        obs = await env.reset()
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

        episode_score = 0.0
        done = False
        final = 0.0

        while not done:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print("[WARN] No tool call — ending episode.")
                break

            tc = msg.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"\n[STEP] Agent calls: {name}({json.dumps(args)})")

            action = _build_action(name, args)
            result = await env.step(action)

            done = result.done
            episode_score = result.info.get("episode_score", 0.0)

            print(f"  ← {result.observation.last_action_result}")
            print(f"  ← step_reward={result.reward:.3f}  done={done}  episode_score={episode_score:.3f}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.observation.last_action_result,
            })
            messages.append({
                "role": "user",
                "content": f"Updated environment state:\n{result.observation.model_dump_json(indent=2)}",
            })

        state = await env.state()
        final = state.episode_score
        print(f"\n  📊 Final Episode Score: {final:.3f}")
        return float(final)


async def main(env_url: str):
    api_key = os.environ.get("HF_TOKEN")
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    if not api_key or not api_base_url or not model_name:
        print("ERROR: HF_TOKEN, API_BASE_URL, and MODEL_NAME environment variables must be set.")
        sys.exit(1)

    # Narrow types from Optional[str] to str (guaranteed by the check above)
    assert api_key and api_base_url and model_name

    print(f"\n  Connecting to API: {api_base_url} (Model: {model_name})")
    oai = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
    task_ids = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}

    for task in task_ids:
        scores[task] = await run_task(env_url, task, model_name, oai)

    print(f"\n{'='*62}")
    print(f"  BASELINE RESULTS  (model={model_name})")
    print(f"{'='*62}")
    for t, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  {t:<8}  {s:.3f}  |{bar:<20}|")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'avg':<8}  {avg:.3f}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv-SOC Baseline Inference Script")
    parser.add_argument("--env-url", default="http://localhost:7860", help="URL of the OpenEnv server")
    args = parser.parse_args()
    asyncio.run(main(args.env_url))
