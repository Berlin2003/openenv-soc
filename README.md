# OpenEnv-SOC: Autonomous Security Operations Center Analyst

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://huggingface.co/spaces)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co)

## Motivation

Security Operations Centers (SOCs) receive thousands of alerts daily. Analysts suffer from **alert fatigue**, repetitive L1/L2 triage, and slow response times that let attackers linger. Training AI agents to autonomously investigate and remediate security incidents is one of the **highest-value enterprise AI problems** today.

OpenEnv-SOC provides a realistic, partial-observability cybersecurity environment where agents must:
1. Read ambiguous alerts
2. **Actively investigate** by pulling logs and querying threat intelligence  
3. Take precise remediation (block/isolate/revoke) without causing false positives
4. Close the incident with an accurate resolution summary

---

## Environment Description

The agent acts as a **Tier-2 SOC Analyst** in a simulated corporate network. It communicates with the environment via HTTP, receiving observations and sending typed action tool calls.

**Design principles:**
- **Partial observability** — the agent does NOT see all logs upfront. It must query them.
- **Partial reward shaping** — every informative action yields a small positive reward. Noise/wrong actions incur penalties. Full episode score is assigned at `CloseAlert`.
- **Deterministic grading** — reproducible across runs using fixed seeded scenarios.

---

## Project Structure

```
soc/
├── models.py             # Action, Observation, State dataclass contracts
├── client.py             # HTTPEnvClient (async + sync) — import this in training
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Installable Python package
├── inference.py          # Baseline OpenAI agent script
└── server/
    ├── mock_network.py   # Mock corporate network simulation
    ├── environment.py    # Core Environment logic + graders
    ├── app.py            # FastAPI server with /reset /step /state /health
    └── Dockerfile        # Container for HF Spaces
```

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `query_logs` | `source`, `query` | Search firewall / endpoint / activedirectory logs |
| `query_threat_intel` | `indicator` | Check IP/domain/hash in threat database |
| `block_ip` | `ip_address` | Add IP to perimeter firewall block list |
| `isolate_host` | `hostname` | Quarantine host from the corporate network |
| `revoke_session` | `username` | Terminate active sessions for a user |
| `close_alert` | `alert_id`, `resolution_summary`, `is_false_positive` | **Terminal action** — triggers grading |

## Observation Space

```python
SOCObservation(
    open_alerts: List[Alert],         # Active security incidents
    network_status: NetworkStatus,    # isolated_hosts, blocked_ips, revoked_sessions
    last_action_result: str,          # Human-readable result of last action
    step_count: int,
    done: bool,
    reward: float,                    # Step-level reward signal
)
```

---

## Tasks

| ID | Name | Difficulty | Max Steps | Target Score |
|----|------|-----------|-----------|-------------|
| `easy` | Phishing False Positive | ⭐ Easy | 15 | 1.0 |
| `medium` | Credential Stuffing Attack | ⭐⭐ Medium | 15 | 1.0 |
| `hard` | Ransomware Lateral Movement | ⭐⭐⭐ Hard | 10 | 1.0 |

### Task 1: Phishing False Positive (Easy)
An employee reports a suspicious login link. The agent must query threat intelligence and discover it is the company's own SSO portal — then close the alert as a false positive without disrupting any hosts.

**Grader:** `1.0` for closing as FP with no host isolations. Penalises overreaction (`-0.5` per unnecessary isolation).

### Task 2: Credential Stuffing Attack (Medium)
An alert fires for 58 failed VPN logins from an external IP. The agent must query ActiveDirectory logs, discover one successful login for `j.smith`, confirm the source IP is a known botnet node, block the IP, revoke `j.smith`'s session, and close the alert.

**Grader:** `+0.5` per correct remediation action. `-0.25` per wrong IP blocked.

### Task 3: Ransomware Lateral Movement (Hard)
A critical EDR alert fires for suspicious PowerShell execution on `marketing-laptop-02`. The agent must trace the attack chain through firewall logs, discover the ransomware has moved laterally to `db-server-main`, and isolate both hosts — all within 6 steps for full score.

**Grader:** `1.0` (both isolated, ≤6 steps) | `0.75` (both, ≤8 steps) | `0.5` (both, >8 steps) | `0.3` (one host) | `0.0` (none).

---

## Reward Function

| Event | Reward |
|-------|--------|
| Query returns relevant results | `+0.05` |
| Query returns no results | `-0.02` |
| Invalid action type | `-0.05` |
| Episode end (grader) | `0.0–1.0` |

---

## Baseline Scores (gpt-4o-mini)

| Task | Score |
|------|-------|
| easy | ~0.85 |
| medium | ~0.65 |
| hard | ~0.40 |
| **avg** | **~0.63** |

---

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline agent (in a separate terminal)
export OPENAI_API_KEY="sk-..."
python inference.py --base-url http://localhost:7860
```

### Docker

```bash
docker build -t openenv-soc .
docker run -p 7860:7860 openenv-soc
```

### Use the client in your training code

```python
import asyncio
from client import SOCEnvClient
from models import QueryThreatIntel, CloseAlert

async def main():
    async with SOCEnvClient("http://localhost:7860", task="easy") as env:
        obs = await env.reset()
        print(obs.open_alerts)

        obs = await env.step(QueryThreatIntel(indicator="login.secure-oauth.com"))
        print(obs.last_action_result)

asyncio.run(main())
```

### Health check

```bash
curl http://localhost:7860/health
# {"status": "healthy", "environment": "OpenEnv-SOC", "version": "1.0.0"}
```
