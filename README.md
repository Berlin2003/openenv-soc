# OpenEnv-SOC v2: Autonomous Security Operations Center Analyst

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://huggingface.co/spaces)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co)

## Motivation

Security Operations Centers (SOCs) receive thousands of alerts daily. Analysts suffer from **alert fatigue**, repetitive triage, and slow response times. Training AI agents to autonomously investigate and remediate security incidents is one of the **highest-value enterprise AI problems** today.

OpenEnv-SOC provides a realistic, partial-observability cybersecurity environment where agents must:
1. Read ambiguous, noisy alerts
2. **Actively investigate** by querying timestamped logs across 3 sources and running threat intelligence lookups
3. Navigate **red-herring indicators** (safe IPs that look suspicious, typosquat domains, benign monitoring traffic)
4. Take **precise, targeted remediation** without causing collateral damage
5. Manage multi-hop attack chains involving multiple hosts, IPs, and compromised accounts

---

## Environment Description

The agent acts as a **Tier-2 SOC Analyst** in a simulated corporate network. It communicates with the environment via HTTP, receiving observations and sending typed action calls.

**Design principles:**
- **Partial observability** — the agent sees only the initial alert. It must actively query logs.
- **Timestamped, realistic logs** — all log entries carry ISO 8601 timestamps and host context.
- **Red herrings** — each task contains benign-looking malicious indicators AND malicious-looking benign ones.
- **Shaped reward** — every informative action yields a small positive reward. Wrong/noisy actions incur penalties. Full episode score assigned at `close_alert`.
- **Penalty for overreaction** — blocking safe IPs, isolating unaffected hosts, or revoking clean accounts reduces the score.
- **Deterministic grading** — reproducible across runs using fixed seeded scenarios.

---

## Project Structure

```
soc/
├── models.py             # Action, Observation, State Pydantic models
├── client.py             # Async + sync HTTP client for the environment
├── openenv.yaml          # Environment manifest (OpenEnv spec)
├── pyproject.toml        # Installable Python package
├── inference.py          # Baseline OpenAI agent script
└── server/
    ├── mock_network.py   # Stateful corporate network simulation (logs, threat DB)
    ├── environment.py    # Core environment logic + graders
    └── app.py            # FastAPI server exposing /reset /step /state /health
```

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `query_logs` | `source` (`firewall`\|`endpoint`\|`activedirectory`), `query` | Search timestamped logs by keyword |
| `query_threat_intel` | `indicator` | Look up IP, domain, filename, or username |
| `block_ip` | `ip_address` | Add IP to perimeter firewall block list |
| `isolate_host` | `hostname` | Quarantine a host from the corporate network |
| `revoke_session` | `username` | Terminate all active sessions for a user or service account |
| `close_alert` | `alert_id`, `resolution_summary`, `is_false_positive` | **Terminal** — triggers grading |

## Observation Space

```python
SOCObservation(
    open_alerts: List[Alert],         # Active security incidents
    network_status: NetworkStatus,    # isolated_hosts, blocked_ips, revoked_sessions
    last_action_result: str,          # Human-readable result of last action (with timestamps)
    step_count: int,                  # Steps taken so far
    done: bool,
    reward: float,                    # Step-level reward signal
)
```

---

## Tasks

| ID | Name | Difficulty | Max Steps | Perfect Score |
|----|------|-----------|-----------|---------------|
| `easy` | Phishing False Positive | ⭐ Easy | 15 | 1.0 |
| `medium` | Coordinated Credential Stuffing | ⭐⭐⭐ Medium-Hard | 15 | 1.0 |
| `hard` | Ransomware Lateral Movement (4-hop) | ⭐⭐⭐⭐⭐ Very Hard | 10 | 1.0 |

### Task 1: Phishing False Positive (`easy`)

An employee flags a suspicious login link `login.secure-oauth.com`. The agent must:
- Query firewall, endpoint, and/or ActiveDirectory logs
- Check threat intelligence (and avoid being fooled by the typosquat `login.secure0auth.com`)
- Conclude the link is the corporate SSO portal
- Close as **false positive without isolating any hosts or revoking any sessions**

**Grader:**

| Condition | Points |
|-----------|--------|
| Declared false positive | +0.50 |
| ≥1 investigation step before closing | +0.30 |
| No hosts isolated, no sessions revoked | +0.20 |
| Per isolated host (overreaction) | −0.25 |
| Per revoked session (overreaction) | −0.15 |

---

### Task 2: Coordinated Credential Stuffing (`medium`)

A 3-IP botnet (`104.22.33.44`, `185.156.72.10`, `91.108.4.201`) attack compromises two VPN accounts: `j.smith` and `m.rodriguez`. `j.smith`'s session is then used to read 820MB from `file-server-02`. A red-herring IP (`203.0.113.5`) appears in logs but is a safe internal health-check probe.

Agent must: block all 3 malicious IPs, revoke both compromised sessions. Do NOT block the safe IP.

**Grader:**

| Condition | Points |
|-----------|--------|
| Declared not false positive | +0.10 |
| Per correct malicious IP blocked (×3 max) | +0.15 each |
| Per correct compromised user revoked (×2 max) | +0.20 each |
| Per wrong IP blocked | −0.10 |
| Per innocent user revoked | −0.15 |

---

### Task 3: Ransomware Lateral Movement — 4-hop (`hard`)

LockBit ransomware on `marketing-laptop-02` pivots through:
1. `marketing-laptop-02` → (credential dump) → steals `svc-backup` + `svc-monitor` creds
2. → `file-server-02` (SMB, encrypts 14,822 files)
3. → `backup-server` (SSH, wipes backup catalog)
4. → `db-server-main` (PostgreSQL, exfiltrates 2.1M customer records)

Two C2 IPs exfiltrate data: `185.220.101.52` and `91.92.248.77`. Red-herring Nagios monitoring traffic (`10.0.5.22`) connects to all hosts normally.

Agent must isolate all 4 hosts, block both C2 IPs, revoke both service accounts.

**Grader:**

| Condition | Points |
|-----------|--------|
| Declared not false positive | +0.05 |
| Per correct host isolated (×4 max) | +0.10 each |
| Per C2 IP blocked (×2 max) | +0.10 each |
| Per service account revoked (×2 max) | +0.10 each |
| Speed bonus: all objectives met in ≤8 steps | +0.05 |
| Per wrong host / IP / account | −0.10 each |

---

## Reward Function (Step-Level)

| Event | Reward |
|-------|--------|
| Query returns relevant results | `+0.05` |
| Threat intel hit (known indicator) | `+0.05` |
| Threat intel miss (unknown indicator) | `+0.02` |
| Query returns no results | `−0.02` |
| Blocking a known-safe (red-herring) IP | `−0.15` |
| Isolating a host in easy task | `−0.15` |
| Revoking a session in easy task | `−0.10` |
| Episode end (grader score) | `0.0 – 1.0` |

---

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| easy | gpt-4o-mini | ~0.80 |
| medium | gpt-4o-mini | ~0.55 |
| hard | gpt-4o-mini | ~0.25 |
| **avg** | | **~0.53** |

---

## Setup & Usage

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In a separate terminal:
export HF_TOKEN="your_token"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py --env-url http://localhost:7860
```

### Docker

```bash
docker build -t openenv-soc .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  openenv-soc
```

### Python Client

```python
import asyncio
from client import SOCEnvClient
from models import SOCAction

async def main():
    async with SOCEnvClient("http://localhost:7860", task="hard") as env:
        obs = await env.reset()
        print(obs.open_alerts[0].description)

        # Investigate the attack chain
        result = await env.step(SOCAction(
            action_type="query_logs",
            source="endpoint",
            query="Invoke-Crypt"
        ))
        print(result.observation.last_action_result)

asyncio.run(main())
```

### Health Check

```bash
curl http://localhost:7860/health
# {"status": "healthy", "environment": "OpenEnv-SOC", "version": "2.0.0"}
```
