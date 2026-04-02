---
title: OpenEnv SOC
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🛡️ OpenEnv-SOC: Autonomous Security Operations Center Analyst

<p align="center">
  <a href="https://github.com/meta-pytorch/OpenEnv"><img src="https://img.shields.io/badge/OpenEnv-v2.0-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+" alt="OpenEnv"></a>
  <a href="https://huggingface.co/spaces"><img src="https://img.shields.io/badge/HuggingFace-Space-FFD21E?logo=huggingface" alt="HF Space"></a>
  <a href="#docker"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker" alt="Docker"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Domain-Cybersecurity%20%2F%20SOC-red" alt="Domain">
</p>

---

## Table of Contents

1. [Motivation](#motivation)
2. [What the Agent Does](#what-the-agent-does)
3. [Environment Architecture](#environment-architecture)
4. [Action Space](#action-space)
5. [Observation Space](#observation-space)
6. [Tasks](#tasks)
   - [Task 1 — Phishing False Positive (Easy)](#task-1--phishing-false-positive-easy)
   - [Task 2 — Coordinated Credential Stuffing (Medium)](#task-2--coordinated-credential-stuffing-medium)
   - [Task 3 — Ransomware Lateral Movement 4-hop (Hard)](#task-3--ransomware-lateral-movement-4-hop-hard)
7. [Reward Function](#reward-function)
8. [Grader Specifications](#grader-specifications)
9. [Baseline Scores](#baseline-scores)
10. [Project Structure](#project-structure)
11. [Setup & Usage](#setup--usage)
    - [Local Development](#local-development)
    - [Docker](#docker)
    - [Hugging Face Space](#hugging-face-space)
    - [Python Client API](#python-client-api)
12. [Environment Variables](#environment-variables)
13. [OpenEnv Spec Compliance](#openenv-spec-compliance)

---

## Motivation

Security Operations Centers (SOCs) process **thousands of security alerts daily**. Tier-1 and Tier-2 analysts spend 60–80% of their time on repetitive triage tasks — querying logs, looking up threat intelligence, and deciding whether to block, quarantine, or dismiss. This leads to:

- **Alert fatigue** — analysts miss real threats buried in noise
- **Slow mean-time-to-respond (MTTR)** — attackers have hours or days to move laterally
- **High analyst burnout** — turnover rates in SOC roles exceed 30% annually

**OpenEnv-SOC** is the first OpenEnv environment that trains and evaluates autonomous AI agents on realistic SOC triage workflows. The environment is designed to measure whether agents can:

1. Read ambiguous, noisy alerts (not just pattern-match on keywords)
2. **Actively investigate** by querying timestamped logs across 3 data sources
3. Navigate **red-herring indicators** planted deliberately to penalise agents that act without evidence
4. Apply **precise, targeted remediation** (block/isolate/revoke) without collateral damage
5. Handle **multi-hop attack chains** that span 4 internal hosts, multiple IPs, and compromised service accounts

This environment fills a genuine gap: SOC triage agents trained against OpenEnv-SOC can be evaluated on realistic incident response tasks ranging from trivial false-positive triage to complex ransomware containment.

---

## What the Agent Does

The agent acts as a **Tier-2 SOC Analyst**. Each episode, it:

1. Receives an **initial security alert** with an alert ID, severity level, description, and affected host
2. Queries available data sources to gather evidence:
   - `firewall` logs (inbound/outbound traffic with timestamps)
   - `endpoint` logs (process execution, file activity, login events per host)
   - `activedirectory` logs (authentication successes/failures, VPN events)
   - `threat_intel` database (IP reputation, domain verdicts, malware hashes)
3. Takes remediation actions based on what it finds:
   - `block_ip` — adds an IP to the perimeter firewall blocklist
   - `isolate_host` — quarantines a host from the corporate network
   - `revoke_session` — terminates all active sessions for a user or service account
4. Closes the alert with a summary and a declaration of whether it was a **false positive** or a **confirmed incident**

The environment rewards evidence-based, precise action and penalises both **over-reaction** (blocking safe IPs, isolating unaffected hosts) and **under-investigation** (closing cases without gathering evidence).

---

## Environment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       AI Agent (LLM)                        │
│  Uses: query_logs / query_threat_intel / block_ip /         │
│        isolate_host / revoke_session / close_alert          │
└───────────────────┬─────────────────────────────────────────┘
                    │  HTTP  (JSON / Pydantic)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Server  (server/app.py)                 │
│  POST /reset   POST /step   GET /state   GET /health        │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│         SOCEnvironment  (server/environment.py)             │
│  · OpenEnv-compliant episode management                     │
│  · Shaped step-level reward function                        │
│  · Task-specific graders (_grade_easy / medium / hard)      │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│       MockCorporateNetwork  (server/mock_network.py)        │
│  · Timestamped log database across 3 sources                │
│  · Threat intelligence lookup (ThreatDB)                    │
│  · Stateful: isolated_hosts, blocked_ips, revoked_sessions  │
│  · Red-herring indicators in every task                     │
└─────────────────────────────────────────────────────────────┘
```

**Design principles:**
- **Partial observability** — the agent sees only the initial alert. It must actively pull logs to gather evidence.
- **Timestamped, realistic logs** — all log entries carry ISO 8601 timestamps, hostnames, and structured fields that mirror real SIEM output.
- **Deterministic grading** — scenarios are fully seeded with no randomness. Identical inputs produce identical scores.
- **Shaped reward** — continuous per-step rewards for informative actions, penalties for red-herring actions. Final episode score is the grader output only (always in `[0.0, 1.0]`).
- **Penalty for overreaction** — blocking safe IPs, isolating uninvolved hosts, or revoking clean accounts reduces the score.

---

## Action Space

All actions are expressed as a single `SOCAction` Pydantic model with a required `action_type` discriminator field. Only the parameters relevant to the chosen action type need to be provided.

```python
class SOCAction(Action):
    action_type: Literal[
        "query_logs", "query_threat_intel", "block_ip",
        "isolate_host", "revoke_session", "close_alert"
    ]
    # Parameters (supply only those needed for the chosen action_type):
    source:             Optional[Literal["firewall", "endpoint", "activedirectory"]]
    query:              Optional[str]          # keyword to search in logs
    indicator:          Optional[str]          # IP / domain / filename for threat lookup
    ip_address:         Optional[str]          # IP to block
    hostname:           Optional[str]          # host to isolate
    username:           Optional[str]          # user / service account to revoke
    alert_id:           Optional[str]          # required for close_alert
    resolution_summary: Optional[str]          # human-readable closure note
    is_false_positive:  Optional[bool]         # True = FP, False = confirmed incident
```

| `action_type` | Required params | Description |
|---|---|---|
| `query_logs` | `source`, `query` | Keyword-search timestamped logs in the chosen source. Returns matching log lines with timestamps. |
| `query_threat_intel` | `indicator` | Look up an IP, domain, filename, or username in the threat intelligence database. Returns a verdict and context. |
| `block_ip` | `ip_address` | Add an IP to the perimeter firewall DROP list. Persists for the rest of the episode. |
| `isolate_host` | `hostname` | Quarantine a host from the corporate network. Persists for the rest of the episode. |
| `revoke_session` | `username` | Terminate all active sessions for a user or service account. Persists for the rest of the episode. |
| `close_alert` | `alert_id`, `resolution_summary`, `is_false_positive` | **Terminal action** — concludes the episode and triggers the grader. |

---

## Observation Space

Every `reset()` and `step()` returns a `SOCObservation`:

```python
class Alert(BaseModel):
    alert_id:    str
    severity:    str            # "Low" | "Medium" | "High" | "Critical"
    description: str            # Human-readable incident description
    source_ip:   Optional[str]  # Attacker IP if known at alert time
    target_host: Optional[str]  # Affected internal host

class NetworkStatus(BaseModel):
    isolated_hosts:   List[str]  # Hosts currently quarantined
    blocked_ips:      List[str]  # IPs currently on the firewall blocklist
    revoked_sessions: List[str]  # Users/accounts with sessions revoked

class SOCObservation(Observation):
    open_alerts:        List[Alert]    # Active, unresolved alerts
    network_status:     NetworkStatus  # Current remediation state
    last_action_result: str            # Timestamped output of the last action
    step_count:         int            # Steps taken so far this episode
    reward:             float          # Step-level reward signal
    done:               bool           # True when episode is over
```

**State (read-only, via `GET /state`):**

```python
class SOCState(State):
    task_name:        str
    step_count:       int
    max_steps:        int
    episode_score:    float   # Always in [0.0, 1.0] — grader output only
    done:             bool
    isolated_hosts:   List[str]
    blocked_ips:      List[str]
    revoked_sessions: List[str]
```

---

## Tasks

### Task 1 — Phishing False Positive (`easy`)

**Max steps:** 15 | **Severity:** Medium

**Scenario:**  
Employee `alice.chen@corp.com` reports a suspicious Slack link: `login.secure-oauth.com/sso?redirect=dashboard`. She did not click it. The Helpdesk raised a ticket. The environment contains 12 timestamped log entries showing that:
- 50+ colleagues successfully authenticate through this domain every morning
- The firewall confirms it resolves to Cloudflare CDN node `104.18.22.187`
- ActiveDirectory shows successful SSO auth for multiple users via this domain
- A **red-herring typosquat domain** `login.secure0auth.com` exists in ThreatDB and IS malicious — agents must not confuse it with the real SSO URL

**Expected agent behaviour:**
1. Query any log source or threat intel to gather evidence
2. Identify the domain as the legitimate corporate SSO
3. Close the alert as a **false positive** without taking any remediaton actions

**What trips up naive agents:** Agents that immediately block or isolate without investigating, or agents that confuse `secure-oauth.com` with `secure0auth.com` (the typosquat).

---

### Task 2 — Coordinated Credential Stuffing (`medium`)

**Max steps:** 15 | **Severity:** High

**Scenario:**  
SIEM rule `VPN-BRUTE-001` fires: 147 combined failed VPN logins across **3 attacker IPs** in 8 minutes. Two accounts (`j.smith`, `m.rodriguez`) were successfully compromised (MFA bypassed). `j.smith`'s session was then used to read 820MB from `file-server-02` via SMB — an abnormal data volume.

**Attacker IPs (all malicious):**
- `104.22.33.44` — AbuseIPDB 98/100, confirmed botnet
- `185.156.72.10` — AbuseIPDB 94/100, Storm-1044 threat actor
- `91.108.4.201` — AbuseIPDB 91/100, Tor exit relay

**Red herring:** `203.0.113.5` appears in VPN health-check logs. ThreatDB confirms it is an RFC 5737 documentation IP used by the internal monitoring probe. Blocking it incurs a penalty.

**Expected agent behaviour:**
1. Query ActiveDirectory logs to find the failed + successful logins
2. Query threat intel to confirm all 3 attacker IPs are malicious (and that `203.0.113.5` is safe)
3. Block all 3 malicious IPs
4. Revoke sessions for both compromised users
5. Close as confirmed incident

---

### Task 3 — Ransomware Lateral Movement 4-hop (`hard`)

**Max steps:** 10 | **Severity:** Critical

**Scenario:**  
CrowdStrike Falcon blocked `Invoke-Crypt.ps1` (LockBit 4.0 variant) on `marketing-laptop-02`. Unknown to the initial alert, the ransomware had already moved laterally through the network in a **4-hop attack chain**:

```
marketing-laptop-02
  └─[credential dump: svc-backup, svc-monitor]
    └─► file-server-02     (SMB, 14,822 files renamed to *.locked)
          └─► backup-server   (SSH, backup catalog wiped)
                └─► db-server-main  (PostgreSQL, 2.1M customer records exfiltrated)
```

Two C2 IPs are actively exfiltrating data:
- `185.220.101.52` — active LockBit C2 server (14.7GB outbound)
- `91.92.248.77` — Tor guard relay / secondary C2 channel (2.4GB outbound)

Two service accounts are compromised:
- `svc-backup` — used for SMB and SSH lateral movement
- `svc-monitor` — used to authenticate to the PostgreSQL database

**Red herring:** Nagios monitoring server `10.0.5.22` connects to all hosts for health checks. Its traffic pattern looks similar to lateral movement. It is NOT malicious.

**Expected agent behaviour:**
1. Investigate endpoint logs on `marketing-laptop-02` to find the credential dump
2. Follow the attack chain through firewall + endpoint logs
3. Isolate all 4 hosts: `marketing-laptop-02`, `file-server-02`, `backup-server`, `db-server-main`
4. Block both C2 IPs: `185.220.101.52`, `91.92.248.77`
5. Revoke both service accounts: `svc-backup`, `svc-monitor`
6. Close as confirmed incident — within **8 steps** for the speed bonus

---

## Reward Function

Per-step rewards provide a continuous signal throughout the trajectory. The **final `episode_score`** is set exclusively to the grader output at `close_alert` and is always in `[0.0, 1.0]`.

| Event | Step Reward |
|---|---|
| Log query returns relevant matching entries | `+0.05` |
| Log query returns no matches | `−0.02` |
| Threat intel hit (known indicator in ThreatDB) | `+0.05` |
| Threat intel miss (unknown indicator) | `+0.02` |
| Blocking a known-safe / red-herring IP | `−0.15` |
| Isolating any host during the `easy` task | `−0.15` |
| Revoking any session during the `easy` task | `−0.10` |
| All other remediation actions | `+0.05` |
| `close_alert` (terminal) | `= grader score` |

> **Note:** Step rewards are reported in the `reward` field of each `SOCObservation` for use as trajectory-level training signal. They do **not** accumulate into `episode_score`. This ensures `episode_score` always reflects the grader's verdict on a scale of `0.0–1.0`.

---

## Grader Specifications

All graders are **deterministic** — given the same sequence of actions, the score is always identical. Scores are bounded to `[0.0, 1.0]` via `max(0.0, min(1.0, score))`.

### Easy Grader (`_grade_easy`)

| Condition | Points |
|---|---|
| Declared `is_false_positive = True` | `+0.50` |
| At least 1 investigation step taken before closing | `+0.30` |
| No hosts isolated AND no sessions revoked | `+0.20` |
| Per host incorrectly isolated (overreaction) | `−0.25` |
| Per session incorrectly revoked (overreaction) | `−0.15` |
| **Perfect score** | **1.00** |

### Medium Grader (`_grade_medium`)

| Condition | Points |
|---|---|
| Declared `is_false_positive = False` | `+0.10` |
| Per malicious IP correctly blocked (×3 max) | `+0.20 each` → max `+0.60` |
| Per compromised account session correctly revoked (×2 max) | `+0.15 each` → max `+0.30` |
| Per wrong / red-herring IP blocked | `−0.10` |
| Per innocent user session revoked | `−0.15` |
| **Perfect score** | **1.00** |

### Hard Grader (`_grade_hard`)

| Condition | Points |
|---|---|
| Declared `is_false_positive = False` | `+0.05` |
| Per correct host isolated (×4 max) | `+0.125 each` → max `+0.50` |
| Per C2 IP correctly blocked (×2 max) | `+0.10 each` → max `+0.20` |
| Per service account correctly revoked (×2 max) | `+0.10 each` → max `+0.20` |
| Speed bonus: **all objectives met in ≤8 steps** | `+0.05` |
| Per wrong host isolated / wrong IP blocked / wrong account revoked | `−0.10` |
| **Perfect score (with speed bonus)** | **1.00** |
| **Perfect score (without speed bonus)** | **0.95** |

---

## Baseline Scores

The following scores are **estimated expected performance** for `gpt-4o-mini` based on the grader design and task difficulty. These estimates reflect:
- Easy: agent likely identifies the SSO domain and closes as FP but may skip one investigation step
- Medium: agent often misses one of the 3 attacker IPs or confuses red-herring IPs, and may not trace both compromised users
- Hard: agent rarely traces the full 4-hop chain within the step budget; typically finds 1–2 hosts

> **To produce actual reproduced scores**, run `inference.py` with a valid API key (see [Setup & Usage](#setup--usage)).

| Task | Model | Estimated Score | Grader Max |
|---|---|---|---|
| `easy` | gpt-4o-mini | ~0.80 | 1.00 |
| `medium` | gpt-4o-mini | ~0.55 | 1.00 |
| `hard` | gpt-4o-mini | ~0.25 | 1.00 |
| **Average** | | **~0.53** | |

These estimates serve as a **difficulty calibration reference**. Frontier models (GPT-4o, Claude 3.5 Sonnet) are expected to score significantly higher, especially on medium. The hard task is designed to challenge even frontier models — the 4-hop chain requires systematic log traversal that most LLMs do not naturally perform.

---

## Project Structure

```
soc/
├── inference.py          # Baseline OpenAI agent script (root — required by spec)
├── client.py             # Async + sync HTTP client for the environment
├── models.py             # Pydantic models: SOCAction, SOCObservation, SOCState
├── openenv.yaml          # Environment manifest (OpenEnv spec)
├── requirements.txt      # Runtime dependencies
├── pyproject.toml        # Installable package metadata
├── Dockerfile            # Container definition for HF Spaces
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server: /reset  /step  /state  /health
    ├── environment.py    # Episode management, reward shaping, graders
    └── mock_network.py   # Deterministic corporate network simulation
```

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server (terminal 1)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 3. Verify the server is healthy
curl http://localhost:7860/health
# → {"status": "healthy", "environment": "OpenEnv-SOC", "version": "2.0.0"}

# 4. Run the baseline inference agent (terminal 2)
export HF_TOKEN="your_token_or_openai_key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py --env-url http://localhost:7860
```

**Expected output:**

```
  Connecting to API: https://api.openai.com/v1 (Model: gpt-4o-mini)

══════════════════════════════════════════════════════════════
  TASK: EASY
══════════════════════════════════════════════════════════════

[STEP] Agent calls: query_threat_intel({"indicator": "login.secure-oauth.com"})
  ← ThreatDB [login.secure-oauth.com]: SAFE — Verified internal SSO portal ...
  ← step_reward=0.050  done=False  episode_score=0.000

[STEP] Agent calls: close_alert({"alert_id": "ALT-001", ...})
  ← Alert ALT-001 closed as FALSE POSITIVE. Grader episode score: 0.800

  📊 Final Episode Score: 0.800

══════════════════════════════════════════════════════════════
  BASELINE RESULTS  (model=gpt-4o-mini)
══════════════════════════════════════════════════════════════
  easy      0.800  |████████████████    |
  medium    0.550  |███████████         |
  hard      0.250  |█████               |
  avg       0.533
══════════════════════════════════════════════════════════════
```

### Docker

```bash
# Build the image
docker build -t openenv-soc .

# Run the server
docker run -p 7860:7860 openenv-soc

# Run inference against the containerised server (from host)
export HF_TOKEN="your_token"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py --env-url http://localhost:7860
```

### Hugging Face Space

The environment is deployed as a Hugging Face Space. To use it directly:

```bash
# Point the inference script at the HF Space URL
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py --env-url https://Berlin2003-openenv-soc.hf.space
```

### Python Client API

Use `SOCEnvClient` for programmatic access in training loops:

```python
import asyncio
from client import SOCEnvClient
from models import SOCAction

async def run():
    async with SOCEnvClient("http://localhost:7860", task="hard") as env:
        obs = await env.reset()
        print(f"Alert: {obs.open_alerts[0].description}")
        print(f"Severity: {obs.open_alerts[0].severity}")

        # Step 1: investigate the initial host
        result = await env.step(SOCAction(
            action_type="query_logs",
            source="endpoint",
            query="Invoke-Crypt"
        ))
        print(result.observation.last_action_result)
        print(f"Step reward: {result.reward}")

        # Step 2: check threat intel on the payload
        result = await env.step(SOCAction(
            action_type="query_threat_intel",
            indicator="Invoke-Crypt.ps1"
        ))
        print(result.observation.last_action_result)

        # ... continue investigation and remediation ...

        state = await env.state()
        print(f"Final episode score: {state.episode_score}")

asyncio.run(run())
```

**Synchronous wrapper** (no async required):

```python
from client import SOCEnvClient
from models import SOCAction

with SOCEnvClient("http://localhost:7860", task="easy").sync() as env:
    obs = env.reset()
    result = env.step(SOCAction(
        action_type="query_threat_intel",
        indicator="login.secure-oauth.com"
    ))
    print(result.observation.last_action_result)
```

---

## Environment Variables

| Variable | Required | Description | Example |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | Hugging Face API token (or OpenAI key if using `api.openai.com`) | `hf_...` |
| `API_BASE_URL` | ✅ Yes | LLM API base URL (OpenAI-compatible) | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | ✅ Yes | Model identifier to use for inference | `gpt-4o-mini` |

The inference script will exit with a clear error message if any of these are missing.

---

## OpenEnv Spec Compliance

| Requirement | Implementation |
|---|---|
| `typed Action model` | `SOCAction(Action)` — Pydantic model with `action_type` discriminator |
| `typed Observation model` | `SOCObservation(Observation)` — includes `reward`, `done`, `metadata` |
| `typed State model` | `SOCState(State)` — read-only episode metadata |
| `reset()` endpoint | `POST /reset` — accepts `task` and `episode_id`, returns clean `SOCObservation` |
| `step()` endpoint | `POST /step` — accepts `SOCAction`, returns `{observation, reward, done, info}` |
| `state()` endpoint | `GET /state` — returns current `SOCState` with `episode_score` ∈ `[0.0, 1.0]` |
| `openenv.yaml` | Present in repo root with `name`, `version`, `description`, `entrypoint`, `tags`, `tasks` |
| `health` endpoint | `GET /health` — returns `{"status": "healthy", "environment": "OpenEnv-SOC", "version": "2.0.0"}` |
| 3+ tasks with graders | `easy`, `medium`, `hard` — all graders return scores in `[0.0, 1.0]` |
| Deterministic grading | Fully seeded `MockCorporateNetwork` — no randomness |
| Containerised deployment | `Dockerfile` in repo root |

```bash
# Validate locally (requires openenv-core installed)
openenv validate --config openenv.yaml --url http://localhost:7860
```

---

*OpenEnv-SOC v2.0.0 — Built for the OpenEnv Community Competition*
