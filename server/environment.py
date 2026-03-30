"""
server/environment.py
=====================
OpenEnv-compliant SOC environment.

Inherits from openenv.core.Environment and implements:
  reset()  — initialises episode state
  step()   — processes a SOCAction, returns shaped reward + observation
  state()  — returns read-only episode metadata

Grader logic (called on close_alert):
  easy   : requires ≥1 investigation step; penalises overreaction
  medium : multi-IP / multi-user scoring; penalises wrong blocks/revokes
  hard   : 4-host + 2-C2-IP + 2-svc-account scoring; speed bonus included
"""

from typing import Optional
from openenv.core import Environment
from models import SOCAction, SOCObservation, SOCState, Alert, NetworkStatus
from server.mock_network import MockCorporateNetwork


class SOCEnvironment(Environment[SOCAction, SOCObservation, SOCState]):

    # -----------------------------------------------------------------------
    # Task configuration
    # -----------------------------------------------------------------------
    _MAX_STEPS = {"easy": 15, "medium": 15, "hard": 10}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._network: Optional[MockCorporateNetwork] = None
        self._task_name: str = "easy"
        self._step_count: int = 0
        self._max_steps: int = 15
        self._episode_score: float = 0.0
        self._done: bool = False
        self._initial_alert: Optional[Alert] = None
        # Track investigative actions taken before close_alert (for easy grader)
        self._investigation_steps: int = 0

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs,
    ) -> SOCObservation:
        self._task_name = task
        self._step_count = 0
        self._max_steps = self._MAX_STEPS.get(task, 15)
        self._episode_score = 0.0
        self._done = False
        self._investigation_steps = 0
        self._network = MockCorporateNetwork(task_name=task)
        self._initial_alert = self._network.alerts[0]

        return SOCObservation(
            open_alerts=[self._initial_alert],
            network_status=NetworkStatus(**self._network.get_status()),
            last_action_result=(
                f"[SOC Terminal] Incident queue loaded | task='{task}' "
                f"| max_steps={self._max_steps} | Investigate and remediate."
            ),
            step_count=self._step_count,
            reward=0.0,
            done=False,
            metadata={"episode_score": self._episode_score},
        )

    def step(self, action: SOCAction, timeout_s: Optional[float] = None, **kwargs) -> SOCObservation:
        if self._done:
            return SOCObservation(
                open_alerts=[],
                network_status=NetworkStatus(**self._network.get_status()),
                last_action_result="Episode already done. Call reset().",
                step_count=self._step_count,
                done=True,
                reward=0.0,
                metadata={"episode_score": self._episode_score},
            )

        self._step_count += 1
        reward = 0.0
        result_msg = ""

        # ---- Dispatch action -----------------------------------------------
        if action.action_type == "query_logs":
            source_str = action.source.value if hasattr(action.source, "value") else action.source
            res = self._network.query_logs(source_str, action.query or "")
            if res.startswith("No "):
                reward = -0.02
                result_msg = res
            else:
                reward = 0.05
                self._investigation_steps += 1
                result_msg = res

        elif action.action_type == "query_threat_intel":
            res = self._network.query_threat_intel(action.indicator or "")
            if "UNKNOWN" in res:
                reward = 0.02
            else:
                reward = 0.05
                self._investigation_steps += 1
            result_msg = res

        elif action.action_type == "block_ip":
            ip = action.ip_address or ""
            self._network.block_ip(ip)
            # Penalise blocking the known-safe red-herring IPs
            red_herrings = {"203.0.113.5", "10.0.5.22", "10.0.1.50", "104.18.22.187"}
            if ip in red_herrings:
                reward = -0.15
                result_msg = f"[WARNING] Firewall rule added for {ip} — but this is a known-safe internal/CDN node. False positive action."
            else:
                reward = 0.05
                result_msg = f"Firewall rule added: DROP {ip} at perimeter."

        elif action.action_type == "isolate_host":
            host = action.hostname or ""
            self._network.isolate_host(host)
            # Penalise isolating clean hosts in easy task
            if self._task_name == "easy":
                reward = -0.15
                result_msg = f"[WARNING] {host} isolated — no evidence of compromise in this case."
            else:
                reward = 0.05
                result_msg = f"Host {host} quarantined from corporate network."

        elif action.action_type == "revoke_session":
            username = action.username or ""
            self._network.revoke_session(username)
            # In easy task, revoking any session is overreaction
            if self._task_name == "easy":
                reward = -0.1
                result_msg = f"[WARNING] Sessions revoked for {username} — no evidence of account compromise in this task."
            else:
                reward = 0.05
                result_msg = f"All active sessions for {username} terminated."

        elif action.action_type == "close_alert":
            self._done = True
            grade = self._grade_task(action.is_false_positive or False)
            reward = grade
            self._episode_score = round(self._episode_score + reward, 4)
            result_msg = (
                f"Alert {action.alert_id} closed as "
                f"{'FALSE POSITIVE' if action.is_false_positive else 'CONFIRMED INCIDENT'}. "
                f"Episode score: {self._episode_score:.3f}"
            )
            obs = SOCObservation(
                open_alerts=[],
                network_status=NetworkStatus(**self._network.get_status()),
                last_action_result=result_msg,
                step_count=self._step_count,
                reward=reward,
                done=True,
                metadata={"episode_score": self._episode_score},
            )
            return obs

        # Accumulate cumulative episode score (non-terminal steps)
        self._episode_score = round(self._episode_score + reward, 4)

        # Step-limit check
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            result_msg += f" | Max steps ({self._max_steps}) reached — episode forced closed."

        return SOCObservation(
            open_alerts=[self._initial_alert] if not self._done else [],
            network_status=NetworkStatus(**self._network.get_status()),
            last_action_result=result_msg,
            step_count=self._step_count,
            reward=reward,
            done=self._done,
            metadata={"episode_score": self._episode_score},
        )

    def state(self) -> SOCState:
        status = self._network.get_status() if self._network else {}
        return SOCState(
            task_name=self._task_name,
            step_count=self._step_count,
            max_steps=self._max_steps,
            episode_score=self._episode_score,
            done=self._done,
            isolated_hosts=status.get("isolated_hosts", []),
            blocked_ips=status.get("blocked_ips", []),
            revoked_sessions=status.get("revoked_sessions", []),
        )

    # -----------------------------------------------------------------------
    # Graders  (called only at close_alert)
    # -----------------------------------------------------------------------
    def _grade_task(self, declared_fp: bool) -> float:
        if self._task_name == "easy":
            return self._grade_easy(declared_fp)
        elif self._task_name == "medium":
            return self._grade_medium(declared_fp)
        elif self._task_name == "hard":
            return self._grade_hard(declared_fp)
        return 0.0

    # ------------------------------------------------------------------
    # EASY GRADER
    # Goal: close as false-positive WITHOUT isolating/revoking anything.
    # Requirement: agent must investigate at least once before closing.
    #
    # Scoring:
    #   +0.50  declared as false positive
    #   +0.30  ≥1 investigation step taken (query_logs or query_threat_intel)
    #   +0.20  no host isolations and no sessions revoked
    #   -0.25  per isolated host (overreaction penalty)
    #   -0.15  per revoked session (overreaction penalty)
    # Max: 1.0    Min clipped at 0.0
    # ------------------------------------------------------------------
    def _grade_easy(self, declared_fp: bool) -> float:
        status = self._network.get_status()
        score = 0.0

        if declared_fp:
            score += 0.50
        if self._investigation_steps >= 1:
            score += 0.30
        if not status["isolated_hosts"] and not status["revoked_sessions"]:
            score += 0.20

        # Overreaction penalties
        score -= 0.25 * len(status["isolated_hosts"])
        score -= 0.15 * len(status["revoked_sessions"])

        return round(max(0.0, score), 4)

    # ------------------------------------------------------------------
    # MEDIUM GRADER
    # Goal: identify all 3 attacker IPs, revoke both compromised sessions.
    # Do NOT block the red-herring IP (203.0.113.5) or revoke clean accounts.
    #
    # Scoring (per correct action):
    #   +0.10  declared NOT false positive
    #   +0.15  per malicious IP blocked  (104.22.33.44, 185.156.72.10, 91.108.4.201)
    #   +0.20  per compromised user session revoked  (j.smith, m.rodriguez)
    #   -0.10  per wrong IP blocked (any IP not in the malicious set)
    #   -0.15  per wrong user session revoked (any user not compromised)
    # Max: 0.10 + 3*0.15 + 2*0.20 = 0.10+0.45+0.40 = 0.95  (≈1.0 with rounding)
    # Full 1.0 achievable if agent also gets the 0.05 step-reward bonus in trajectory.
    # ------------------------------------------------------------------
    def _grade_medium(self, declared_fp: bool) -> float:
        status = self._network.get_status()
        MALICIOUS_IPS = {"104.22.33.44", "185.156.72.10", "91.108.4.201"}
        RED_HERRING_IPS = {"203.0.113.5"}
        COMPROMISED_USERS = {"j.smith", "m.rodriguez"}

        score = 0.0
        if not declared_fp:
            score += 0.10

        for ip in status["blocked_ips"]:
            if ip in MALICIOUS_IPS:
                score += 0.15
            elif ip in RED_HERRING_IPS:
                score -= 0.10  # blocked the safe IP
            else:
                score -= 0.10  # blocked an unknown/unrelated IP

        for user in status["revoked_sessions"]:
            # Normalise — agent might pass full email or just username
            uname = user.split("@")[0].lower()
            if uname in COMPROMISED_USERS or user.lower() in COMPROMISED_USERS:
                score += 0.20
            else:
                score -= 0.15  # revoked an innocent user

        return round(max(0.0, score), 4)

    # ------------------------------------------------------------------
    # HARD GRADER
    # Goal: trace 4-hop attack chain, isolate ALL 4 hosts, block BOTH C2 IPs,
    #       revoke BOTH service accounts — within 10 steps.
    #
    # Scoring:
    #   +0.05  declared NOT false positive
    #   +0.10  per host isolated:
    #            marketing-laptop-02, file-server-02, backup-server, db-server-main
    #   +0.10  per C2 IP blocked:  185.220.101.52, 91.92.248.77
    #   +0.10  per svc account revoked:  svc-backup, svc-monitor
    #   +0.05  speed bonus if ALL objectives met in ≤8 steps
    #   -0.10  per wrong host isolated (collateral damage)
    #   -0.10  per wrong IP blocked
    #   -0.10  per wrong user revoked
    # Max: 0.05 + 4*0.10 + 2*0.10 + 2*0.10 + 0.05 = 0.05+0.40+0.20+0.20+0.05 = 0.90
    # (trajectory step rewards push total to 1.0 for a clean run)
    # ------------------------------------------------------------------
    def _grade_hard(self, declared_fp: bool) -> float:
        status = self._network.get_status()
        REQUIRED_HOSTS = {"marketing-laptop-02", "file-server-02", "backup-server", "db-server-main"}
        C2_IPS = {"185.220.101.52", "91.92.248.77"}
        SVC_ACCOUNTS = {"svc-backup", "svc-monitor"}

        score = 0.0
        if not declared_fp:
            score += 0.05

        # Hosts
        for host in status["isolated_hosts"]:
            if host in REQUIRED_HOSTS:
                score += 0.10
            else:
                score -= 0.10  # wrong host isolated

        # C2 IPs
        for ip in status["blocked_ips"]:
            if ip in C2_IPS:
                score += 0.10
            else:
                score -= 0.10  # wrong IP blocked

        # Service accounts
        for user in status["revoked_sessions"]:
            uname = user.split("@")[0].lower() if "@" in user else user.lower()
            if uname in SVC_ACCOUNTS or user.lower() in SVC_ACCOUNTS:
                score += 0.10
            else:
                score -= 0.10  # wrong account revoked

        # Speed bonus: all objectives achieved in ≤8 steps
        all_hosts_isolated = REQUIRED_HOSTS.issubset(set(status["isolated_hosts"]))
        all_c2_blocked = C2_IPS.issubset(set(status["blocked_ips"]))
        all_accounts_revoked = SVC_ACCOUNTS.issubset(
            {u.split("@")[0].lower() for u in status["revoked_sessions"]}
            | {u.lower() for u in status["revoked_sessions"]}
        )
        if all_hosts_isolated and all_c2_blocked and all_accounts_revoked:
            if self._step_count <= 8:
                score += 0.05  # speed bonus

        return round(max(0.0, score), 4)
