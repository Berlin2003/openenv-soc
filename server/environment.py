"""
server/environment.py
Core SOC Environment — server-side logic.
Uses Pydantic models from models.py.
step() API: returns StepResult(observation, reward, done, info)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    SOCObservation, SOCReward, SOCState, StepResult,
    Alert, NetworkStatus,
    QueryLogs, QueryThreatIntel, BlockIPAddress,
    IsolateHost, RevokeUserSession, CloseAlert, SOCAction,
)
from server.mock_network import MockCorporateNetwork

MAX_STEPS = {"easy": 15, "medium": 15, "hard": 10}


class SOCEnvironment:
    """
    OpenEnv-SOC core environment.
    Implements reset() / step() / state() per the competition spec.
    step() returns StepResult(observation, reward, done, info).
    """

    def __init__(self):
        self._task: str = "easy"
        self._net: MockCorporateNetwork | None = None
        self._step_count: int = 0
        self._done: bool = False
        self._episode_score: float = 0.0
        self._last_result: str = ""

    # ------------------------------------------------------------------
    # reset() → SOCObservation
    # ------------------------------------------------------------------
    def reset(self, task: str = "easy") -> SOCObservation:
        """Start a fresh episode for the given task. Returns initial observation."""
        self._task = task
        self._net = MockCorporateNetwork(task)
        self._step_count = 0
        self._done = False
        self._episode_score = 0.0
        self._last_result = (
            f"[SOC Terminal] Incident queue loaded for task='{task}'. "
            f"Max steps: {MAX_STEPS.get(task, 15)}. Investigate and remediate."
        )
        return self._make_obs()

    # ------------------------------------------------------------------
    # step(action) → StepResult(observation, reward, done, info)
    # ------------------------------------------------------------------
    def step(self, action: SOCAction) -> StepResult:
        """Execute one action. Returns (observation, reward, done, info)."""
        if self._done:
            return StepResult(
                observation=self._make_obs(),
                reward=SOCReward(value=0.0, reason="Episode already finished."),
                done=True,
                info={"warning": "Episode already done."},
            )

        self._step_count += 1
        step_reward = 0.0
        reward_reason = ""

        # ---- Action dispatch -----------------------------------------
        if isinstance(action, QueryLogs):
            result = self._net.query_logs(action.source, action.query)
            self._last_result = result
            if "No " not in result:
                step_reward = 0.05
                reward_reason = "Relevant logs found. Partial progress reward."
            else:
                step_reward = -0.02
                reward_reason = "Query returned no results. Small penalty for wasted step."

        elif isinstance(action, QueryThreatIntel):
            result = self._net.query_threat_intel(action.indicator)
            self._last_result = result
            if "UNKNOWN" not in result:
                step_reward = 0.05
                reward_reason = "Threat intel hit. Partial progress reward."
            else:
                step_reward = 0.01
                reward_reason = "No threat intel match."

        elif isinstance(action, BlockIPAddress):
            self._net.block_ip(action.ip_address)
            self._last_result = f"Firewall updated: {action.ip_address} BLOCKED."
            reward_reason = "IP blocked — episode score determined at close."

        elif isinstance(action, IsolateHost):
            self._net.isolate_host(action.hostname)
            self._last_result = f"Host '{action.hostname}' isolated from network."
            reward_reason = "Host isolated — episode score determined at close."

        elif isinstance(action, RevokeUserSession):
            self._net.revoke_session(action.username)
            self._last_result = f"All active sessions for '{action.username}' revoked."
            reward_reason = "Session revoked — episode score determined at close."

        elif isinstance(action, CloseAlert):
            self._net.close_alert(
                action.alert_id, action.resolution_summary, action.is_false_positive
            )
            self._done = True
            self._last_result = (
                f"Alert {action.alert_id} closed. Resolution: {action.resolution_summary}"
            )

        else:
            self._last_result = "Unknown action type. No effect."
            step_reward = -0.05
            reward_reason = "Penalty for invalid action."

        # ---- Max steps termination -----------------------------------
        max_s = MAX_STEPS.get(self._task, 15)
        if self._step_count >= max_s and not self._done:
            self._done = True
            self._last_result += " | MAX STEPS REACHED — episode terminated."

        # ---- Final grading at episode end ----------------------------
        if self._done:
            step_reward = self._grade()
            reward_reason = f"Final episode grade for task='{self._task}'."

        self._episode_score += step_reward
        obs = self._make_obs()
        reward = SOCReward(value=round(max(0.0, min(1.0, step_reward)), 4), reason=reward_reason)

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"episode_score": round(self._episode_score, 4), "step": self._step_count},
        )

    # ------------------------------------------------------------------
    # state → SOCState
    # ------------------------------------------------------------------
    @property
    def state(self) -> SOCState:
        """Return episode-level metadata."""
        return SOCState(
            task_name=self._task,
            step_count=self._step_count,
            max_steps=MAX_STEPS.get(self._task, 15),
            episode_score=round(self._episode_score, 4),
            done=self._done,
            isolated_hosts=list(self._net.isolated_hosts) if self._net else [],
            blocked_ips=list(self._net.blocked_ips) if self._net else [],
            revoked_sessions=list(self._net.revoked_sessions) if self._net else [],
        )

    # ------------------------------------------------------------------
    # Graders — deterministic, 0.0–1.0
    # ------------------------------------------------------------------
    def _grade(self) -> float:
        n = self._net
        if self._task == "easy":
            return self._grade_easy(n)
        elif self._task == "medium":
            return self._grade_medium(n)
        elif self._task == "hard":
            return self._grade_hard(n)
        return 0.0

    @staticmethod
    def _grade_easy(n: MockCorporateNetwork) -> float:
        """
        Phishing False Positive.
        Full score for closing as FP with no host isolations.
        Penalise unnecessary host isolation.
        """
        lc = n._last_close
        if lc is None:
            return 0.0
        score = 1.0 if lc["is_false_positive"] else 0.0
        score -= 0.5 * len(n.isolated_hosts)   # over-reaction penalty
        return max(0.0, min(1.0, score))

    @staticmethod
    def _grade_medium(n: MockCorporateNetwork) -> float:
        """
        Credential Stuffing.
        +0.5 block attacker IP, +0.5 revoke compromised user session.
        """
        score = 0.0
        if "104.22.33.44" in n.blocked_ips:
            score += 0.5
        if "j.smith" in n.revoked_sessions or "j.smith@corp.com" in n.revoked_sessions:
            score += 0.5
        wrong_blocks = [ip for ip in n.blocked_ips if ip != "104.22.33.44"]
        score -= 0.25 * len(wrong_blocks)
        return max(0.0, min(1.0, score))

    def _grade_hard(self, n: MockCorporateNetwork) -> float:
        """
        Ransomware Lateral Movement.
        Both hosts must be isolated. Speed bonus for fast containment.
        """
        laptop = "marketing-laptop-02" in n.isolated_hosts
        db_srv = "db-server-main" in n.isolated_hosts

        if laptop and db_srv:
            if self._step_count <= 6:
                return 1.0
            elif self._step_count <= 8:
                return 0.75
            return 0.5      # contained but probably too late
        elif laptop or db_srv:
            return 0.3      # partial containment
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_obs(self) -> SOCObservation:
        alerts = [
            Alert(
                alert_id=a.alert_id, severity=a.severity,
                description=a.description, source_ip=a.source_ip,
                target_host=a.target_host,
            )
            for a in (self._net.alerts if self._net else [])
        ]
        ns = NetworkStatus(
            isolated_hosts=list(self._net.isolated_hosts) if self._net else [],
            blocked_ips=list(self._net.blocked_ips) if self._net else [],
            revoked_sessions=list(self._net.revoked_sessions) if self._net else [],
        )
        return SOCObservation(
            open_alerts=alerts,
            network_status=ns,
            last_action_result=self._last_result,
            step_count=self._step_count,
        )
