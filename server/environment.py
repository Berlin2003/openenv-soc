from typing import Optional
from openenv.core import Environment
from models import SOCAction, SOCObservation, SOCState, Alert, NetworkStatus
from server.mock_network import MockCorporateNetwork

class SOCEnvironment(Environment[SOCAction, SOCObservation, SOCState]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._network = None
        self._task_name = "easy"
        self._step_count = 0
        self._max_steps = 15
        self._episode_score = 0.0
        self._done = False
        self._initial_alert = None

    def _generate_alert(self, task: str) -> Alert:
        if task == "easy":
            return Alert(
                alert_id="ALT-001", severity="Medium",
                description="User reported suspicious login link: login.secure-oauth.com",
                target_host="marketing-laptop-01"
            )
        elif task == "medium":
            return Alert(
                alert_id="ALT-002", severity="High",
                description="Multiple failed logins followed by successful login for j.smith from external IP",
                target_host="vpn-gateway"
            )
        elif task == "hard":
            return Alert(
                alert_id="ALT-003", severity="Critical",
                description="EDR blocked 'Invoke-Crypt' payload execution",
                target_host="marketing-laptop-02"
            )
        return Alert(alert_id="ALT-000", severity="Low", description="Unknown task")

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task: str = "easy", **kwargs) -> SOCObservation:
        self._task_name = task
        self._step_count = 0
        self._episode_score = 0.0
        self._done = False
        self._network = MockCorporateNetwork(task_name=task)
        self._initial_alert = self._generate_alert(task)
        
        obs = SOCObservation(
            open_alerts=[self._initial_alert],
            network_status=self._network.get_status(),
            last_action_result=f"[SOC Terminal] Incident queue loaded for task='{task}'. Max steps: {self._max_steps}. Investigate and remediate.",
            step_count=self._step_count,
            reward=0.0,
            done=False,
            metadata={"episode_score": self._episode_score}
        )
        # Required by OpenEnv core:
        obs.done = False
        obs.reward = 0.0
        return obs

    def step(self, action: SOCAction, timeout_s: Optional[float] = None, **kwargs) -> SOCObservation:
        if self._done:
            return SOCObservation(
                open_alerts=[], network_status=self._network.get_status(),
                last_action_result="Episode is already done. Please reset().",
                step_count=self._step_count,
                done=True, reward=0.0, metadata={"episode_score": self._episode_score}
            )
        
        self._step_count += 1
        reward = 0.0
        result_msg = ""
        is_terminal = False

        if action.action_type == "query_logs":
            source_str = action.source.value if hasattr(action.source, 'value') else action.source
            res = self._network.query_logs(source_str, action.query)
            if res.startswith("No "):
                reward = -0.02
            else:
                reward = 0.05
            result_msg = res
        
        elif action.action_type == "query_threat_intel":
            res = self._network.query_threat_intel(action.indicator)
            reward = 0.05
            result_msg = f"ThreatDB lookup for '{action.indicator}': {res}"
            
        elif action.action_type == "block_ip":
            self._network.block_ip(action.ip_address)
            reward = 0.05
            result_msg = f"Firewall rule added: DROP IP {action.ip_address}"
            
        elif action.action_type == "isolate_host":
            self._network.isolate_host(action.hostname)
            reward = -0.1 if self._task_name == "easy" else 0.05
            result_msg = f"Host {action.hostname} isolated from network."
            
        elif action.action_type == "revoke_session":
            self._network.revoke_session(action.username)
            reward = 0.05
            result_msg = f"All active sessions revoked for {action.username}."
            
        elif action.action_type == "close_alert":
            self._done = True
            is_terminal = True
            reward = self._grade_task(action.is_false_positive)
            result_msg = f"Alert {action.alert_id} closed."
            
        self._episode_score = round(self._episode_score + reward, 4)
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            is_terminal = True
            result_msg += f" \\nMax steps ({self._max_steps}) reached."
            
        obs = SOCObservation(
            open_alerts=[self._initial_alert] if not self._done else [],
            network_status=self._network.get_status(),
            last_action_result=result_msg,
            step_count=self._step_count,
            reward=reward,
            done=self._done,
            metadata={"episode_score": self._episode_score}
        )
        return obs

    def _grade_task(self, declared_fp: bool) -> float:
        status = self._network.get_status()
        if self._task_name == "easy":
            expected_fp = True
            hosts = status.isolated_hosts
            score = 1.0 if declared_fp else 0.0
            if hosts: score -= 0.5
            return max(0.0, score)
        
        elif self._task_name == "medium":
            expected_fp = False
            score = 0.0
            if not declared_fp: score += 0.2
            if "104.22.33.44" in status.blocked_ips: score += 0.4
            if "j.smith" in status.revoked_sessions: score += 0.4
            return max(0.0, score)
            
        elif self._task_name == "hard":
            expected_fp = False
            score = 0.0
            if not declared_fp: score += 0.2
            if "marketing-laptop-02" in status.isolated_hosts: score += 0.4
            if "db-server-main" in status.isolated_hosts:
                score += 0.4 if self._step_count <= 6 else 0.1
            return max(0.0, score)
            
        return 0.0

    @property
    def state(self) -> SOCState:
        status = self._network.get_status() if self._network else NetworkStatus()
        return SOCState(
            task_name=self._task_name,
            step_count=self._step_count,
            max_steps=self._max_steps,
            episode_score=self._episode_score,
            done=self._done,
            isolated_hosts=status.isolated_hosts,
            blocked_ips=status.blocked_ips,
            revoked_sessions=status.revoked_sessions
        )
