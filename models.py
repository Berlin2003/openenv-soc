"""
OpenEnv-SOC models.py
Typed Action, Observation, Reward, and State Pydantic models.
Competition spec: step(action) -> (observation, reward, done, info)
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Action Models  (Pydantic — typed + serializable to/from JSON)
# ---------------------------------------------------------------------------

class QueryLogs(BaseModel):
    """Search corporate log sources for security events."""
    action_type: Literal["query_logs"] = "query_logs"
    source: Literal["firewall", "endpoint", "activedirectory"] = Field(
        ..., description="Log source to query"
    )
    query: str = Field(..., description="Keyword to search for in log messages")


class QueryThreatIntel(BaseModel):
    """Look up an IP, domain, or file hash in threat intelligence."""
    action_type: Literal["query_threat_intel"] = "query_threat_intel"
    indicator: str = Field(..., description="IP, domain, or filename to look up")


class BlockIPAddress(BaseModel):
    """Add an IP address to the perimeter firewall block list."""
    action_type: Literal["block_ip"] = "block_ip"
    ip_address: str = Field(..., description="IP address to block")


class IsolateHost(BaseModel):
    """Quarantine a host from the corporate network."""
    action_type: Literal["isolate_host"] = "isolate_host"
    hostname: str = Field(..., description="Hostname to isolate")


class RevokeUserSession(BaseModel):
    """Immediately terminate all active sessions for a compromised user."""
    action_type: Literal["revoke_session"] = "revoke_session"
    username: str = Field(..., description="Username whose sessions to revoke")


class CloseAlert(BaseModel):
    """
    Terminal action — marks the incident as resolved and ends the episode.
    Triggers final grading.
    """
    action_type: Literal["close_alert"] = "close_alert"
    alert_id: str = Field(..., description="ID of the alert to close")
    resolution_summary: str = Field(..., description="Summary of findings and actions taken")
    is_false_positive: bool = Field(False, description="Whether this is a false positive")


# Discriminated union for the /step endpoint
SOCAction = Union[
    QueryLogs, QueryThreatIntel, BlockIPAddress,
    IsolateHost, RevokeUserSession, CloseAlert
]


# ---------------------------------------------------------------------------
# Sub-models used in Observation
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    alert_id: str
    severity: str              # "Low" | "Medium" | "High" | "Critical"
    description: str
    source_ip: Optional[str] = None
    target_host: Optional[str] = None


class NetworkStatus(BaseModel):
    isolated_hosts: List[str] = Field(default_factory=list)
    blocked_ips: List[str] = Field(default_factory=list)
    revoked_sessions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation  — returned by reset() and step()
# ---------------------------------------------------------------------------

class SOCObservation(BaseModel):
    """What the agent sees after each step."""
    open_alerts: List[Alert] = Field(description="Currently open security alerts")
    network_status: NetworkStatus
    last_action_result: str = Field(description="Human-readable result of the last action")
    step_count: int


# ---------------------------------------------------------------------------
# Reward  — separate Pydantic model per competition spec
# ---------------------------------------------------------------------------

class SOCReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward signal for this step (0.0–1.0)")
    reason: str = Field(description="Human-readable explanation of the reward")


# ---------------------------------------------------------------------------
# StepResult  — full return type of step(): (obs, reward, done, info)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result of a step() call as required by competition spec."""
    observation: SOCObservation
    reward: SOCReward
    done: bool
    info: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State  — returned by GET /state (episode-level metadata)
# ---------------------------------------------------------------------------

class SOCState(BaseModel):
    task_name: str
    step_count: int
    max_steps: int
    episode_score: float = Field(description="Cumulative score so far")
    done: bool
    isolated_hosts: List[str]
    blocked_ips: List[str]
    revoked_sessions: List[str]
