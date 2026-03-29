from openenv.core import Action, Observation, State
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Alert(BaseModel):
    alert_id: str
    severity: str
    description: str
    source_ip: Optional[str] = None
    target_host: Optional[str] = None

class NetworkStatus(BaseModel):
    isolated_hosts: List[str] = Field(default_factory=list)
    blocked_ips: List[str] = Field(default_factory=list)
    revoked_sessions: List[str] = Field(default_factory=list)

class SOCAction(Action):
    """Unified Action class representing all possible SOC operations."""
    action_type: Literal[
        "query_logs", "query_threat_intel", "block_ip", 
        "isolate_host", "revoke_session", "close_alert"
    ]
    
    source: Optional[Literal["firewall", "endpoint", "activedirectory"]] = None
    query: Optional[str] = None
    indicator: Optional[str] = None
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    username: Optional[str] = None
    alert_id: Optional[str] = None
    resolution_summary: Optional[str] = None
    is_false_positive: Optional[bool] = False

class SOCObservation(Observation):
    open_alerts: List[Alert]
    network_status: NetworkStatus
    last_action_result: str
    step_count: int

class SOCState(State):
    task_name: str
    max_steps: int
    episode_score: float
    isolated_hosts: List[str]
    blocked_ips: List[str]
    revoked_sessions: List[str]
