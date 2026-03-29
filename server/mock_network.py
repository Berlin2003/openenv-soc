"""
server/mock_network.py
Stateful simulation of a corporate IT network.
Contains seeded log databases, threat intelligence, and alert queues
for each of the 3 SOC tasks.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Alert:
    alert_id: str
    severity: str
    description: str
    source_ip: Optional[str] = None
    target_host: Optional[str] = None


class MockCorporateNetwork:
    """
    Simulates a minimal corporate network suitable for the 3 SOC tasks.
    All data is deterministic and seed-based — ensures reproducible grading.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.isolated_hosts: List[str] = []
        self.blocked_ips: List[str] = []
        self.revoked_sessions: List[str] = []
        self.alerts: List[Alert] = []
        self._logs: List[Dict] = []
        self._threat_db: Dict[str, str] = {}
        self._closed_alerts: List[str] = []
        self._last_close: Optional[Dict] = None
        self._setup()

    # ------------------------------------------------------------------
    # Scenario Setup  (deterministic, seeded per task)
    # ------------------------------------------------------------------
    def _setup(self):
        if self.task_name == "easy":
            self.alerts = [Alert(
                alert_id="ALT-001",
                severity="Medium",
                description="User reported suspicious login link: login.secure-oauth.com",
                target_host="marketing-laptop-01",
            )]
            self._threat_db = {
                "login.secure-oauth.com": "SAFE — Known internal SSO portal.",
                "104.55.66.77": "MALICIOUS — Known phishing C2 node.",
            }
            self._logs = []

        elif self.task_name == "medium":
            self.alerts = [Alert(
                alert_id="ALT-002",
                severity="High",
                description="Threshold exceeded: 58 failed VPN login attempts from 104.22.33.44 in 10 minutes.",
                source_ip="104.22.33.44",
            )]
            self._threat_db = {
                "104.22.33.44": "MALICIOUS — Confirmed botnet / credential stuffing node (AbuseIPDB score: 97/100).",
            }
            self._logs = [
                {"source": "activedirectory", "msg": "Failed login: admin@corp.com from 104.22.33.44"},
                {"source": "activedirectory", "msg": "Failed login: ceo@corp.com from 104.22.33.44"},
                {"source": "activedirectory", "msg": "Successful login: j.smith@corp.com from 104.22.33.44"},
                {"source": "firewall", "msg": "VPN tunnel established. User: j.smith IP: 104.22.33.44"},
                {"source": "firewall", "msg": "Outbound connection from vpn-gw-01 to 104.22.33.44 port 443"},
            ]

        elif self.task_name == "hard":
            self.alerts = [Alert(
                alert_id="ALT-003",
                severity="Critical",
                description="EDR: Suspicious PowerShell execution on marketing-laptop-02. Script: Invoke-Crypt.ps1",
                target_host="marketing-laptop-02",
            )]
            self._threat_db = {
                "Invoke-Crypt.ps1": "MALICIOUS — Ransomware encryptor/exfil dropper. MITRE T1486.",
                "marketing-laptop-02": "CLEAN — Known corporate asset.",
                "db-server-main": "CLEAN — Primary customer database.",
            }
            self._logs = [
                {"source": "endpoint", "host": "marketing-laptop-02",
                 "msg": "Process spawned: powershell.exe -ExecutionPolicy Bypass -File Invoke-Crypt.ps1"},
                {"source": "firewall",
                 "msg": "marketing-laptop-02 initiated SSH connection to db-server-main port 22"},
                {"source": "endpoint", "host": "db-server-main",
                 "msg": "Successful SSH login from marketing-laptop-02 user: svc-backup"},
                {"source": "endpoint", "host": "db-server-main",
                 "msg": "Large outbound data transfer: 14.7 GB to 185.220.101.52"},
                {"source": "firewall",
                 "msg": "marketing-laptop-02 also connecting to 185.220.101.52 port 443 (Tor exit node)"},
            ]

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------
    def query_logs(self, source: str, query: str) -> str:
        hits = [
            log["msg"] for log in self._logs
            if log.get("source", "") == source and query.lower() in log.get("msg", "").lower()
        ]
        if hits:
            return "\n".join(f"  [{source.upper()}] {h}" for h in hits)
        return f"No {source} logs found matching '{query}'."

    def query_threat_intel(self, indicator: str) -> str:
        for key, verdict in self._threat_db.items():
            if indicator.lower() in key.lower() or key.lower() in indicator.lower():
                return f"ThreatDB lookup for '{indicator}': {verdict}"
        return f"ThreatDB lookup for '{indicator}': UNKNOWN — No records found."

    def block_ip(self, ip: str):
        if ip not in self.blocked_ips:
            self.blocked_ips.append(ip)

    def isolate_host(self, host: str):
        if host not in self.isolated_hosts:
            self.isolated_hosts.append(host)

    def revoke_session(self, username: str):
        if username not in self.revoked_sessions:
            self.revoked_sessions.append(username)

    def close_alert(self, alert_id: str, resolution: str, is_fp: bool):
        self.alerts = [a for a in self.alerts if a.alert_id != alert_id]
        self._closed_alerts.append(alert_id)
        self._last_close = {"resolution": resolution, "is_false_positive": is_fp}
