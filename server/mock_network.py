"""
server/mock_network.py
======================
Rich, deterministic simulation of a corporate IT network.

Each task contains:
  - Timestamped log entries across 3 sources (firewall / endpoint / activedirectory)
  - A threat-intelligence database with SAFE, MALICIOUS, and UNKNOWN indicators
  - Red-herring entries designed to confuse agents that don't investigate carefully
  - Multi-hop attack chains (medium: 3 IPs, 2 users; hard: 4 hosts, 2 C2 IPs, 2 svc accounts)

All data is fully deterministic — guarantees reproducible grading across runs.
"""

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
    Stateful simulation of a corporate IT environment for the 3 SOC tasks.

    Task 1 — easy   : Phishing False Positive
      An employee flags 'login.secure-oauth.com'.  Logs show it's used by
      50+ colleagues daily.  A red-herring suspicious IP appears in firewall
      logs.  Agent must investigate BEFORE closing, or face a penalty.

    Task 2 — medium : Coordinated Credential Stuffing
      A three-IP botnet hits VPN.  Two accounts (j.smith, m.rodriguez) are
      compromised.  j.smith then moves laterally to 'file-server-02'.
      Agent must block ALL 3 IPs and revoke BOTH sessions.  Blocking the
      wrong IPs or revoking innocent users is penalised.

    Task 3 — hard   : Ransomware Lateral Movement (4-hop chain)
      Ransomware on 'marketing-laptop-02' pivots to 'file-server-02', then
      'backup-server', then 'db-server-main'.  Two C2 IPs are exfiltrating
      data.  Two service accounts (svc-backup, svc-monitor) are compromised.
      Agent must isolate all 4 hosts, block both C2s, revoke both accounts —
      all within 10 steps for a perfect score.  Red-herring benign traffic
      is mixed into every log source.
    """

    # -----------------------------------------------------------------------
    # Public state (mutated by action handlers)
    # -----------------------------------------------------------------------
    isolated_hosts: List[str]
    blocked_ips: List[str]
    revoked_sessions: List[str]

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.isolated_hosts: List[str] = []
        self.blocked_ips: List[str] = []
        self.revoked_sessions: List[str] = []
        self.alerts: List[Alert] = []
        self._logs: List[Dict] = []
        self._threat_db: Dict[str, str] = {}
        self._setup()

    def get_status(self) -> Dict:
        return {
            "isolated_hosts": self.isolated_hosts,
            "blocked_ips": self.blocked_ips,
            "revoked_sessions": self.revoked_sessions,
        }

    # -----------------------------------------------------------------------
    # Scenario Initialisation
    # -----------------------------------------------------------------------
    def _setup(self):
        if self.task_name == "easy":
            self._setup_easy()
        elif self.task_name == "medium":
            self._setup_medium()
        elif self.task_name == "hard":
            self._setup_hard()

    # ------------------------------------------------------------------
    # TASK 1 — EASY : Phishing False Positive
    # ------------------------------------------------------------------
    def _setup_easy(self):
        self.alerts = [Alert(
            alert_id="ALT-001",
            severity="Medium",
            description=(
                "User alice.chen@corp.com flagged a suspicious login link received via Slack: "
                "'login.secure-oauth.com/sso?redirect=dashboard'. "
                "Alice did not click it. Ticket raised by Helpdesk."
            ),
            target_host="marketing-laptop-01",
        )]

        self._threat_db = {
            # The reported URL — actually the corporate SSO
            "login.secure-oauth.com": (
                "SAFE — Verified internal SSO portal operated by IT (cert CN=secure-oauth.com, "
                "owner: Corp IT Dept, last cert audit 2026-02-10). No threat matches."
            ),
            # Red herring: a REAL phishing domain that looks similar
            "login.secure0auth.com": (
                "MALICIOUS — Typosquat domain. Hosts credential-harvesting kit. "
                "Reported by PhishTank #9981234. Do NOT confuse with secure-oauth.com."
            ),
            # Firewall red herring — CDN that looks suspicious
            "104.18.22.187": (
                "SAFE — Cloudflare CDN egress node. Serves login.secure-oauth.com. "
                "Not a threat indicator."
            ),
            # Another benign internal asset
            "10.0.1.50": "SAFE — Internal jump host. Managed by IT Ops.",
        }

        self._logs = [
            # ---- Firewall (shows normal corporate SSO traffic) ----
            {"source": "firewall", "ts": "2026-03-30T06:01:12Z",
             "msg": "ALLOW outbound 10.0.2.14 -> 104.18.22.187:443 (login.secure-oauth.com) bytes=2840"},
            {"source": "firewall", "ts": "2026-03-30T06:03:44Z",
             "msg": "ALLOW outbound 10.0.2.55 -> 104.18.22.187:443 (login.secure-oauth.com) bytes=3100"},
            {"source": "firewall", "ts": "2026-03-30T06:10:01Z",
             "msg": "ALLOW outbound 10.0.2.78 -> 104.18.22.187:443 (login.secure-oauth.com) bytes=2900"},
            {"source": "firewall", "ts": "2026-03-30T06:22:33Z",
             "msg": "ALLOW outbound marketing-laptop-01 -> 104.18.22.187:443 (login.secure-oauth.com) bytes=3040"},
            # Red herring: outbound to an unrelated IP that LOOKS suspicious
            {"source": "firewall", "ts": "2026-03-30T06:45:00Z",
             "msg": "DENY inbound 185.220.100.99:22 -> dmz-server-01 (SSH brute-force block rule #44)"},
            # ---- Endpoint (shows SSO agent on marketing-laptop-01) ----
            {"source": "endpoint", "host": "marketing-laptop-01", "ts": "2026-03-30T06:22:30Z",
             "msg": "Process: msedge.exe opened URL https://login.secure-oauth.com/sso?redirect=dashboard"},
            {"source": "endpoint", "host": "marketing-laptop-01", "ts": "2026-03-30T06:22:35Z",
             "msg": "SSO agent returned auth token for alice.chen@corp.com — login successful"},
            # ---- ActiveDirectory (shows broad org use of SSO) ----
            {"source": "activedirectory", "ts": "2026-03-30T06:23:00Z",
             "msg": "SSO auth success: alice.chen@corp.com via login.secure-oauth.com"},
            {"source": "activedirectory", "ts": "2026-03-30T06:24:10Z",
             "msg": "SSO auth success: b.jones@corp.com via login.secure-oauth.com"},
            {"source": "activedirectory", "ts": "2026-03-30T06:25:02Z",
             "msg": "SSO auth success: t.wang@corp.com via login.secure-oauth.com"},
            {"source": "activedirectory", "ts": "2026-03-30T06:26:15Z",
             "msg": "SSO auth success: r.patel@corp.com via login.secure-oauth.com"},
        ]

    # ------------------------------------------------------------------
    # TASK 2 — MEDIUM : Coordinated Credential Stuffing
    # ------------------------------------------------------------------
    def _setup_medium(self):
        self.alerts = [Alert(
            alert_id="ALT-002",
            severity="High",
            description=(
                "SIEM Rule 'VPN-BRUTE-001' fired: 147 combined failed VPN login attempts "
                "across 3 external IPs in 8 minutes. Two successful logins detected. "
                "Possible coordinated credential stuffing campaign."
            ),
            source_ip="multiple",
            target_host="vpn-gateway-01",
        )]

        self._threat_db = {
            # Attacker IPs — all malicious
            "104.22.33.44": (
                "MALICIOUS — AbuseIPDB score 98/100. Confirmed botnet node used in "
                "credential stuffing campaigns. AS: AS13335 (Cloudflare abuse network)."
            ),
            "185.156.72.10": (
                "MALICIOUS — AbuseIPDB score 94/100. Known proxy exit node used by "
                "Storm-1044 threat actor. Linked to 12 corp breaches in Q1 2026."
            ),
            "91.108.4.201": (
                "MALICIOUS — AbuseIPDB score 91/100. Tor exit relay also used for "
                "password spray attacks. Block immediately."
            ),
            # Red herring: a 4th IP that looks suspicious but is the company's own VPN probe
            "203.0.113.5": (
                "SAFE — Reserved documentation IP (RFC 5737). Appears in internal "
                "VPN health-check logs — not a real attacker."
            ),
            # Compromised users (for agent awareness — NOT a direct threat intel hit)
            "j.smith@corp.com": "INFO — Account. Check recent AD events.",
            "m.rodriguez@corp.com": "INFO — Account. Check recent AD events.",
        }

        self._logs = [
            # ---- ActiveDirectory — three IPs, two successful logins ----
            {"source": "activedirectory", "ts": "2026-03-30T08:00:01Z",
             "msg": "Failed login: admin@corp.com from 104.22.33.44 (VPN)"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:04Z",
             "msg": "Failed login: ceo@corp.com from 104.22.33.44 (VPN)"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:10Z",
             "msg": "Failed login: it.support@corp.com from 185.156.72.10 (VPN)"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:15Z",
             "msg": "Failed login: hr.manager@corp.com from 91.108.4.201 (VPN)"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:22Z",
             "msg": "Failed login: j.smith@corp.com from 104.22.33.44 (VPN)"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:28Z",
             "msg": "Failed login: m.rodriguez@corp.com from 185.156.72.10 (VPN)"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:35Z",
             "msg": "Successful login: j.smith@corp.com from 104.22.33.44 (VPN) — MFA bypassed"},
            {"source": "activedirectory", "ts": "2026-03-30T08:00:49Z",
             "msg": "Successful login: m.rodriguez@corp.com from 185.156.72.10 (VPN) — MFA bypassed"},
            # Red herring: legitimate IT admin login that timestamp-overlaps
            {"source": "activedirectory", "ts": "2026-03-30T08:01:00Z",
             "msg": "Successful login: sysadmin@corp.com from 10.0.1.50 (Internal) — expected"},
            # ---- Firewall — attacker traffic ----
            {"source": "firewall", "ts": "2026-03-30T08:00:36Z",
             "msg": "VPN tunnel established: user=j.smith src=104.22.33.44 assigned_ip=172.16.10.55"},
            {"source": "firewall", "ts": "2026-03-30T08:00:51Z",
             "msg": "VPN tunnel established: user=m.rodriguez src=185.156.72.10 assigned_ip=172.16.10.56"},
            # Health-check red herring
            {"source": "firewall", "ts": "2026-03-30T08:00:00Z",
             "msg": "VPN health probe from 203.0.113.5 — OK (internal monitor)"},
            # j.smith lateral movement after VPN entry
            {"source": "firewall", "ts": "2026-03-30T08:02:11Z",
             "msg": "ALLOW 172.16.10.55 (j.smith VPN) -> file-server-02:445 (SMB)"},
            {"source": "firewall", "ts": "2026-03-30T08:02:44Z",
             "msg": "ALLOW 172.16.10.55 (j.smith VPN) -> file-server-02:445 SMB large read 820MB"},
            # ---- Endpoint — file-server-02 shows j.smith access ----
            {"source": "endpoint", "host": "file-server-02", "ts": "2026-03-30T08:02:15Z",
             "msg": "SMB session opened: user=j.smith\\corp ip=172.16.10.55"},
            {"source": "endpoint", "host": "file-server-02", "ts": "2026-03-30T08:02:50Z",
             "msg": "Bulk read: 820MB of /projects/q1-roadmap/ by j.smith — ABNORMAL volume"},
        ]

    # ------------------------------------------------------------------
    # TASK 3 — HARD : Ransomware Lateral Movement (4-hop chain)
    # ------------------------------------------------------------------
    def _setup_hard(self):
        self.alerts = [Alert(
            alert_id="ALT-003",
            severity="Critical",
            description=(
                "EDR (CrowdStrike Falcon) detected and BLOCKED 'Invoke-Crypt.ps1' on "
                "'marketing-laptop-02'. Process tree: explorer.exe -> powershell.exe "
                "-ExecutionPolicy Bypass -File Invoke-Crypt.ps1. Agent requests SOC triage."
            ),
            target_host="marketing-laptop-02",
        )]

        self._threat_db = {
            # Ransomware payload
            "Invoke-Crypt.ps1": (
                "MALICIOUS — Ransomware encryptor + exfiltration dropper. MITRE ATT&CK: "
                "T1486 (Data Encrypted for Impact), T1041 (Exfil over C2). "
                "Family: LockBit 4.0 variant. First seen 2026-02-01."
            ),
            # Both C2 IPs
            "185.220.101.52": (
                "MALICIOUS — Active LockBit C2 server. IPs seen in 34 ransomware incidents "
                "this month. Exfiltration endpoint. Block at perimeter immediately."
            ),
            "91.92.248.77": (
                "MALICIOUS — Secondary C2 / Tor guard relay used by LockBit 4.0 for "
                "redundant command channel. AbuseIPDB: 99/100."
            ),
            # Red herring: svc-backup looks suspicious but is a known account
            "svc-backup": (
                "INFO — Service account for scheduled backup jobs. Compromised in this "
                "incident — verify lateral movement to backup-server."
            ),
            "svc-monitor": (
                "INFO — Service account for Nagios monitoring. Credentials leaked via "
                "Invoke-Crypt credential harvester. Revoke immediately."
            ),
            # Benign red herring
            "10.0.5.22": (
                "SAFE — Internal Nagios monitoring server. Normally connects to all hosts "
                "for health checks. High traffic volume is expected."
            ),
        }

        self._logs = [
            # === HOP 1: marketing-laptop-02 — initial infection ===
            {"source": "endpoint", "host": "marketing-laptop-02", "ts": "2026-03-30T03:12:01Z",
             "msg": "Process spawned: powershell.exe -ExecutionPolicy Bypass -File Invoke-Crypt.ps1 (blocked by EDR)"},
            {"source": "endpoint", "host": "marketing-laptop-02", "ts": "2026-03-30T03:12:05Z",
             "msg": "Credential dump attempt: LSASS memory read by powershell.exe PID 4488"},
            {"source": "endpoint", "host": "marketing-laptop-02", "ts": "2026-03-30T03:12:10Z",
             "msg": "Credentials extracted: svc-backup\\corp (plaintext), svc-monitor\\corp (NTLM hash)"},
            # === HOP 2: marketing-laptop-02 -> file-server-02 ===
            {"source": "firewall", "ts": "2026-03-30T03:12:30Z",
             "msg": "ALLOW marketing-laptop-02 -> file-server-02:445 SMB (user: svc-backup)"},
            {"source": "endpoint", "host": "file-server-02", "ts": "2026-03-30T03:12:35Z",
             "msg": "SMB logon: svc-backup@corp from marketing-laptop-02 [UNUSUAL — svc-backup normally runs on backup-server only]"},
            {"source": "endpoint", "host": "file-server-02", "ts": "2026-03-30T03:12:40Z",
             "msg": "Mass file rename detected: 14,822 files renamed to *.locked in /shared/finance/"},
            # === HOP 3: file-server-02 -> backup-server ===
            {"source": "firewall", "ts": "2026-03-30T03:13:10Z",
             "msg": "ALLOW file-server-02 -> backup-server:22 SSH (user: svc-backup)"},
            {"source": "endpoint", "host": "backup-server", "ts": "2026-03-30T03:13:15Z",
             "msg": "SSH session: svc-backup@corp from file-server-02 — authenticated via stolen key"},
            {"source": "endpoint", "host": "backup-server", "ts": "2026-03-30T03:13:20Z",
             "msg": "backup catalog wiped: /var/backups/veeam/* deleted by svc-backup"},
            # === HOP 4: backup-server -> db-server-main ===
            {"source": "firewall", "ts": "2026-03-30T03:13:50Z",
             "msg": "ALLOW backup-server -> db-server-main:5432 PostgreSQL (user: svc-monitor)"},
            {"source": "endpoint", "host": "db-server-main", "ts": "2026-03-30T03:13:55Z",
             "msg": "DB login: svc-monitor via pg_hba.conf trust — [ABNORMAL: svc-monitor has no DB role]"},
            {"source": "endpoint", "host": "db-server-main", "ts": "2026-03-30T03:14:00Z",
             "msg": "COPY TO: SELECT * FROM customers -> /tmp/export.csv — 2.1M rows exfiltrated"},
            # === C2 Exfiltration — both IPs ===
            {"source": "firewall", "ts": "2026-03-30T03:14:10Z",
             "msg": "ALLOW marketing-laptop-02 -> 185.220.101.52:443 HTTPS — 14.7GB outbound (C2 exfil)"},
            {"source": "firewall", "ts": "2026-03-30T03:14:22Z",
             "msg": "ALLOW db-server-main -> 91.92.248.77:9001 — 2.4GB outbound (Tor/C2 secondary channel)"},
            # === Red herrings — benign Nagios traffic ===
            {"source": "firewall", "ts": "2026-03-30T03:12:00Z",
             "msg": "ALLOW 10.0.5.22 (nagios) -> marketing-laptop-02:22 SSH health-check — OK"},
            {"source": "firewall", "ts": "2026-03-30T03:12:00Z",
             "msg": "ALLOW 10.0.5.22 (nagios) -> db-server-main:5432 health-check — OK"},
            {"source": "endpoint", "host": "marketing-laptop-02", "ts": "2026-03-30T03:11:58Z",
             "msg": "Scheduled task: Windows Defender quick scan — completed OK (pre-infection)"},
        ]

    # -----------------------------------------------------------------------
    # Action Handlers
    # -----------------------------------------------------------------------
    def query_logs(self, source: str, query: str) -> str:
        hits = [
            log["msg"] for log in self._logs
            if log.get("source", "") == source
            and query.lower() in log.get("msg", "").lower()
        ]
        timestamps = [
            log.get("ts", "")
            for log in self._logs
            if log.get("source", "") == source
            and query.lower() in log.get("msg", "").lower()
        ]
        if hits:
            lines = [f"  [{source.upper()}] {ts} — {msg}" for ts, msg in zip(timestamps, hits)]
            return "\n".join(lines)
        return f"No {source} logs matching '{query}'."

    def query_threat_intel(self, indicator: str) -> str:
        for key, verdict in self._threat_db.items():
            if indicator.lower() in key.lower() or key.lower() in indicator.lower():
                return f"ThreatDB [{indicator}]: {verdict}"
        return f"ThreatDB [{indicator}]: UNKNOWN — no records. Treat as suspicious until confirmed."

    def block_ip(self, ip: str):
        if ip not in self.blocked_ips:
            self.blocked_ips.append(ip)

    def isolate_host(self, host: str):
        if host not in self.isolated_hosts:
            self.isolated_hosts.append(host)

    def revoke_session(self, username: str):
        if username not in self.revoked_sessions:
            self.revoked_sessions.append(username)
