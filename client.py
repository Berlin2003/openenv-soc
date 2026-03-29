"""
client.py
Async + sync HTTP client for OpenEnv-SOC.
Implements the standard OpenEnv HTTPEnvClient pattern.

Usage (async):
    async with SOCEnvClient("http://localhost:7860", task="medium") as env:
        obs = await env.reset()
        result = await env.step(BlockIPAddress(ip_address="104.22.33.44"))
        print(result.reward.value, result.done)

Usage (sync):
    with SOCEnvClient("http://localhost:7860", task="easy").sync() as env:
        obs = env.reset()
        result = env.step(CloseAlert(alert_id="ALT-001", ...))
"""
import asyncio
import httpx

from models import (
    SOCObservation, SOCState, StepResult, SOCAction,
)


class SOCEnvClient:
    """Async HTTP client wrapping the SOC FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:7860", task: str = "easy"):
        self.base_url = base_url.rstrip("/")
        self.task = task
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self) -> SOCObservation:
        """POST /reset — returns initial SOCObservation."""
        resp = await self._client.post("/reset", json={"task": self.task})
        resp.raise_for_status()
        return SOCObservation.model_validate(resp.json())

    async def step(self, action: SOCAction) -> StepResult:
        """POST /step — returns StepResult(observation, reward, done, info)."""
        payload = action.model_dump()
        resp = await self._client.post("/step", json=payload)
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    async def state(self) -> SOCState:
        """GET /state — returns SOCState episode metadata."""
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return SOCState.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Sync convenience wrapper
    # ------------------------------------------------------------------
    class _SyncWrapper:
        def __init__(self, async_client: "SOCEnvClient"):
            self._a = async_client
            self._loop = asyncio.new_event_loop()

        def __enter__(self):
            self._loop.run_until_complete(self._a.__aenter__())
            return self

        def __exit__(self, *args):
            self._loop.run_until_complete(self._a.__aexit__(*args))
            self._loop.close()

        def reset(self) -> SOCObservation:
            return self._loop.run_until_complete(self._a.reset())

        def step(self, action: SOCAction) -> StepResult:
            return self._loop.run_until_complete(self._a.step(action))

        def state(self) -> SOCState:
            return self._loop.run_until_complete(self._a.state())

    def sync(self) -> "_SyncWrapper":
        """Return a synchronous wrapper for use without async."""
        return self._SyncWrapper(self)
