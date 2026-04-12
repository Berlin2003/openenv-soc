import asyncio
import httpx
import uuid
from typing import Optional
from pydantic import BaseModel, ConfigDict

from models import (
    SOCObservation, SOCState, SOCAction,
)

class StepResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    observation: SOCObservation
    reward: float
    done: bool
    info: dict = {}

class SOCEnvClient:
    """Async HTTP client wrapping the final OpenEnv SOC FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:7860", task: str = "easy"):
        self.base_url = base_url.rstrip("/")
        self.task = task
        self.episode_id = str(uuid.uuid4())
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def reset(self) -> SOCObservation:
        """POST /reset — returns initial SOCObservation."""
        try:
            resp = await self._client.post("/reset", json={"episode_id": self.episode_id, "task": self.task})
            resp.raise_for_status()
            data = resp.json()
            return SOCObservation.model_validate(data.get("observation", data))
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Failed to reset environment: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error during reset: {str(e)}")
            raise

    async def step(self, action: SOCAction) -> StepResult:
        """POST /step — returns StepResult(observation, reward, done, info)."""
        payload = {
            "episode_id": self.episode_id,
            "action": action.model_dump(exclude_none=True)
        }
        try:
            resp = await self._client.post("/step", json=payload)
            resp.raise_for_status()
            return StepResult.model_validate(resp.json())
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Failed to execute step: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error during step: {str(e)}")
            raise

    async def state(self) -> SOCState:
        """GET /state — returns SOCState episode metadata."""
        try:
            resp = await self._client.get(f"/state?episode_id={self.episode_id}")
            resp.raise_for_status()
            data = resp.json()
            # The openenv framework /state endpoint may return only base State
            # fields. Fill in SOCState defaults so validation doesn't fail.
            data.setdefault("task_name", "unknown")
            data.setdefault("max_steps", 15)
            data.setdefault("episode_score", 0.0)
            data.setdefault("isolated_hosts", [])
            data.setdefault("blocked_ips", [])
            data.setdefault("revoked_sessions", [])
            return SOCState.model_validate(data)
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Failed to fetch state: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error fetching state: {str(e)}")
            raise

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

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._loop.run_until_complete(self._a.__aexit__(exc_type, exc_val, exc_tb))
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
