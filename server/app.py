"""
server/app.py
FastAPI server for OpenEnv-SOC.
POST /reset  → SOCObservation
POST /step   → StepResult (observation, reward, done, info)
GET  /state  → SOCState
GET  /health → {"status": "healthy"}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Annotated, Union
from fastapi import FastAPI, HTTPException
from pydantic import Field

from models import (
    SOCObservation, SOCState, StepResult,
    QueryLogs, QueryThreatIntel, BlockIPAddress,
    IsolateHost, RevokeUserSession, CloseAlert,
)
from server.environment import SOCEnvironment

app = FastAPI(
    title="OpenEnv-SOC",
    description=(
        "Autonomous Security Operations Center Analyst. "
        "A real-world environment for training AI agents to triage and remediate cybersecurity incidents."
    ),
    version="1.0.0",
)

env = SOCEnvironment()


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------
from pydantic import BaseModel

class ResetRequest(BaseModel):
    task: str = Field("easy", description="Task name: 'easy', 'medium', or 'hard'")


# Discriminated union so FastAPI dispatches on action_type
ActionRequest = Annotated[
    Union[QueryLogs, QueryThreatIntel, BlockIPAddress, IsolateHost, RevokeUserSession, CloseAlert],
    Field(discriminator="action_type"),
]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health", tags=["meta"])
async def health():
    """Health check — returns status of the running environment."""
    return {"status": "healthy", "environment": "OpenEnv-SOC", "version": "1.0.0"}


@app.get("/", tags=["meta"])
async def root():
    return {
        "name": "OpenEnv-SOC",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
    }


@app.post("/reset", response_model=SOCObservation, tags=["env"])
async def reset(req: ResetRequest) -> SOCObservation:
    """
    Start a new episode.
    Returns the initial SOCObservation.
    """
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be 'easy', 'medium', or 'hard'")
    return env.reset(task=req.task)


@app.post("/step", response_model=StepResult, tags=["env"])
async def step(action: ActionRequest) -> StepResult:
    """
    Execute one action.
    Returns StepResult(observation, reward, done, info).
    """
    return env.step(action)


@app.get("/state", response_model=SOCState, tags=["env"])
async def state() -> SOCState:
    """Return current episode metadata (step_count, score, done, network_status)."""
    return env.state
