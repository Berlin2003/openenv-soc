from openenv.core import create_fastapi_app
from server.environment import SOCEnvironment
from models import SOCAction, SOCObservation

# Enforce singleton pattern to maintain state across stateless HTTP calls
global_env = SOCEnvironment()

# The OpenEnv framework handles all JSON schema, routing, and Pydantic validation
app = create_fastapi_app(
    env=lambda: global_env,
    action_cls=SOCAction,
    observation_cls=SOCObservation,
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
