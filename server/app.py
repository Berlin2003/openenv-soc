from openenv.core import create_fastapi_app
from server.environment import SOCEnvironment
from models import SOCAction, SOCObservation

# The OpenEnv framework handles all JSON schema, routing, and Pydantic validation
app = create_fastapi_app(
    env=SOCEnvironment,
    action_cls=SOCAction,
    observation_cls=SOCObservation,
)
