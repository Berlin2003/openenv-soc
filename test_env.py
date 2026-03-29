from server.environment import SOCEnvironment
from models import SOCAction

env = SOCEnvironment()
obs = env.reset(task="easy")
act = SOCAction(action_type="query_threat_intel", indicator="login.secure-oauth.com")
obs2 = env.step(act)
print("SUCCESS. obs2 reward:", getattr(obs2, 'reward', None))
