import asyncio
from client import SOCEnvClient
from models import SOCAction

async def test():
    async with SOCEnvClient("http://localhost:7860", task="easy") as env:
        obs = await env.reset()
        print("RESET SUCCESS! Step count:", obs.step_count)
        
        act = SOCAction(action_type="query_threat_intel", indicator="login.secure-oauth.com")
        res = await env.step(act)
        print("STEP SUCCESS! Reward:", res.reward, "Done:", res.done)
        print("Message:", res.observation.last_action_result)

asyncio.run(test())
