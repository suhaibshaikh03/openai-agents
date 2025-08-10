from agents import Agent, Runner, handoff, RunContextWrapper
from dotenv import load_dotenv , find_dotenv    
import os
import asyncio
from pydantic import BaseModel

_:bool = load_dotenv(find_dotenv())

billing_agent = Agent(name = "Billing agent", instructions="You are a billing agent")
refund_agent = Agent(name = "Refund Agent", instructions="You are a refund agent")

class MyInfo(BaseModel):
    name:str
    age:int


def refund(ctx:RunContextWrapper[MyInfo]):
    print(f"Hi {ctx.context.name} handed off to refund agent")


general_agent = Agent(name = "general Agent", instructions="You are a general agent",
handoffs=[billing_agent, 
handoff(agent = refund_agent, tool_name_override = "external_refund",
        tool_description_override = "Use this tool to refund an order",
        is_enabled = True,
        on_handoff=refund)]
)

async def main():
    result = await Runner.run(
        general_agent,
        "I need to refund my order. Details : order id : 804, reason : wrong item",
        context=MyInfo(name="Suhaib", age=21)
    )
    print(result.final_output)
    print(result.last_agent.name)

if __name__ == "__main__":
    asyncio.run(main())
