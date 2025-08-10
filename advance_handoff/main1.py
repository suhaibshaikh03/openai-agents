from agents import Agent, Runner, handoff, RunContextWrapper, function_tool
from agents.extensions import handoff_filters
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

def can_refund(ctx:RunContextWrapper[MyInfo],agent:Agent[MyInfo])->bool:
    print("context",ctx.context)
    print("agent",agent.name)
    if ctx.context.name == "Suhaib":
        return True
    return False

def refund(ctx:RunContextWrapper[MyInfo]):
    print(f"Hi {ctx.context.name} handed off to refund agent")

@function_tool
def get_weather(city:str)->str:
    return f"weather in {city} is 20 degrees"


general_agent = Agent(name = "general Agent",
                     instructions="You are a general agent",
                     tools=[get_weather],
                     model="gpt-5",


                    handoffs=[billing_agent, 
handoff(agent = refund_agent, tool_name_override = "external_refund",
        tool_description_override = "Use this tool to refund an order",
        on_handoff=refund,
        input_filter=handoff_filters.remove_all_tools,
        is_enabled=can_refund)]
)

async def main():
    result = await Runner.run(
        general_agent,
        "What is the weather in karachi and secondly I need to refund my order. Details : order id : 804, reason : wrong item",
        context=MyInfo(name="Suhaib", age=21)
    )
    print(result.final_output)
    print(result.last_agent.name)

if __name__ == "__main__":
    asyncio.run(main())
