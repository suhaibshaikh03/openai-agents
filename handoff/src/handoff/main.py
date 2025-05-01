#uv run src/hanoff.main.pys
from dotenv import load_dotenv, find_dotenv
import os
from agents.run import RunConfig
load_dotenv(find_dotenv())
from agents import function_tool
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

# @function_tool
# def get_current_weather(city:str) -> str:
#     """takes city name as argument and returns it's weather condition"""
#     return f"{city} weather is sunny"


panaversity_agent: Agent = Agent(name="panaversity_agent",
                     instructions="You are a helpful panaversity assistant. You answer query about panaversity",
                     model="gemini-2.0-flash",
                     handoff_description="Panaversity Expert Agent",
                     )

agentic_ai_expert: Agent = Agent(name="agentic_ai_expert",
                     instructions="You are a helpful agentic ai expert assistant",
                     model="gemini-2.0-flash",
                     handoff_description="Agentic AI Expert Agent",
                     )

chat_agent: Agent = Agent(name="Assistant",
                     instructions="You are a helpful chat agent. You cater general user queries and handoff to other agents when needed",
                     model="gemini-2.0-flash",
                     handoffs=[panaversity_agent,agentic_ai_expert]
                     )


result = Runner.run_sync(starting_agent=chat_agent, input="who is the founder of panaversity")


print(result.final_output)
print(result.last_agent.name)