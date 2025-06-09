from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api
import os
from dotenv import load_dotenv, find_dotenv
from agents import enable_verbose_stdout_logging, function_tool
_:bool = load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(disabled=True)
set_default_openai_api("chat_completions")

enable_verbose_stdout_logging()


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)
@function_tool
def get_current_weather(city:str)->str:
    """Return current weather of any city"""
    return "Curent weather for {city} is sunny"

@function_tool
def get_current_news(topic:str)->str:
    """Returns current news about a specific topic"""
    return "Latest news in {topic} is Agent Native Cloud Development"


agent: Agent = Agent(name="Assistant", 
                    instructions="You are a helpful assistant",
                    model="gemini-2.0-flash",
                    tools=[get_current_weather,get_current_news] 
                     )

result = Runner.run_sync(agent, "What is the current weather in Karachi and what is the current latest news in tech?")

print(result.final_output)