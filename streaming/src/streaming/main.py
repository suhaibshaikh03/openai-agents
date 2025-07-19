import chainlit as cl
from dotenv import load_dotenv, find_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
load_dotenv(find_dotenv())
from openai.types.responses import ResponseTextDeltaEvent
gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent1 = Agent(
    instructions="You are a helpful assistant that can answer questions",
    name="Panaversity Support Agent"
)



# print(result.final_output)
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm the panaversity Support Agent. How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    msg = cl.Message(content="")
    await msg.send()
    history.append({"role":"user","content":message.content})
    result = Runner.run_streamed(
    agent1,
    input=history,
    run_config=config,
    
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    history.append({"role":"assistant","content":result.final_output})
    cl.user_session.set("history", history)
    #await cl.Message(content=result.final_output).send()






# import asyncio

# from openai.types.responses import ResponseTextDeltaEvent

# from agents import Agent, Runner, function_tool

# @function_tool
# def get_weather(city:str):
#   """Gets weather for a city"""
#   return f"The weather in {city} is sunny"

# async def main():
#     agent = Agent(
#         name="Assistant",
#         instructions="You are a helpful assistant.",
#         model="gpt-4.1",
#         tools=[get_weather]

#     )

#     result = Runner.run_streamed(agent, input="What is the weather in New York?")
#     async for event in result.stream_events():
#         if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
#             print(event.data.delta, end="", flush=True)
#         elif event.type == "agent_updated_stream_event":
#             print(f"Agent updated: {event.new_agent.name}")
#         elif event.type == "run_item_stream_event":
#           print(event.name)



# asyncio.run(main())
