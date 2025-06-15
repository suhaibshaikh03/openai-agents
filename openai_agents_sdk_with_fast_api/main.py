from agents import Agent, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, ModelProvider, RunConfig
from agents.extensions.models.litellm_model import LitellmModel
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
import json
import os
from uuid import uuid4
from typing import cast
from pydantic import BaseModel, Field
from datetime import datetime, UTC
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseTextDeltaEvent


load_dotenv(find_dotenv())
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-1.5-flash"
set_tracing_disabled(disabled=True)

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyAz8O5klugfUrXBjaD45xytXHUZVRNkY6Y"



client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL
)

app = FastAPI(
    title="Suhaib Chatbot API",
    description="A FastAPI-based API for a chatbot",
    version="0.1.0",
)

class Metadata(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str = Field(default_factory=lambda: str(uuid4()))

class Message(BaseModel):
    user_id: str
    text: str
    metadata: Metadata | None = None
    tags: list[str] | None = None


class Response(BaseModel):
    user_id: str
    reply: str
    metadata: Metadata


Cloud_Agent = Agent(
    name="Cloud Computing Agent",
    instructions="Specialized Agent to answer queries regarding Cloud computing in the contect of Agentic AI",
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
    # model=LitellmModel(model="gemini/gemini-2.0-flash",api_key=GEMINI_API_KEY) 
)

OpenAI_agents_sdk_agent = Agent(
    name="Openai agents sdk agent",
    instructions="Specialized Agent to answer queries regarding the Openai agents sdk agentic framework",
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
    # model=LitellmModel(model="gemini/gemini-2.0-flash",api_key=GEMINI_API_KEY)

)
MobileDev_agent = Agent(
    name="Mobile Developer Agent",
    instructions="You are the MobileDev agent, an expert in mobile development. \
                Handle queries about iOS (Swift), Android (Kotlin/Java), and cross-platform tools like Flutter. Provide detailed guidance, code snippets, and mobile-specific solutions.\
                Respond only to mobile development-specific queries handed off by the Panaversity agent.",
     model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    # model=LitellmModel(model="gemini/gemini-2.0-flash",api_key=GEMINI_API_KEY),
    handoff_description="handles all queries regarding MObile Development and it's development"
)

WebDev_agent = Agent(
    name="Web Developer Agent",
    instructions="You are the WebDev agent, an expert in web development. \
                 Handle queries about HTML, CSS, JavaScript, and web frameworks. Provide detailed code examples, best practices, and troubleshooting advice.\
                 Respond only to web development-specific queries handed off by the Panaversity agent.",
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    # model=LitellmModel(model="gemini/gemini-2.0-flash",api_key=GEMINI_API_KEY),
    handoff_description="specialized in handling all queries regarding Web development and it's development"
)

AgenticAI_Agent = Agent(
    name="Agentic AI Agent",
    instructions="You are the AgenticAI agent, an expert in agentic AI systems.\
                 Handle queries about AI agent design, implementation, and optimization.\
                 Use the CloudAgent and OpenAIAgent as tools to assist with cloud-based processing and advanced AI model interactions. \
                 Respond only to agentic AI-specific queries handed off by the Panaversity agent.",
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),             
    # model=LitellmModel(model="gemini/gemini-2.0-flash",api_key=GEMINI_API_KEY),
    handoff_description="specialized in handling all queries regaring Agentic AI and it's development",
    tools=[
        Cloud_Agent.as_tool(
            tool_name="Cloud_Agent",
            tool_description="A tool for retrieving cloud-based information in the field of Agentic AI",
        ),
        OpenAI_agents_sdk_agent.as_tool(
            tool_name="OpenAI_agents_sdk_Agent",
            tool_description="A tool for retrieving information OpenAI agent sdk agentic framework in the field of Agentic AI"
        )
    ]
)


Panacloud_agent = Agent(
      name="Panacloud Agent",
      instructions="You are the Panacloud agent, a knowledgeable supervisor and teacher agent. \
                  Your role is to answer general queries about education, technology, and development. \
                  To hand off, analyze the query: if it contains 'web', 'HTML', 'CSS', or 'JavaScript', transfer to WebDev_agent; \
                  if it contains 'mobile', 'iOS', 'Android', or 'Flutter', transfer to MobileDev_agent; \
                  if it contains 'agentic', 'AI agent', or 'cloud' in the context of AI, transfer to AgenticAI_agent. \
                  For all other queries, provide clear, concise, and educational responses with a teaching tone.",
      handoffs=[WebDev_agent, AgenticAI_Agent, MobileDev_agent],
      model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    #   model=LitellmModel(model="gemini/gemini-2.0-flash",api_key=GEMINI_API_KEY)
  )

@app.post("/chat/", response_model=Response)
async def chat(message: Message):
    if not message.text.strip():
        raise HTTPException(
            status_code=400, detail="Message text cannot be empty")

    # Use the OpenAI Agents SDK to process the message
    result = await Runner.run(starting_agent=Panacloud_agent, input=message.text)
    reply_text = result.final_output  # Get the agent's response
    print(result.last_agent.name)

    return Response(
        user_id=message.user_id,
        reply=reply_text,
        metadata=Metadata()
    )

async def stream_response(message: Message):
    result = Runner.run_streamed(starting_agent=Panacloud_agent, input=message.text)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
            # Serialize dictionary to JSON string
            chunk = json.dumps({"chunk": event.data.delta})
            yield f"data: {chunk}\n\n"
            
@app.post("/chat/stream", response_model=Response)
async def chat_stream(message: Message):
    if not message.text.strip():
        raise HTTPException(
            status_code=400, detail="Message text cannot be empty")

    return StreamingResponse(
        stream_response(message),
        media_type="text/event-stream"
    )

