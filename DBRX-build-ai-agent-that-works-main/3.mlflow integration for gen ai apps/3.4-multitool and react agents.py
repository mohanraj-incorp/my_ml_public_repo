# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # ReAct Agent with LangGraph - Reasoning and Acting Pattern
# MAGIC
# MAGIC This code creates a **ReAct (Reasoning and Acting) agent** using LangGraph's prebuilt functionality, which combines **reasoning** and **tool usage** in an iterative loop.
# MAGIC
# MAGIC ## 🔧 **Setup & Configuration**
# MAGIC ```python
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC system_prompt = "You are a helpful assistant that can run Python code."
# MAGIC ```
# MAGIC - **LLM**: Databricks Llama 3.3 70B model
# MAGIC - **System prompt**: Defines the agent's behavior
# MAGIC
# MAGIC ## 🛠️ **Simple Tool Definitions**
# MAGIC ```python
# MAGIC def get_weather(city: str) -> str:
# MAGIC     """Get weather for a given city."""
# MAGIC     return f"The weather in {city} is sunny."
# MAGIC
# MAGIC def get_time(zone: str) -> str:
# MAGIC     """Get current time for a given timezone."""
# MAGIC     return f"The current time in {zone} is 2:00 PM."
# MAGIC ```
# MAGIC - **Simple functions**: Basic weather and time tools
# MAGIC - **Docstrings**: Help the LLM understand what each tool does
# MAGIC - **Mock responses**: Return static responses for demonstration
# MAGIC
# MAGIC ## 🤖 **ReAct Agent Creation**
# MAGIC ```python
# MAGIC agent_graph = create_react_agent(
# MAGIC     model=llm,
# MAGIC     tools=tools,
# MAGIC     prompt=system_prompt
# MAGIC )
# MAGIC ```
# MAGIC **Key advantage**: One-line agent creation using LangGraph's prebuilt ReAct implementation
# MAGIC
# MAGIC ## 🔄 **How ReAct Works**
# MAGIC The ReAct pattern follows this loop:
# MAGIC
# MAGIC 1. **Thought**: LLM reasons about the user's question
# MAGIC 2. **Action**: Decides which tool to use (or if to respond directly)
# MAGIC 3. **Observation**: Sees the tool's result
# MAGIC 4. **Repeat**: Continues reasoning until it has enough information
# MAGIC 5. **Answer**: Provides final response to the user
# MAGIC
# MAGIC **Example Flow**:
# MAGIC
# MAGIC
# MAGIC **Example Flow**:
# MAGIC 1. **User**: "What's the weather in Paris and what time is it in Tokyo?"
# MAGIC 2. **Thought**: I need to get weather for Paris and time for Tokyo
# MAGIC 3. **Action**: get_weather("Paris")
# MAGIC 4. **Observation**: "The weather in Paris is sunny"
# MAGIC 5. **Thought**: Now I need the time in Tokyo
# MAGIC 6. **Action**: get_time("Tokyo") 
# MAGIC 7. **Observation**: "The current time in Tokyo is 2:00 PM"
# MAGIC 8. **Thought**: I have both pieces of information now
# MAGIC 9. **Answer**: "The weather in Paris is sunny and it's 2:00 PM in Tokyo"
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## 🎯 **Key Benefits of ReAct Pattern**
# MAGIC - **Iterative reasoning**: Can use multiple tools in sequence
# MAGIC - **Self-correction**: Can reason about tool results and adjust
# MAGIC - **Transparency**: Shows its "thinking" process
# MAGIC - **Flexible**: Can handle complex multi-step problems
# MAGIC - **Automatic**: LangGraph handles the reasoning loop
# MAGIC
# MAGIC ## 🚀 **Prebuilt Advantage**
# MAGIC Instead of manually building the graph with nodes and edges, `create_react_agent()`:
# MAGIC - **Auto-creates** the reasoning loop
# MAGIC - **Handles** tool calling logic
# MAGIC - **Manages** state transitions
# MAGIC - **Provides** built-in ReAct prompting
# MAGIC
# MAGIC This makes creating sophisticated reasoning agents incredibly simple compared to building the graph manually.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using langraph create_react_agent

# COMMAND ----------

import json
from typing import Any, Generator
from uuid import uuid4

import mlflow
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langchain_core.messages import AIMessageChunk
from langgraph.prebuilt import create_react_agent

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

############################################
# Define LLM and tools
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# UC_TOOL_NAMES = ["system.ai.python_exec"]
# uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# tools = uc_toolkit.tools


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny."

def get_time(zone: str) -> str:
    """Get current time for a given timezone."""
    return f"The current time in {zone} is 2:00 PM."


    
  

tools = [get_weather,get_time]

system_prompt = "You are a helpful assistant that give weather and time information."

############################################
# Build the ReAct agent automatically
############################################
agent_graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,  # this injects your system prompt
)





# COMMAND ----------

############################################
# Register the agent with MLflow
############################################
from helpers import LangGraphResponsesAgent
mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent(agent_graph)
mlflow.models.set_model(AGENT)

# COMMAND ----------

result = AGENT.predict({"input": [{"role": "user", "content": "What’s the weather in Paris and the time in EST?"}]})
print(result.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC https://python.langchain.com/docs/integrations/tools/

# COMMAND ----------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# --- create Wikipedia tool ---
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# COMMAND ----------

tools = [get_weather,get_time,wiki]

prompt = """
You are a helpful AI assistant. You can think step-by-step and use the following tools to answer user questions:

1. get_weather(city: str) — returns the current weather in the given city.
2. get_time(zone: str) — returns the current time for the given timezone.
3. WikipediaQueryRun(input: str) — searches Wikipedia and returns relevant information.

Rules:
- If a question needs factual information or background knowledge, prefer WikipediaQueryRun.
- If the user asks about weather, use get_weather.
- If the user asks about time, use get_time.
- Always return a clear final answer to the user after using tools.
- Think carefully about which tool is best for each part of the question. 
- You can use multiple tools in one conversation if the question needs it.
- Do not guess values you can fetch with a tool — always call the correct tool.

Follow the ReAct pattern:
- First, think about what is needed.
- Then, call the right tool with correct arguments.
- Finally, summarize the result back to the user.
"""


############################################
# Build the ReAct agent automatically
############################################
agent_graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,  # this injects your system prompt
)


# COMMAND ----------

############################################
# Register the agent with MLflow
############################################
from helpers import LangGraphResponsesAgent
mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent(agent_graph)
mlflow.models.set_model(AGENT)

# COMMAND ----------

result = AGENT.predict({"input": [{"role": "user", "content": "What’s the weather in Paris and the time in EST? and search wikipedia for fun facts about paris"}]})
print(result.model_dump(exclude_none=True))
