# Databricks notebook source
# Imports
import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union
from uuid import uuid4

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)


# COMMAND ----------


LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)


system_prompt = """You are the Foodly Policy Agent.  
Answer ONLY questions about Foodly’s official policies (refunds, cancellations, delivery, loyalty, privacy, fraud, escalation).  
Always use the vector search tool to find the relevant section.  
Summarize or quote faithfully; never guess, invent, or take actions.  
If no policy is found, reply: “I couldn’t find any official Foodly policy about that. Would you like me to escalate this?”  
Keep responses clear, friendly, and precise"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# Initialize the retriever tool.
vs_tool = VectorSearchRetrieverTool(
  index_name="agents.main.foodly_index_name_test",
  tool_name="foodly_policy_document_retrieval_tool",
  num_results=2,
  tool_description="Use this tool to search the Foodly knowledge base for policies, procedures, and service-related information. It retrieves the most relevant chunks from the company’s official documentation, including refund rules, cancellation terms, delivery guidelines, loyalty program details, privacy policies, and escalation procedures"
)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = [vs_tool]


tools.extend(VECTOR_SEARCH_TOOLS)



# COMMAND ----------

from helpers import LangGraphResponsesAgent, create_tool_calling_agent
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What are foodly return policies?"}]}
):
    print(chunk.model_dump(exclude_none=True))
