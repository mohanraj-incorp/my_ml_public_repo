# Databricks notebook source
# MAGIC %md
# MAGIC DOC link : https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use ResponsesAgent to author agents

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks recommends the MLflow interface ResponsesAgent to create production-grade agents. ResponsesAgent lets you build agents with any third-party framework, then integrate it with Databricks AI features for **robust logging, tracing, evaluation, deployment, and monitoring capabilities.**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ![](https://docs.databricks.com/aws/en/assets/images/responses-agent-overview-611d843718bf94974d277a365695043c.svg)

# COMMAND ----------

# MAGIC %md
# MAGIC # LangGraph + MLflow ResponsesAgent Integration
# MAGIC
# MAGIC This code creates a **production-ready AI agent** by combining LangGraph's tool-calling capabilities with MLflow's ResponsesAgent interface for deployment.
# MAGIC
# MAGIC ## 🔧 **Setup & Configuration**
# MAGIC ```python
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC system_prompt = "You are a helpful assistant that can run Python code."
# MAGIC ```
# MAGIC - **LLM**: Databricks Llama 3.3 70B model
# MAGIC - **System prompt**: Defines the agent's behavior and capabilities
# MAGIC
# MAGIC ## 🛠️ **Tools Integration**
# MAGIC ```python
# MAGIC UC_TOOL_NAMES = ["system.ai.python_exec"]  # Unity Catalog Python executor
# MAGIC VECTOR_SEARCH_TOOLS = []  # Placeholder for vector search tools
# MAGIC ```
# MAGIC - **UC Tools**: Includes Python code execution capability from Unity Catalog
# MAGIC - **Vector Search**: Can be extended with document retrieval tools
# MAGIC - **Extensible**: Easy to add more tools as needed
# MAGIC
# MAGIC ## 🕸️ **LangGraph Agent Creation**
# MAGIC ```python
# MAGIC def create_tool_calling_agent(model, tools, system_prompt):
# MAGIC     # Creates workflow: agent → [conditional] → tools → agent → END
# MAGIC ```
# MAGIC - **State management**: `AgentState` with messages and custom inputs/outputs
# MAGIC - **Conditional routing**: Continues to tools if LLM makes tool calls, ends otherwise
# MAGIC - **Tool binding**: LLM gets access to all defined tools
# MAGIC
# MAGIC ## 🌉 **MLflow Bridge - Key Innovation**
# MAGIC ```python
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def _responses_to_cc(self, message): # Convert ResponsesAgent → LangChain
# MAGIC     def _langchain_to_responses(self, messages): # Convert LangChain → ResponsesAgent
# MAGIC ```
# MAGIC **Purpose**: Makes LangGraph compatible with MLflow serving infrastructure
# MAGIC
# MAGIC **Format Translation**:
# MAGIC - **Input**: Converts MLflow ResponsesAgent format to LangChain messages
# MAGIC - **Output**: Converts LangChain responses back to ResponsesAgent format
# MAGIC - **Streaming**: Handles real-time response streaming
# MAGIC
# MAGIC ## 🚀 **Deployment Integration**
# MAGIC ```python
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphResponsesAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)
# MAGIC ```
# MAGIC - **MLflow registration**: Makes the agent deployable via MLflow
# MAGIC - **Production ready**: Can be served as a Databricks model endpoint
# MAGIC - **Auto-logging**: Tracks all interactions for monitoring
# MAGIC
# MAGIC ## 🎯 **Key Benefits**
# MAGIC - **Best of both worlds**: LangGraph's flexibility + MLflow's deployment infrastructure
# MAGIC - **Tool calling**: Advanced function calling capabilities
# MAGIC - **Streaming support**: Real-time response generation
# MAGIC - **Enterprise deployment**: Direct integration with Databricks serving
# MAGIC - **Format compatibility**: Seamless translation between different agent formats
# MAGIC
# MAGIC This pattern enables deploying sophisticated LangGraph agents as production MLflow models while maintaining full tool-calling and streaming capabilities.

# COMMAND ----------

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

############################################
# Define your LLM endpoint and system prompt
############################################x
# TODO: Replace with your model serving endpoint
# LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = "You are a helpful assistant that can run Python code."

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
# Below, we add the `system.ai.python_exec` UDF, which provides
# a python code interpreter tool to our agent
# You can also add local LangChain python tools. See https://python.langchain.com/docs/concepts/tools

# TODO: Add additional tools
UC_TOOL_NAMES = ["system.ai.python_exec"]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = []

# To add vector search retriever tools,
# use VectorSearchRetrieverTool and create_tool_info,
# then append the result to TOOL_INFOS.
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )

tools.extend(VECTOR_SEARCH_TOOLS)

#####################
## Define agent logic
#####################


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


############################################
# Bridge to MLflow ResponsesAgent
############################################
class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        #Converts MLflow-style messages into LangChain-style messages so the agent can process them
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [{"role": message["role"], "content": c["text"]} for c in message["content"]]
        return [{"role": message.get("role", "assistant"), "content": message.get("content", "")}]

    def _langchain_to_responses(self, messages) -> list[dict[str, Any]]:
        #Converts LangChain message objects back into MLflow ResponsesAgent items so MLflow can output or stream them
        outputs = []
        for message in messages:
            message = message.model_dump()
            if message["type"] == "ai":
                if tool_calls := message.get("tool_calls"):
                    for tc in tool_calls:
                        outputs.append(
                            self.create_function_call_item(
                                id=message.get("id") or str(uuid4()),
                                call_id=tc["id"],
                                name=tc["name"],
                                arguments=json.dumps(tc["args"]),
                            )
                        )
                else:
                    outputs.append(
                        self.create_text_output_item(
                            text=message["content"],
                            id=message.get("id") or str(uuid4()),
                        )
                    )
            elif message["type"] == "tool":
                outputs.append(
                    self.create_function_call_output_item(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                    )
                )
        return outputs

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        
        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        if isinstance(item, (bool, str, bytes, int, float)) or item is None:
                            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                chunk = event[1][0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    if isinstance(content, (bool, str, bytes, int, float)) or content is None:
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)


# COMMAND ----------

result = AGENT.predict({"input": [{"role": "user", "content": "What is 6*7 in Python?"}]})
print(result.model_dump(exclude_none=True))
