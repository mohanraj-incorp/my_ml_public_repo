# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # LangGraph with Databricks LLM - Code Explanation
# MAGIC
# MAGIC This code demonstrates a simple conversational AI workflow using **LangGraph** (a graph-based framework for building language model applications) integrated with **Databricks' LLM serving endpoint**.
# MAGIC
# MAGIC
# MAGIC ## 🤖 **Core Function: `call_llm`**
# MAGIC
# MAGIC ```python
# MAGIC def call_llm(state: MessagesState):
# MAGIC     system_msg = {"role": "system", "content": "Reply only with plain text. No formatting."}
# MAGIC     all_msgs = [system_msg] + state["messages"]
# MAGIC     return {"messages": [llm.invoke(all_msgs)]}
# MAGIC ```
# MAGIC
# MAGIC **Purpose**: This function processes the conversation state and generates LLM responses.
# MAGIC
# MAGIC **How it works**:
# MAGIC 1. **System Message**: Adds a system prompt instructing the LLM to respond in plain text only (no markdown/formatting)
# MAGIC 2. **Message Preparation**: Combines the system message with existing conversation messages
# MAGIC 3. **LLM Invocation**: Calls the Databricks LLM with the complete message history
# MAGIC 4. **State Update**: Returns the new message to be added to the conversation state
# MAGIC
# MAGIC ## 🕸️ **Graph Construction**
# MAGIC
# MAGIC ```python
# MAGIC builder = StateGraph(MessagesState)  # Create graph with message state management
# MAGIC builder.add_node("call_llm", call_llm)  # Add the LLM processing node
# MAGIC builder.add_edge(START, "call_llm")  # Connect start to LLM node
# MAGIC builder.add_edge("call_llm", END)  # Connect LLM node to end
# MAGIC graph = builder.compile()  # Compile the graph for execution
# MAGIC ```
# MAGIC
# MAGIC **Graph Flow**: `START → call_llm → END`
# MAGIC
# MAGIC This creates a simple linear workflow where:
# MAGIC - The conversation starts
# MAGIC - Messages are processed by the LLM
# MAGIC - The conversation ends
# MAGIC
# MAGIC ## 🚀 **Execution**
# MAGIC
# MAGIC ```python
# MAGIC messages = graph.invoke({"messages": [HumanMessage("Tell me more about Databricks ?")]})
# MAGIC ```
# MAGIC
# MAGIC **What happens**:
# MAGIC 1. **Input**: Creates a human message asking about Databricks
# MAGIC 2. **Processing**: The graph processes this through the `call_llm` node
# MAGIC 3. **Output**: Returns the complete conversation including the LLM's response
# MAGIC 4. **Result**: The `messages` variable contains both the input question and the generated answer
# MAGIC
# MAGIC ## 🎯 **Key Concepts**
# MAGIC
# MAGIC - **StateGraph**: Manages conversation state and message flow
# MAGIC - **MessagesState**: Built-in state type for handling conversation messages
# MAGIC - **Node**: A processing unit in the graph (here, the LLM call)
# MAGIC - **Edges**: Define the flow between nodes
# MAGIC - **Invoke**: Executes the graph with given input
# MAGIC
# MAGIC ## 💡 **Use Cases**
# MAGIC
# MAGIC This pattern is useful for:
# MAGIC - Building conversational AI applications
# MAGIC - Creating chatbots with Databricks LLMs
# MAGIC - Implementing structured conversation flows
# MAGIC - Integrating with larger AI workflows
# MAGIC
# MAGIC The code represents a foundational building block that can be extended with additional nodes for more complex behaviors like tool calling, memory management, or multi-step reasoning.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

from dotenv import load_dotenv
import os
import random
from typing import Literal, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState


llm = ChatDatabricks(endpoint = "databricks-gpt-oss-120b")


# In LangGraph, MessageState is a built-in helper class (or type) that represents the state of messages flowing through a LangGraph state graph — especially useful when you’re building conversational or agentic workflows where messages (like chat history) are passed between nodes.

def call_llm(state: MessagesState):

    system_msg = {"role": "system", "content": "Reply only with plain text. No formatting."}
    all_msgs = [system_msg] + state["messages"]

    return {"messages": [llm.invoke(all_msgs)]}


builder = StateGraph(MessagesState)
builder.add_node("call_llm", call_llm)


builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

graph = builder.compile()

messages = graph.invoke({"messages": [HumanMessage("Tell me more about Databricks ?")]})

# COMMAND ----------

message = messages['messages'][-1]

for part in message.content:
    if part.get("type") == "text":
       ai_message = part.get("text", "")

print(ai_message)
