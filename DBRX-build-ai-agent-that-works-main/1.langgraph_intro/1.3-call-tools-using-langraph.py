# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # LangGraph Tool Calling - Simple Explanation
# MAGIC
# MAGIC This code builds a **conversational AI system** that can use **tools/functions** to answer questions, using LangGraph + Databricks LLM.
# MAGIC
# MAGIC
# MAGIC ## 🛠️ **Tool Definition**
# MAGIC ```python
# MAGIC def multiply(a: int, b: int) -> int:
# MAGIC     """Multiply two numbers"""
# MAGIC     return a * b
# MAGIC ```
# MAGIC A simple function that multiplies two numbers. The LLM can call this when users ask math questions.
# MAGIC
# MAGIC ## 🤖 **LLM Setup**
# MAGIC ```python
# MAGIC llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")
# MAGIC llm_with_tools = llm.bind_tools([multiply])  # Give LLM access to the multiply tool
# MAGIC ```
# MAGIC
# MAGIC ## 🕸️ **Graph Structure**
# MAGIC
# MAGIC
# MAGIC START → tool_calling_llm → [conditional] → tools → tool_calling_llm → END
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC - **tool_calling_llm**: LLM decides whether to use tools or respond directly
# MAGIC - **tools**: Executes the actual tool (multiply function)
# MAGIC - **Conditional edge**: Routes to tools only if LLM wants to use them
# MAGIC
# MAGIC ## 🔄 **How It Works**
# MAGIC 1. User asks: *"What is the weather in Tokyo? and what is 2 * 3?"*
# MAGIC 2. LLM analyzes the question
# MAGIC 3. For "2 * 3", it calls the `multiply(2, 3)` tool
# MAGIC 4. Gets result `6` from the tool
# MAGIC 5. Responds with both weather info and math result
# MAGIC
# MAGIC ## 🎯 **Key Features**
# MAGIC - **Smart routing**: LLM automatically decides when to use tools
# MAGIC - **Tool integration**: Functions become available to the AI
# MAGIC - **Conversation flow**: Maintains chat history and context
# MAGIC - **Conditional logic**: Uses tools only when needed
# MAGIC
# MAGIC The output shows each message in the conversation, including tool calls and responses.

# COMMAND ----------

# DBTITLE 1,Cell 2
from dotenv import load_dotenv
import os
import random
from typing import Literal, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from langgraph.graph import MessagesState

# load_dotenv()


import mlflow
mlflow.langchain.autolog()

class State(MessagesState):
    pass



def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    """Example:
    >>> multiply(2, 3)
    6
    """
    """Args:
    a: int
    b: int
    """
    """Returns:
    int
    """
    return a * b


llm = ChatDatabricks(endpoint = "databricks-gpt-oss-120b")

llm_with_tools = llm.bind_tools([multiply])

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)


builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Hello,what is the weather in Tokyo? and what is 2 * 3?")]})
print(f'result={result}')

for i, message in enumerate(result["messages"]):
    print(f"Message {i+1} ({type(message).__name__}): {message.content}")




# COMMAND ----------

from dotenv import load_dotenv
import os
import random
from typing import Literal, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from langgraph.graph import MessagesState

# load_dotenv()


import mlflow
mlflow.langchain.autolog()

class State(MessagesState):
    pass



def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    """Example:
    >>> multiply(2, 3)
    6
    """
    """Args:
    a: int
    b: int
    """
    """Returns:
    int
    """
    return a * b


llm = ChatDatabricks(endpoint = "databricks-gpt-oss-120b")

llm_with_tools = llm.bind_tools([multiply])

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))


builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Hello,what is the weather in Tokyo? and what is 2 * 3?")]})


for i, message in enumerate(result["messages"]):
    print(f"Message {i+1} ({type(message).__name__}): {message.content}")




# COMMAND ----------

# MAGIC %md
# MAGIC LangChain Inbuilt tools https://python.langchain.com/docs/integrations/tools/

# COMMAND ----------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# COMMAND ----------

from dotenv import load_dotenv
import os
import random
from typing import Literal, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from langgraph.graph import MessagesState

# load_dotenv()


import mlflow
mlflow.langchain.autolog()

class State(MessagesState):
    pass




llm = ChatDatabricks(endpoint = "databricks-gpt-oss-120b")

llm_with_tools = llm.bind_tools([wikipedia])

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([wikipedia]))


builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Tell me about databricks.")]})


for i, message in enumerate(result["messages"]):
    print(f"Message {i+1} ({type(message).__name__}): {message.content}")


