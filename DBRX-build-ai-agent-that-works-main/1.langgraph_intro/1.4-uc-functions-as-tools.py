# Databricks notebook source
# MAGIC %md
# MAGIC # LangGraph with Unity Catalog Functions - Tool Calling System
# MAGIC
# MAGIC This code creates a **conversational AI** that can call **Unity Catalog functions** as tools, using LangGraph + Databricks.
# MAGIC
# MAGIC ## 🔧 **Setup & Configuration**
# MAGIC - **MLflow tracking**: `mlflow.langchain.autolog()` logs all LLM interactions
# MAGIC - **State management**: `State(MessagesState)` handles conversation history
# MAGIC - **LLM**: Databricks GPT OSS 120B model endpoint
# MAGIC
# MAGIC ## 🗃️ **Unity Catalog Integration**
# MAGIC ```python
# MAGIC uc_tool_names = ("agents.main.*",)  # Pattern to match UC functions
# MAGIC tools = UCFunctionToolkit(function_names=list(uc_tool_names)).tools
# MAGIC ```
# MAGIC - **UCFunctionToolkit**: Automatically discovers and loads functions from Unity Catalog
# MAGIC - **Pattern matching**: `"agents.main.*"` loads all functions from the `agents.main` schema
# MAGIC - **Auto-discovery**: No need to manually define tools - UC functions become AI tools
# MAGIC
# MAGIC ## 🤖 **LLM + Tools Setup**
# MAGIC ```python
# MAGIC llm_with_tools = llm.bind_tools(tools)  # LLM gets access to UC functions
# MAGIC ```
# MAGIC The LLM can now call any function from the Unity Catalog schema as needed.
# MAGIC
# MAGIC ## 🕸️ **Graph Workflow**
# MAGIC
# MAGIC
# MAGIC START → tool_calling_llm → [conditional routing] → tools → tool_calling_llm → END
# MAGIC
# MAGIC
# MAGIC
# MAGIC - **tool_calling_llm**: LLM decides whether to use UC functions or respond directly
# MAGIC - **tools**: Executes the actual Unity Catalog functions
# MAGIC - **Conditional routing**: Automatically routes to tools when LLM needs them
# MAGIC
# MAGIC ## 🔄 **Execution Flow**
# MAGIC 1. **User query**: *"Can you tell me more info about brittanyramos@example.org?"*
# MAGIC 2. **LLM analysis**: Determines if UC functions can help with this email query
# MAGIC 3. **Tool calling**: May call UC functions like user lookup, email validation, etc.
# MAGIC 4. **Response**: Combines tool results with LLM reasoning
# MAGIC
# MAGIC ## 🎯 **Key Benefits**
# MAGIC - **Enterprise integration**: Direct access to company's UC functions
# MAGIC - **Zero setup**: No manual tool definition required
# MAGIC - **Scalable**: Automatically discovers new UC functions
# MAGIC - **Secure**: Uses Unity Catalog's permissions and governance
# MAGIC - **Production ready**: Enterprise-grade function execution
# MAGIC
# MAGIC This pattern enables AI agents to use your organization's existing data functions and business logic stored in Unity Catalog.

# COMMAND ----------

# MAGIC %sql 
# MAGIC create catalog if not exists agents;
# MAGIC create schema if not exists agents.main;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get the customer details based on their email
# MAGIC
# MAGIC Let's add a function to retrieve a customer detail based on their email.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION agents.main.get_customer_by_email(email_input STRING COMMENT 'customer email used to retrieve customer information')
# MAGIC RETURNS TABLE (
# MAGIC     customerID BIGINT,
# MAGIC     first_name STRING,
# MAGIC     last_name STRING,
# MAGIC     email_address STRING,
# MAGIC     phone_number STRING,
# MAGIC     address STRING,
# MAGIC     city STRING,
# MAGIC     state STRING,
# MAGIC     postal_zip_code STRING,
# MAGIC     country STRING,
# MAGIC     continent STRING,
# MAGIC     gender STRING
# MAGIC )
# MAGIC COMMENT 'Returns the customer record matching the provided email address. Includes its customerID, first_name, last_name and more.'
# MAGIC RETURN (
# MAGIC     SELECT * FROM samples.bakehouse.sales_customers
# MAGIC     WHERE email_address = email_input
# MAGIC     LIMIT 1
# MAGIC );

# COMMAND ----------

# MAGIC %sql SELECT * FROM agents.main.get_customer_by_email('brittanyramos@example.org');

# COMMAND ----------

from dotenv import load_dotenv
import os
import random
from typing import Literal, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from langgraph.graph import MessagesState



import mlflow
mlflow.langchain.autolog()


class State(MessagesState):
    pass




llm = ChatDatabricks(endpoint = "databricks-gpt-oss-120b")


uc_tool_names  = ("agents.main.*",)


tools  = UCFunctionToolkit(function_names=list(uc_tool_names)).tools


llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))


builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Can you tell me more info about brittanyramos@example.org?")]})


for i, message in enumerate(result["messages"]):
    print(f"Message {i+1} ({type(message).__name__}): {message.content}")




# COMMAND ----------

result = graph.invoke({"messages": [HumanMessage("Which table are you using to get the user details")]})


for i, message in enumerate(result["messages"]):
    print(f"Message {i+1} ({type(message).__name__}): {message.content}")

# COMMAND ----------



# COMMAND ----------

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. 
An input string is valid if: 
Open brackets must be closed by the same type of brackets. 
Open brackets must be closed in the correct order. 
Every close bracket has a corresponding open bracket of the same type. 
  
Example 1: 
Input: s = "()" 
Output: true 
Example 2: 
Input: s = "()[]{}" 
Output: true 
Example 3: 
Input: s = "(]" 
Output: false 
Example 4: 
Input: s = "([])" 
Output: true 
Example 5: 
Input: s = "([)]" 
Output: false 
 
 
Monday 2:23 PM Meeting ended: 57m 11s
 

# COMMAND ----------



def sum_divd(dvdnd, dvsr):
    sign =
    j=abs(dvsr)
    i=1
    while(i<= 231):
       if j <= dvdnd:
          j=j+dvsr
          i=i+1
       else:
          break
    return i-1 

print(sum_divd(121,4))
