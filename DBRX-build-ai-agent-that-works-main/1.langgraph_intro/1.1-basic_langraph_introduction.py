# Databricks notebook source
# MAGIC %md
# MAGIC ![1.png](./images/langchain_and_langgraph.png "1.png")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Why Use Frameworks like LangChain & LangGraph?
# MAGIC
# MAGIC Even though you *can* write plain Python, frameworks help with things that get complicated fast:
# MAGIC
# MAGIC - **Workflow Management:** Organize multi-step tasks, agents, and APIs cleanly.  
# MAGIC - **State Management:** Keep track of variables, context, or intermediate results across steps.  
# MAGIC - **Memory:** Remember past interactions or user inputs for smarter responses.  
# MAGIC - **Retries & Error Handling:** Automatically retry failed steps or handle exceptions.  
# MAGIC - **Reusability & Maintainability:** Reuse components, swap models/tools without rewriting everything.  
# MAGIC - **Dynamic Logic & Branching:** Easily implement loops, conditional paths, and agent collaboration.  
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic use of LangChain
# MAGIC

# COMMAND ----------

from databricks_langchain import ChatDatabricks

# Initialize model
llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")

print(llm.invoke("What is Databricks?"))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, AIMessage
import json

# Initialize model
llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")

# Store all messages
messages = []

# 1️⃣ User message
user_msg = HumanMessage(content="What is Databricks?")
messages.append(user_msg)

# 2️⃣ Model response
response = llm.invoke(messages)

# Save response
messages.append(AIMessage(content=response.content))

# 3️⃣ Print everything nicely
print("\n--- All messages so far ---")
print(f'\n\n messages: {messages} \n\n\n')
for m in messages:
    print(f"\n[{m.type.upper()} MESSAGE]")
    if isinstance(m.content, list):
        # Print structured message (reasoning + text)
        for part in m.content:
            print(f"  - Type: {part.get('type')}")
            if 'summary' in part:
                print("    Summary:")
                print(json.dumps(part['summary'], indent=4))
            if 'text' in part:
                print("    Text:")
                print(part['text'][:1000])  # truncate long text for readability
    else:
        # Simple message (string)
        print(m.content)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC **Databricks** is a cloud‑native data‑and‑AI platform built around **Apache Spark** that provides a unified workspace for data engineering, data science, machine learning (ML), and analytics. It combines a managed Spark service with a collaborative notebook environment, a data‑lakehouse architecture, and a suite of tools for the entire data lifecycle.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. Core Idea – the “Lakehouse”
# MAGIC | Concept | Traditional Approach | Databricks’ Approach |
# MAGIC |---------|----------------------|----------------------|
# MAGIC | **Data storage** | Separate data warehouse (structured) + data lake (raw files) | **Lakehouse**: a single open‑format storage layer (Delta Lake) that offers both ACID transactions (warehouse‑like guarantees) and low‑cost object‑store scalability (lake‑like). |
# MAGIC | **Governance** | Multiple tools, fragmented metadata | **Unity Catalog** (centralized metadata, fine‑grained access control). |
# MAGIC | **Performance** | Queries on warehouse are fast; lake queries are slower | **Delta Lake** + **P

# COMMAND ----------

# MAGIC %md
# MAGIC # 🧩 Simple StateGraph Workflow — Concepts Explained
# MAGIC
# MAGIC This code creates a **state-based workflow** using LangGraph. Instead of calling functions directly in sequence, we define:
# MAGIC
# MAGIC - A **State**: a typed dictionary (`TypedDict`) that describes what data flows between nodes (`graph_state` in this case).
# MAGIC - **Nodes**: plain Python functions (`node_1`, `node_2`, `node_3`) that:
# MAGIC   - Receive the current `state` (a dict)
# MAGIC   - Perform some operation (here, just string concatenation)
# MAGIC   - Return an updated `state` (partial dict with changed values)
# MAGIC
# MAGIC - A **StateGraph**: an object that stores the nodes and how they connect.
# MAGIC   - We add each node to the graph by name.
# MAGIC   - We define **edges** between nodes — this describes the execution path (`START → node_1 → node_2 → node_3 → END`).
# MAGIC
# MAGIC - **Compilation**: `builder.compile()` turns the defined graph into an executable workflow.
# MAGIC - **Visualization**: `graph.get_graph().draw_mermaid_png()` renders a diagram of the graph so you can see the structure visually in the notebook.
# MAGIC
# MAGIC **Conceptually:**  
# MAGIC This pattern abstracts workflows into **directed graphs**. Each node is independent, the graph controls the order, and the shared `state` moves through the graph — a foundation for building more complex, branching, or parallel agent systems.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 17
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
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END


from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str

def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}



























# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)












# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
print(graph.get_graph().draw_mermaid())
#display(Image(graph.get_graph().draw_mermaid_png(max_retries=5,retry_delay=2)))

# COMMAND ----------

graph.invoke({"graph_state" : "Hi, this is Anirvan."})


# COMMAND ----------

# MAGIC %md
# MAGIC # 🔀 Conditional Branching in a LangGraph Workflow
# MAGIC
# MAGIC In this example, we introduce **dynamic decision-making** into our StateGraph. Instead of always following a fixed sequence, a node can decide which next node to run at runtime.
# MAGIC
# MAGIC We do this in three steps:
# MAGIC
# MAGIC 1. **Decision Function**  
# MAGIC    We create a Python function (here called `decide_mood`) that takes the current `state` and returns the name of the next node.  
# MAGIC    In this demo, it just flips a coin: 50% chance to go to `node_2`, 50% to `node_3`. In a real project, this function could look at values in `state` (like user input, an LLM output, or an API response) to pick the right branch.
# MAGIC
# MAGIC 2. **Attach Conditional Edges**  
# MAGIC    Instead of a normal `add_edge`, we use `add_conditional_edges("node_1", decide_mood)`.  
# MAGIC    This tells LangGraph: after running `node_1`, call `decide_mood(state)` and follow whatever node name it returns. This is how branching logic is added without hardcoding paths.
# MAGIC
# MAGIC 3. **Complete the Graph**  
# MAGIC    Both possible branches (`node_2` and `node_3`) are then connected to `END`. This ensures no matter which branch is chosen, the workflow eventually terminates cleanly.
# MAGIC
# MAGIC **Conceptually:**  
# MAGIC - Nodes still work the same way — they process and update the `state`.  
# MAGIC - The graph itself becomes more flexible, able to handle different paths dynamically.  
# MAGIC - This pattern is the foundation for routers, conditional processing, or any workflow that depends on runtime data to decide the next step.
# MAGIC
# MAGIC

# COMMAND ----------

import random
from typing import Literal


def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Cell 23
# Build graph
builder1 = StateGraph(State)  # Ensure StateGraph is imported or defined before this line
builder1.add_node("node_1", node_1)
builder1.add_node("node_2", node_2)
builder1.add_node("node_3", node_3)

# Logic
builder1.add_edge(START, "node_1")
builder1.add_conditional_edges("node_1", decide_mood)
builder1.add_edge("node_2", END)
builder1.add_edge("node_3", END)

# Add
graph1 = builder1.compile()

# View
print(graph1.get_graph().draw_mermaid())

# COMMAND ----------

graph1.invoke({"graph_state" : "Hi, this is Anirvan."})
