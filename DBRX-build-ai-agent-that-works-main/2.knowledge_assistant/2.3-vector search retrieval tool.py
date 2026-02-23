# Databricks notebook source
from dotenv import load_dotenv
import os
from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt.tool_node import ToolNode, tools_condition


from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool
)



# Initialize LLM
llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")

def call_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}

# COMMAND ----------

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your PDF file
file_path = "foodly_company_documents.pdf"

# Load the PDF document
loader = PyPDFLoader(file_path)
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

# Each document usually corresponds to one page
print(documents[0].metadata)
print(documents[0].page_content[:1000])

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # ~800 characters per chunk
    chunk_overlap=100,   # overlap to preserve context
    separators=["\n\n", "\n", " ", ""]
)

# Split the documents into chunks
docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks.")

for _doc in docs:
    print(_doc)
    print("-" * 100)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LangGraph with Vector Search Tool - RAG-Enabled AI Agent
# MAGIC
# MAGIC This code creates a **Retrieval-Augmented Generation (RAG)** system using LangGraph + Databricks Vector Search for intelligent document retrieval.
# MAGIC
# MAGIC ## 🔧 **Setup & Components**
# MAGIC - **LLM**: Databricks GPT OSS 120B model
# MAGIC - **Vector Search**: Databricks Vector Search as a retrieval tool
# MAGIC - **Graph**: LangGraph for orchestrating LLM + retrieval workflow
# MAGIC
# MAGIC ## 🔍 **Vector Search Tool Configuration**
# MAGIC ```python
# MAGIC vs_tool = VectorSearchRetrieverTool(
# MAGIC   index_name="agents.main.foodly_policy_embedding_index",
# MAGIC   tool_name="foodly_policy_document_retrieval_tool", 
# MAGIC   num_results=2,
# MAGIC   tool_description="Search Foodly knowledge base for policies, procedures..."
# MAGIC )
# MAGIC ```
# MAGIC - **Index**: Points to a pre-built vector index containing Foodly company documents
# MAGIC - **Retrieval**: Returns top 2 most relevant document chunks
# MAGIC - **Scope**: Searches policies, refund rules, delivery guidelines, etc.
# MAGIC - **Smart description**: Helps LLM understand when to use this tool
# MAGIC
# MAGIC ## 🤖 **LLM + Tool Integration**
# MAGIC ```python
# MAGIC llm_with_tools = llm.bind_tools([vs_tool])  # LLM can now search documents
# MAGIC ```
# MAGIC The LLM gains the ability to search through company documentation when needed.
# MAGIC
# MAGIC ## 🕸️ **Graph Architecture**
# MAGIC
# MAGIC
# MAGIC - **llm node**: Processes user queries and decides if document search is needed
# MAGIC - **tools node**: Executes vector search to retrieve relevant documents  
# MAGIC - **Conditional routing**: Automatically searches docs when LLM determines it's necessary
# MAGIC
# MAGIC ## 🔄 **RAG Workflow**
# MAGIC 1. **User question**: e.g., *"What's the refund policy?"*
# MAGIC 2. **LLM analysis**: Determines this needs company policy information
# MAGIC 3. **Vector search**: Retrieves relevant policy documents from the index
# MAGIC 4. **Augmented response**: LLM answers using retrieved company documents
# MAGIC 5. **Accurate answer**: Response based on actual company policies, not general knowledge
# MAGIC
# MAGIC ## 🎯 **Key Benefits**
# MAGIC - **Up-to-date info**: Always uses current company documents
# MAGIC - **Accurate responses**: Grounded in actual company policies
# MAGIC - **Automatic retrieval**: LLM decides when to search documents
# MAGIC - **Scalable**: Works with large document collections
# MAGIC - **Enterprise-ready**: Uses Databricks' managed vector search
# MAGIC
# MAGIC This pattern enables AI assistants to provide accurate, company-specific answers by automatically retrieving relevant documentation when needed.
# MAGIC

# COMMAND ----------

from dotenv import load_dotenv
import os
from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt.tool_node import ToolNode, tools_condition



from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool
)



# Initialize LLM
llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")



def call_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}


# Initialize the retriever tool.
vs_tool = VectorSearchRetrieverTool(
  index_name="agents.main.foodly_index_name_test",
  tool_name="foodly_policy_document_retrieval_tool",
  num_results=2,
  tool_description="Use this tool to search the Foodly knowledge base for policies, procedures, and service-related information. It retrieves the most relevant chunks from the company’s official documentation, including refund rules, cancellation terms, delivery guidelines, loyalty program details, privacy policies, and escalation procedures"
)


llm_with_tools = llm.bind_tools([vs_tool])

builder = StateGraph(MessagesState)

builder.add_node("llm",call_llm)
builder.add_node("tools",ToolNode([vs_tool]))


builder.add_edge(START,"llm")
builder.add_conditional_edges("llm" , tools_condition)
builder.add_edge("tools","llm")


agent = builder.compile()



# COMMAND ----------

messages = agent.invoke({"messages": [HumanMessage("What are refund timelines for Foodly?")]})

last_message = messages["messages"][-1].content
print(last_message)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Foodly Refund Policy – at a glance**
# MAGIC
# MAGIC | Situation | What you get | How it’s handled |
# MAGIC |-----------|--------------|------------------|
# MAGIC | **Order not delivered** (e.g., restaurant closed, no delivery partner available, extreme weather, system outage) | Full refund | Returned to the original payment method **or** issued instantly as Foodly credit. |
# MAGIC | **Incorrect or missing items** (wrong dish, missing side, wrong quantity) | Partial refund or credit equal to the value of the missing/incorrect items | Refund is processed once the issue is verified (usually within 24 h). |
# MAGIC | **Food‑quality problems** (spoiled, unsafe, contaminated) | Full refund **or** a free re‑delivery (if the restaurant can remake the order) | You can choose the preferred resolution; the refund/credit is applied immediately after approval. |
# MAGIC | **Duplicate payment** (accidental double charge) | Refund of the extra charge(s) | Processed automatically once the duplicate is detected or reported. |
# MAGIC | **Other eligible cases** (e.g., order cancelled by Foodly because a partner canceled) | Full refund | Same as above. |
# MAGIC
# MAGIC ### Non‑Refundable Cases  
# MAGIC Refunds are **not** issued for:
# MAGIC
# MAGIC * An incorrect address supplied by the customer.  
# MAGIC * The customer being unreachable at the time of delivery.  
# MAGIC * Personal taste preferences (e.g., “too spicy” or “too bland”).  
# MAGIC * Delays caused by external factors (traffic, weather, local restrictions) **unless** the delivery is **> 90 minutes** later than the ETA.
# MAGIC
# MAGIC ### Refund Timelines  
# MAGIC
# MAGIC | Refund type | Typical processing time |
# MAGIC |-------------|--------------------------|
# MAGIC | **Bank or card refunds** | 5–7 business days (depends on the card issuer). |
# MAGIC | **Foodly credits** | Instant – you see the credit in the app as soon as the refund is approved. |
# MAGIC | **Notification** | You receive an in‑app notification and an email confirming the refund. |
# MAGIC
# MAGIC ### How to Request a Refund  
# MAGIC
# MAGIC 1. **Open the order** in the Foodly app.  
# MAGIC 2. Tap **“Report an Issue”** and select the appropriate reason (e.g., “Missing items,” “Food quality,” “Did not receive order,” etc.).  
# MAGIC 3. Provide any relevant details or photos (especially for quality issues).  
# MAGIC 4. Submit – Foodly’s support team will review the case, usually within a few hours, and will either:  
# MAGIC    * Approve a full/partial refund, or  
# MAGIC    * Offer a Foodly credit or re‑delivery.  
# MAGIC
# MAGIC If you don’t see a resolution within 24 hours, you can follow up via the **Help Center** or contact **Live Chat** for escalation.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Bottom line:** You’re eligible for a refund (or credit) when Foodly is at fault—undelivered orders, missing/incorrect items, unsafe food, or duplicate charges. Personal preference issues and address errors are excluded. Refunds to cards take a few business days; Foodly credits are immediate. If you need help, start the “Report an Issue” flow in the app.

# COMMAND ----------

messages = agent.invoke({"messages": [HumanMessage("What are refund timelines for Foodly?")]})

last_message = messages["messages"][-1].content


import markdown

# Convert markdown → HTML
html = markdown.markdown(last_message, extensions=["tables", "fenced_code"] )

# Render in the notebook automatically
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC

# COMMAND ----------

messages = agent.invoke({"messages": [HumanMessage("I bought it like 10 days ago , can i make the refund?")]})

last_message = messages["messages"][-1].content
print(last_message)

# COMMAND ----------

messages = agent.invoke({"messages": [HumanMessage("check the order number 345555")]})

last_message = messages["messages"][-1].content
print(last_message)
