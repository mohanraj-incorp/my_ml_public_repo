# Databricks notebook source
from dotenv import load_dotenv
import os
from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# Initialize LLM
llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")

def call_llm(state: MessagesState):
    """Node to call the LLM with a formatted prompt based on the latest user question."""
    
    
    # Build the prompt
    prompt_template = (
        "You are an expert AI assistant. Answer the user's question with clarity, accuracy, and conciseness.\n\n"
        "## Question:\n"
        "{question}\n\n"
        "## Guidelines:\n"
        "- Keep responses factual and to the point.\n"
        "- If relevant, provide examples or step-by-step instructions.\n"
        "- If the question is ambiguous, clarify before answering.\n\n"
        "Respond below:"
    )
    formatted_prompt = prompt_template.format(question=state['messages'])

    # Construct the message list for the model
    all_msgs = [
        SystemMessage(content="You are an expert AI assistant."),
        HumanMessage(content=formatted_prompt)
    ]

    # Call the LLM
    response = llm.invoke(all_msgs)

    return {"messages": state["messages"] + [response]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_llm", call_llm)

builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

graph = builder.compile()

# Example run
messages = graph.invoke({
    "messages": [HumanMessage(content="What is the return policy for Anirvan Decodes ecommerce products?")]
})


# COMMAND ----------

message = messages['messages'][-1]

for part in message.content:
    if part.get("type") == "text":
       ai_message = part.get("text", "")

print(ai_message)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## This is where we need RAG

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Why We Need RAG (Retrieval-Augmented Generation)
# MAGIC
# MAGIC ### 🧠 LLMs don’t know everything
# MAGIC - They’re trained on data up to a certain time.  
# MAGIC - They can’t magically know your company policies, recent data, or private documents.
# MAGIC
# MAGIC ### 🤔 LLMs can “hallucinate”
# MAGIC - Sometimes they just make stuff up — confidently!  
# MAGIC - Without grounding in real data, they may generate wrong or outdated answers.
# MAGIC
# MAGIC ### 🔄 Knowledge changes
# MAGIC - Laws, prices, product specs — they all change.  
# MAGIC - You don’t want to retrain or fine-tune a giant model every time something updates.
# MAGIC
# MAGIC ### 🏢 You need domain-specific answers
# MAGIC - Example: A bank chatbot answering questions about your bank’s loans — it needs **your** documents, not just generic financial knowledge.
# MAGIC
# MAGIC ### ⚡ Cheaper & faster than fine-tuning
# MAGIC - Instead of retraining an entire model, you just fetch relevant info (retrieval) and pass it along with the question.  
# MAGIC - That way, the model uses the right context instantly.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## What’s the problem with plain keyword search?
# MAGIC - Computers match **exact words**, not **meaning**.
# MAGIC - If a user types: “**laptop battery problem**”
# MAGIC   - Keyword search may **miss** a doc that says: “**notebook isn’t charging**”
# MAGIC   - Different words, **same idea** → missed!
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## What’s an embedding (in plain words)?
# MAGIC An **embedding** turns text into a list of numbers (a vector) that captures **meaning**.
# MAGIC - Items with **similar meaning** → vectors are **close** together.
# MAGIC - Items with **different meaning** → vectors are **far** apart.
# MAGIC
# MAGIC Think of a giant map:
# MAGIC - “laptop” and “notebook” are neighbors.
# MAGIC - “banana” is far away from both.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC | Sentence                 | X (Topic) | Y (Tone) | Z (Sentiment) |
# MAGIC | ------------------------ | --------- | -------- | ------------- |
# MAGIC | “How to bake a cake”     | 0.9       | 0.2      | 0.8           |
# MAGIC | “Cooking is fun”         | 0.8       | 0.1      | 0.9           |
# MAGIC | “AI models are powerful” | 0.2       | 0.8      | 0.6           |
# MAGIC | “I hate bad food”        | 0.7       | 0.3      | 0.1           |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ![formula.png](./formula.png "formula.png")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Example: “How to bake a cake” vs “Cooking is fun”
# MAGIC
# MAGIC Vector1 = [0.9, 0.2, 0.8]
# MAGIC Vector2 = [0.8, 0.1, 0.9]
# MAGIC
# MAGIC
# MAGIC Euclidean distance between this two vector is ≈ 0.173  → very small → very similar
# MAGIC  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: “How to bake a cake” vs “AI models are powerful”
# MAGIC
# MAGIC
# MAGIC Euclidean distance between this two vector is ≈ 0.943 → far → not similar
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Short explanation:
# MAGIC
# MAGIC “How to bake a cake” and “Cooking is fun” are close in topic, tone, and sentiment → match
# MAGIC
# MAGIC “How to bake a cake” and “AI models are powerful” are far apart in topic → don’t match

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## How embedding-based search works (step-by-step)
# MAGIC 1. **Prepare your data**
# MAGIC    - Split docs into **small chunks** (e.g., paragraphs).
# MAGIC 2. **Embed the chunks**
# MAGIC    - Convert each chunk into a **vector** (embedding).
# MAGIC    - Store vectors in a **vector database** (FAISS, Chroma, Milvus, etc.).
# MAGIC 3. **Handle a user query**
# MAGIC    - Convert the user’s query into a **vector**.
# MAGIC 4. **Find similar chunks**
# MAGIC    - Use **vector similarity** (e.g., cosine similarity) to find the **closest** vectors.
# MAGIC 5. **Return ranked results**
# MAGIC    - Show the top-K most similar chunks (or feed them to an LLM for a grounded answer).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛠️ How to Build a RAG System (Simple Terms)
# MAGIC
# MAGIC ### 1. Collect & prepare your knowledge
# MAGIC - Gather the stuff you want the AI to know: PDFs, web pages, database rows, manuals, FAQs, etc.  
# MAGIC - Break big documents into small, meaningful chunks (like splitting a book into paragraphs).
# MAGIC
# MAGIC ### 2. Index the knowledge so it’s searchable
# MAGIC - Turn those chunks into numbers (vectors) so a computer can “search by meaning” instead of just keywords.  
# MAGIC - Store them in a **vector database** (think: a smart search engine that understands concepts, not just exact words).
# MAGIC
# MAGIC ### 3. Connect retrieval to generation
# MAGIC - When someone asks a question:  
# MAGIC   - First, **search** the vector database for the most relevant chunks.  
# MAGIC   - Then, **feed** those chunks + the question into your LLM (like GPT).  
# MAGIC   - The LLM reads the context and gives a **grounded** answer (based on the actual data).
# MAGIC
