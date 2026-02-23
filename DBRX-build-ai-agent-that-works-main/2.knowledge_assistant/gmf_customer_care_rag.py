# Databricks notebook source
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DatabricksEmbeddings


# COMMAND ----------

# Initialize Databricks embedding model
embedding_model = DatabricksEmbeddings(
    endpoint="databricks-gte-large-en"
)


# COMMAND ----------

BASE_PATH = "/Workspace/Users/mohanraj.incorp@gmail.com/gmf_customercare_policy"
CATALOG_SCHEMA = "agents.main"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

pdf_files = [
    f for f in os.listdir(BASE_PATH)
    if f.lower().endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDF files.")

for pdf_file in pdf_files:
    pdf_path = os.path.join(BASE_PATH, pdf_file)
    print(f"\nProcessing: {pdf_file}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    docs = text_splitter.split_documents(documents)
    print(f"  → Created {len(docs)} chunks")

    # Generate embeddings (batch call)
    texts = [d.page_content for d in docs]
    embeddings = embedding_model.embed_documents(texts)

    # Prepare chunk records
    chunk_data = []
    for i, d in enumerate(docs):
        chunk_data.append({
            "chunk_id": i + 1,
            "content": d.page_content,
            "embedding": embeddings[i],              # VECTOR COLUMN
            "metadata": str(d.metadata)
        })

    from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    ArrayType,
    DoubleType
    )
    schema = StructType([
    StructField("chunk_id", LongType(), True),
    StructField("content", StringType(), True),
    StructField("embedding", ArrayType(DoubleType()), True),
    StructField("metadata", StringType(), True)
    ])
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(chunk_data,schema=schema)

    display(spark_df)

    # Sanitize table name
    base_table_name = (
        os.path.splitext(pdf_file)[0]
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    table_name = f"{CATALOG_SCHEMA}.{base_table_name}_chunks"

    # Replace table with embeddings
    spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

    spark.sql(f"""ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
    print(f"  ✅ Table replaced with embeddings: {table_name}")


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient


vsc = VectorSearchClient(disable_notice=True)

def index_exists(endpoint_name: str, index_name: str) -> bool:
    try:
        indexes = vsc.list_indexes(endpoint_name)
        return index_name in indexes
    except Exception as e:
        print(f"Warning while checking index existence {index_name}: {e}")
        return False


def create_vector_index_if_not_exists(index_name: str, source_table: str):
    endpoint_name = "agent-db"

    #if index_exists(endpoint_name, index_name):
    #    print(f"✅ Index already exists, skipping creation: {index_name}")
    #    return

    print(f"🚀 Creating vector index: {index_name}")

    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,               # Vector Search endpoint
        index_name=index_name,                     # UC-qualified index name
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_vector_column="embedding",       # precomputed vectors
        embedding_dimension=1024                   # databricks-gte-large-en
    )

    print(f"✅ Successfully created index: {index_name}")

create_vector_index_if_not_exists(
    index_name="agents.main.delinquency_policy_idx",
    source_table="agents.main.deliquency_management_policy_chunks"  # Ensure this table exists or handle the error.
)

create_vector_index_if_not_exists(
    index_name="agents.main.payoff_policy_idx",
    source_table="agents.main.earlypayoff_policy_chunks"
)

create_vector_index_if_not_exists(
    index_name="agents.main.lease_end_policy_idx",
    source_table="agents.main.leaseend_policy_chunks"
)



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
delinquency_tool = VectorSearchRetrieverTool(
    index_name="agents.main.delinquency_policy_idx",
    tool_name="GMF_delinquency_policy_tool",
    num_results=2,
    tool_description=(
        "Defines GM Financial’s policies and procedures for managing delinquent "
        "accounts, including outreach, hardship handling, escalation, and compliance."
    ),
    embedding=embedding_model,                    # ← this was missing
    text_column="content"   
)

early_payoff_tool = VectorSearchRetrieverTool(
    index_name="agents.main.payoff_policy_idx",
    tool_name="GMF_early_payoff_policy_tool",
    num_results=2,
    tool_description=(
        "Documents GM Financial’s early payoff policies including payoff quotes, "
        "interest calculations, fees, timing, and account closure."
    ),
    embedding=embedding_model,                    # ← this was missing
    text_column="content"   
)

lease_end_tool = VectorSearchRetrieverTool(
    index_name="agents.main.lease_end_policy_idx",
    tool_name="GMF_lease_end_policy_tool",
    num_results=2,
    tool_description=(
        "Outlines GM Financial’s lease-end processes including vehicle return options, "
        "inspections, excess wear and mileage, and final billing."
    ),
    embedding=embedding_model,                    # ← this was missing
    text_column="content"   
)


llm_with_tools = llm.bind_tools([delinquency_tool,early_payoff_tool,lease_end_tool])

builder = StateGraph(MessagesState)

builder.add_node("llm",call_llm)
builder.add_node("tools",ToolNode([delinquency_tool,early_payoff_tool,lease_end_tool]))


builder.add_edge(START,"llm")
builder.add_conditional_edges("llm" , tools_condition)
builder.add_edge("tools","llm")


agent = builder.compile()

# COMMAND ----------

messages = agent.invoke({"messages": [HumanMessage("When will someone be deliquent at GMF?")]})

last_message = messages["messages"][-1].content
print(last_message)

# COMMAND ----------

#With Memory implementation
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool
)


# ────────────────────────────────────────────────
# Configurable constants
# ────────────────────────────────────────────────
MAX_TOKENS_BEFORE_SUMMARY = 3000      # trigger summary when approaching this
KEEP_LAST_N_CONVERSATIONS = 3          # keep last 3 full turns (6 messages)
SUMMARY_PROMPT = """Summarize the following conversation history concisely, 
capturing key facts, user questions, decisions, and context relevant to GM Financial policies.
Do NOT include greetings or chit-chat. Keep under 400 tokens.

Conversation to summarize:
{conversation}"""

# Token counter (approximate is usually fine and faster)
token_counter = count_tokens_approximately  # or use exact tokenizer if available

# ────────────────────────────────────────────────
# Your LLM & tools (unchanged except binding)
# ────────────────────────────────────────────────

llm = ChatDatabricks(endpoint="databricks-gpt-oss-120b")

# ... your VectorSearchRetrieverTool definitions remain the same ...
# Initialize the retriever tool.
delinquency_tool = VectorSearchRetrieverTool(
    index_name="agents.main.delinquency_policy_idx",
    tool_name="GMF_delinquency_policy_tool",
    num_results=2,
    tool_description=(
        "Defines GM Financial’s policies and procedures for managing delinquent "
        "accounts, including outreach, hardship handling, escalation, and compliance."
    ),
    embedding=embedding_model,                    # ← this was missing
    text_column="content"   
)

early_payoff_tool = VectorSearchRetrieverTool(
    index_name="agents.main.payoff_policy_idx",
    tool_name="GMF_early_payoff_policy_tool",
    num_results=2,
    tool_description=(
        "Documents GM Financial’s early payoff policies including payoff quotes, "
        "interest calculations, fees, timing, and account closure."
    ),
    embedding=embedding_model,                    # ← this was missing
    text_column="content"   
)

lease_end_tool = VectorSearchRetrieverTool(
    index_name="agents.main.lease_end_policy_idx",
    tool_name="GMF_lease_end_policy_tool",
    num_results=2,
    tool_description=(
        "Outlines GM Financial’s lease-end processes including vehicle return options, "
        "inspections, excess wear and mileage, and final billing."
    ),
    embedding=embedding_model,                    # ← this was missing
    text_column="content"   
)


llm_with_tools = llm.bind_tools([delinquency_tool, early_payoff_tool, lease_end_tool])

# You should already have this somewhere – rough but fast token estimator
def token_counter(messages: List[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total += len(msg.content) // 4 + 10  # rough: ~4 chars/token + overhead
        elif isinstance(msg.content, list):
            # Handle structured content (tool calls, reasoning, etc.)
            for part in msg.content:
                if isinstance(part, dict):
                    text = part.get("text", "") or str(part)
                    total += len(text) // 4 + 5
                else:
                    total += len(str(part)) // 4 + 5
        else:
            total += len(str(msg.content)) // 4 + 10
    return total
# ────────────────────────────────────────────────
# Summarization helper
# ────────────────────────────────────────────────
def summarize_old_conversation(messages: List[BaseMessage]) -> str:
    # Only summarize messages before the last 6
    old_messages = messages[:-6]
    if not old_messages:
        return ""

    # Build readable text from old messages
    old_text_parts = []
    for msg in old_messages:
        if isinstance(msg.content, str):
            old_text_parts.append(f"{msg.type.upper()}: {msg.content}")
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and "text" in part:
                    old_text_parts.append(f"{msg.type.upper()}: {part['text']}")
                elif isinstance(part, dict):
                    old_text_parts.append(f"{msg.type.upper()}: {str(part)}")
        else:
            old_text_parts.append(f"{msg.type.upper()}: {str(msg.content)}")

    old_text = "\n".join(old_text_parts)
    if not old_text.strip():
        return ""

    summary_input = [HumanMessage(content=SUMMARY_PROMPT.format(conversation=old_text))]
    summary_response = llm.invoke(summary_input)  # using plain llm (no tools)
    
    if isinstance(summary_response.content, str):
        return summary_response.content.strip()
    elif isinstance(summary_response.content, list):
        # Extract text parts from structured output
        texts = [p.get("text", "") for p in summary_response.content if isinstance(p, dict)]
        return "\n".join(texts).strip()
    else:
        return str(summary_response.content).strip()

# ────────────────────────────────────────────────
# Main LLM node with smart memory management
# ────────────────────────────────────────────────
def call_llm(state: MessagesState) -> Dict[str, Any]:
    messages: List[BaseMessage] = state["messages"]

    # ── Safety: convert any list-content messages to string for sending ──
    messages_to_use = []
    for msg in messages:
        if isinstance(msg.content, list):
            # Flatten structured content to text (important for Databricks / many providers)
            text_parts = []
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text" or "text" in part:
                        text_parts.append(part.get("text", str(part)))
                    elif "tool_call" in part or "tool_use" in part:
                        # Skip tool calls themselves – they are handled via .tool_calls
                        continue
                    else:
                        text_parts.append(str(part))
                else:
                    text_parts.append(str(part))
            
            content_str = "\n".join(text_parts).strip()
            
            # Rebuild message with string content
            if isinstance(msg, AIMessage):
                new_msg = AIMessage(
                    content=content_str,
                    tool_calls=msg.tool_calls if hasattr(msg, "tool_calls") else None,
                    id=msg.id,
                    additional_kwargs=msg.additional_kwargs,
                )
            elif isinstance(msg, HumanMessage):
                new_msg = HumanMessage(content=content_str, id=msg.id)
            elif isinstance(msg, SystemMessage):
                new_msg = SystemMessage(content=content_str, id=msg.id)
            else:
                new_msg = msg.__class__(content=content_str, **msg.dict(exclude={"content"}))
                
            messages_to_use.append(new_msg)
        else:
            messages_to_use.append(msg)

    # ── Now decide whether to summarize or send full history ──
    current_tokens = token_counter(messages_to_use)

    if current_tokens <= MAX_TOKENS_BEFORE_SUMMARY:
        # Under limit → use as-is
        final_messages = messages_to_use
    else:
        # Over limit → summarize old part + keep last 6 messages
        recent = messages_to_use[-6:]
        old_summary_text = summarize_old_conversation(messages_to_use)

        if old_summary_text:
            summary_msg = AIMessage(
                content=f"Previous conversation summary:\n{old_summary_text}",
                id=f"summary-{hash(old_summary_text) % 1000000:06d}"
            )
            final_messages = [summary_msg] + recent
        else:
            final_messages = recent

        # Final safety trim (fallback)
        final_messages = trim_messages(
            final_messages,
            strategy="last",
            token_counter=token_counter,
            max_tokens=MAX_TOKENS_BEFORE_SUMMARY - 2000,
            start_on="human",
            allow_partial=False,
        )

    # ── Call the model with safe messages ──
    response = llm_with_tools.invoke(final_messages)

    return {"messages": [response]}
# ────────────────────────────────────────────────
# Graph setup (same as before + checkpointer)
# ────────────────────────────────────────────────

builder = StateGraph(MessagesState)

builder.add_node("llm", call_llm)
builder.add_node("tools",ToolNode([delinquency_tool,early_payoff_tool,lease_end_tool]))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

memory = MemorySaver()  # or SqliteSaver, PostgresSaver, etc. for prod
agent = builder.compile(checkpointer=memory)

# COMMAND ----------


#last_message = result["messages"][-1].content
#print(last_message)

def get_message_role(msg):
    """Safely extract role string even when msg.type is list"""
    t = msg.type if isinstance(msg.type, str) else ','.join(str(x) if isinstance(x, str) else x for x in msg.type)
    
    if isinstance(t, str):
        role_key = t
    elif isinstance(t, (list, tuple)):
        role_key = ','.join(str(x) for x in t)  # "ai,tool_calls" or similar
    else:
        role_key = str(t)
    
    role_map = {
        "human":                "USER",
        "ai":                   "ASSISTANT",
        "system":               "SYSTEM",
        "tool":                 "TOOL",
        "function":             "TOOL",
        "remove":               "REMOVE",
        "placeholder":          "PLACEHOLDER",
        "ai,tool_calls":        "ASSISTANT (tool call)",
        "human,tool":           "USER (tool result)",
        # add more compound types if you see them
    }
    
    return role_map.get(role_key, role_key.upper())


def print_memory_state(result):
    print("\n" + "="*80)
    print("CURRENT MEMORY STATE")
    print("="*80)
    
    messages = result["messages"]
    print(f"Total messages: {len(messages)}")
    print("-"*80)
    
    for i, msg in enumerate(messages):
        role = get_message_role(msg)
        
        # Handle different content types safely
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                preview = content[:200] + "…" if len(content) > 200 else content
            elif isinstance(content, list):     # tool calls, structured output, etc.
                preview = f"[list of {len(content)} items – e.g. tool calls]"
            else:
                preview = str(content)[:200] + "…"
        else:
            preview = "[no content]"
            
        is_summary = "Previous conversation summary" in preview
        
        print(f"[{i:2d}] {role:18}  {'→ SUMMARY' if is_summary else ''}")
        print(preview)
        print("-"*80)

# Usage:

# Turn 2, 3, ... → memory preserved
# After ~10+ turns → automatically summarizes older parts, keeps last 3 conversations full

# COMMAND ----------

config = {"configurable": {"thread_id": "customer_session_12345"}}

# Turn 1
result = agent.invoke({"messages": [HumanMessage("How many days before deliquency is declared ? ")]}, config)

#result = agent.invoke({"messages": [HumanMessage(content="test")]}, config)
print_memory_state(result)  # your debug printer

# COMMAND ----------

#Test code to chec CI process
def example():
    print("CI worked")
    return True

example()

# COMMAND ----------

#Test code to chec CI process
def example2():
    print("CI worked")
    return True

example2()
