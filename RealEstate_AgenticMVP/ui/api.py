"""
FastAPI chat API — the bridge between the frontend and the LangGraph agent.

Endpoints:
  POST /chat          — send a user message, get an agent response
  GET  /session/{id}  — fetch current state for a session (for the UI to read flags)

The frontend watches the `needs_pii_collection` flag in the session state.
When it's set to "ssn", the UI replaces the chat input with a secure masked form.
"""
import uuid
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage

from graph.graph import build_graph
from graph.state import AgentState
from tracing.tracer import setup_tracing
from config.settings import settings

app = FastAPI(title="Leasing Agent API")
setup_tracing()

# Serve the chat UI at /
app.mount("/static", StaticFiles(directory="ui/static"), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse("ui/static/index.html")


class ChatRequest(BaseModel):
    session_id: str | None = None   # None on first message — we generate one
    message: str
    ssn_token: str | None = None    # Set by frontend after secure SSN collection


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    pipeline_stage: str
    needs_pii_collection: str | None   # "ssn" signals frontend to show secure form
    decision: str | None
    shortlisted_properties: list | None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # Build the graph with a Postgres checkpointer so state persists across turns
    async with await AsyncPostgresSaver.from_conn_string(
        f"postgresql://{settings.db_user}:{settings.db_password}@/{settings.db_name}"
        f"?host=/cloudsql/{settings.cloud_sql_instance}"
    ) as checkpointer:
        graph = build_graph(checkpointer)

        # Config ties this invocation to the right session checkpoint
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.max_recursion_limit,
        }

        # Build the input — include ssn_token as a system message if provided
        input_messages = [HumanMessage(content=request.message)]
        if request.ssn_token:
            from langchain_core.messages import SystemMessage
            input_messages.insert(0, SystemMessage(content=f"SSN_TOKEN={request.ssn_token}"))

        initial_state: AgentState = {
            "messages": input_messages,
            "session_id": session_id,
            "pipeline_stage": "intake",
            "visit_counts": {},
        }

        final_state = await graph.ainvoke(initial_state, config=config)

    # Extract the last AI message as the reply
    ai_messages = [m for m in final_state.get("messages", []) if m.__class__.__name__ == "AIMessage"]
    reply = ai_messages[-1].content if ai_messages else "I'm here to help. How can I assist you today?"

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        pipeline_stage=final_state.get("pipeline_stage", "intake"),
        needs_pii_collection=final_state.get("needs_pii_collection"),
        decision=final_state.get("decision"),
        shortlisted_properties=final_state.get("shortlisted_properties"),
    )


@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    """Returns key state fields for the frontend to read — not full message history."""
    async with await AsyncPostgresSaver.from_conn_string(
        f"postgresql://{settings.db_user}:{settings.db_password}@/{settings.db_name}"
        f"?host=/cloudsql/{settings.cloud_sql_instance}"
    ) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)

    if not state:
        return {"error": "Session not found"}

    values = state.values
    return {
        "session_id":             session_id,
        "pipeline_stage":         values.get("pipeline_stage"),
        "prospect_name":          values.get("prospect_name"),
        "selected_property_id":   values.get("selected_property_id"),
        "tour_booked":            values.get("tour_booked"),
        "application_submitted":  values.get("application_submitted"),
        "decision":               values.get("decision"),
        "needs_pii_collection":   values.get("needs_pii_collection"),
    }
