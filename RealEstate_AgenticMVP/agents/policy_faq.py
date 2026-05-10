"""
Policy FAQ agent — answers prospect questions about lease policies using Hybrid RAG.

Every answer MUST be grounded in retrieved document chunks.
If no relevant chunks are found, the agent says so rather than guessing.
The agent cites its source so the prospect knows where the information comes from.
"""
from anthropic import Anthropic
from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.policy_tools import search_policy_docs
from guardrails.output_guardrails import run_output_guardrails
from memory.short_term import get_context_messages
from config.settings import settings

client = Anthropic()

SYSTEM_PROMPT = """You are a leasing policy assistant.
Answer the prospect's question using ONLY the policy excerpts provided below.
Do not guess or infer anything not explicitly stated in the excerpts.
Always end your answer with: "Source: [document name]"
If the excerpts don't contain enough information, say: "I don't have the specific policy on file — a leasing agent will follow up."
"""


async def policy_faq_node(state: AgentState) -> dict:
    visit_counts = state.get("visit_counts", {})
    visit_counts["policy_faq"] = visit_counts.get("policy_faq", 0) + 1

    # Extract the prospect's question from the last message
    last_message = state["messages"][-1].content if state["messages"] else ""

    # Retrieve relevant policy chunks (Hybrid RAG)
    chunks = await search_policy_docs(
        query=last_message,
        property_id=state.get("selected_property_id"),
    )

    if not chunks:
        # No chunks found — safe fallback, don't let LLM guess
        return_stage = state.get("previous_stage", "search")
        return {
            "pipeline_stage": return_stage,
            "visit_counts": visit_counts,
            "messages": [AIMessage(
                content="I don't have the specific policy on file for this property. "
                        "A leasing agent will follow up with the details."
            )],
        }

    # Format retrieved chunks for the LLM
    context = "\n\n".join(
        f"[{c['source_file']} | {c['policy_type']}]\n{c['text']}"
        for c in chunks
    )

    context_messages = get_context_messages(state["messages"], state.get("conversation_summary"))
    anthropic_messages = [
        {"role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
         "content": m.content}
        for m in context_messages
        if m.__class__.__name__ in ("HumanMessage", "AIMessage")
    ]

    system = f"{SYSTEM_PROMPT}\n\nPolicy excerpts:\n{context}"

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        system=system,
        messages=anthropic_messages,
    )

    answer = next((b.text for b in response.content if hasattr(b, "text")), "")

    # Output guardrail — check for Fair Housing violations before returning to user
    _, is_safe, block_reason = run_output_guardrails(answer, "policy_faq", chunks)
    if not is_safe:
        answer = "I'm not able to answer that question. Please speak with a leasing agent directly."

    # Return to whichever stage the prospect was in before the policy question
    return_stage = state.get("previous_stage", "search")

    return {
        "pipeline_stage": return_stage,
        "visit_counts": visit_counts,
        "messages": [AIMessage(content=answer)],
    }
