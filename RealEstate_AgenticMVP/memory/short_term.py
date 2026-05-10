"""
Short-term memory — manages conversation history within a session.

As conversations grow, passing the full history to every agent wastes tokens
and can exceed context limits. We use a three-layer strategy:
  1. Recent turns verbatim   (always included)
  2. Rolling summary         (older turns compressed)
  3. Pinned key state fields (always in context regardless of history length)

The summarizer is called by the supervisor when message count exceeds the threshold.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from config.settings import settings

llm = ChatAnthropic(model=settings.anthropic_model, temperature=0)


async def maybe_summarize(messages: list[BaseMessage], existing_summary: str | None) -> str | None:
    """
    If the conversation has grown past the threshold, summarize older turns.
    Returns the new summary string, or None if no summarization was needed.
    """
    if len(messages) <= settings.conversation_summary_threshold:
        return None  # still within threshold, no action needed

    # Keep the last 6 messages verbatim — summarize everything before that
    messages_to_summarize = messages[:-6]

    context = ""
    if existing_summary:
        context = f"Previous summary:\n{existing_summary}\n\n"

    history_text = "\n".join(
        f"{m.__class__.__name__}: {m.content}" for m in messages_to_summarize
    )

    prompt = (
        f"{context}Summarize the following leasing conversation in 3-5 sentences. "
        f"Capture: what the prospect is looking for, which properties were shown, "
        f"and what stage the conversation reached.\n\n{history_text}"
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


def get_context_messages(
    messages: list[BaseMessage],
    conversation_summary: str | None,
    keep_recent: int = 6,
) -> list[BaseMessage]:
    """
    Returns the message list that should be sent to an agent LLM call.
    If there's a summary, it prepends it as a system message and only
    passes the most recent `keep_recent` turns.
    """
    if conversation_summary and len(messages) > keep_recent:
        summary_msg = SystemMessage(
            content=f"[Conversation so far]\n{conversation_summary}"
        )
        return [summary_msg] + messages[-keep_recent:]

    return messages
