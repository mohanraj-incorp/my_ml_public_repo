"""
CLI entry point for local testing — runs the leasing agent in the terminal.
For the full UI experience, run: uvicorn ui.api:app --reload

Usage: python main.py
"""
import asyncio
import uuid
from rich.console import Console
from rich.panel import Panel

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage

from graph.graph import build_graph
from graph.state import AgentState
from tracing.tracer import setup_tracing
from config.settings import settings

console = Console()
setup_tracing()


async def run_cli():
    session_id = str(uuid.uuid4())
    console.print(Panel(
        f"[bold]RealPage Leasing Assistant[/bold]\nSession: {session_id}\nType 'quit' to exit.",
        style="blue"
    ))

    async with await AsyncPostgresSaver.from_conn_string(
        f"postgresql://{settings.db_user}:{settings.db_password}@/{settings.db_name}"
        f"?host=/cloudsql/{settings.cloud_sql_instance}"
    ) as checkpointer:
        graph = build_graph(checkpointer)

        console.print("\n[bold green]Agent:[/bold green] Hi! I'm your leasing assistant. What are you looking for?\n")

        while True:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": settings.max_recursion_limit,
            }

            state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "session_id": session_id,
                "pipeline_stage": "intake",
                "visit_counts": {},
            }

            final_state = await graph.ainvoke(state, config=config)

            # Print the agent's last response
            ai_messages = [m for m in final_state.get("messages", []) if m.__class__.__name__ == "AIMessage"]
            if ai_messages:
                reply = ai_messages[-1].content
                stage = final_state.get("pipeline_stage", "")
                console.print(f"\n[bold green]Agent[/bold green] [dim]({stage})[/dim]: {reply}\n")

            # Show a warning when the frontend would display the secure SSN form
            if final_state.get("needs_pii_collection") == "ssn":
                console.print("[yellow]⚠ In the web UI, a secure SSN form would appear here.[/yellow]")
                ssn_sim = console.input("[dim]Simulated SSN token (press Enter to use mock):[/dim] ").strip()
                if not ssn_sim:
                    ssn_sim = "tok_ssn_mock_12345"
                console.print(f"[dim]Using token: {ssn_sim}[/dim]")
                # Re-invoke with the SSN token as a system message
                from langchain_core.messages import SystemMessage
                state_with_token: AgentState = {
                    "messages": [SystemMessage(content=f"SSN_TOKEN={ssn_sim}"), HumanMessage(content="I have submitted my SSN.")],
                    "session_id": session_id,
                    "pipeline_stage": "qualification",
                    "visit_counts": final_state.get("visit_counts", {}),
                    "ssn_token": ssn_sim,
                }
                final_state = await graph.ainvoke(state_with_token, config=config)
                ai_messages = [m for m in final_state.get("messages", []) if m.__class__.__name__ == "AIMessage"]
                if ai_messages:
                    console.print(f"\n[bold green]Agent:[/bold green] {ai_messages[-1].content}\n")

            # Print final decision prominently
            if final_state.get("decision"):
                decision = final_state["decision"].upper().replace("_", " ")
                color = "green" if "APPROVE" in decision else ("red" if "DENY" in decision else "yellow")
                console.print(Panel(f"[bold]DECISION: {decision}[/bold]", style=color))
                break


if __name__ == "__main__":
    asyncio.run(run_cli())
