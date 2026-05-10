from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from graph.state import AgentState
from agents.supervisor import supervisor_node
from agents.lead_intake import lead_intake_node
from agents.property_search import property_search_node
from agents.policy_faq import policy_faq_node
from agents.qualification import qualification_node
from agents.scheduling import scheduling_node
from agents.application import application_node
from agents.decision import decision_node


# ── Routing function ──────────────────────────────────────────────────────────
# Called after every supervisor run. Returns the name of the next node.
def route_from_supervisor(state: AgentState) -> str:
    if state.get("human_escalation_required"):
        return END

    stage = state.get("pipeline_stage", "intake")

    routing_map = {
        "intake":         "lead_intake",
        "search":         "property_search",
        "policy":         "policy_faq",
        "qualification":  "qualification",
        "scheduling":     "scheduling",
        "application":    "application",
        "decision":       "decision",
        "complete":       END,
        "escalated":      END,
    }

    return routing_map.get(stage, "lead_intake")


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph(checkpointer: AsyncPostgresSaver):
    graph = StateGraph(AgentState)

    # Register every node
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("lead_intake",      lead_intake_node)
    graph.add_node("property_search",  property_search_node)
    graph.add_node("policy_faq",       policy_faq_node)
    graph.add_node("qualification",    qualification_node)
    graph.add_node("scheduling",       scheduling_node)
    graph.add_node("application",      application_node)
    graph.add_node("decision",         decision_node)

    # Every conversation starts at the supervisor
    graph.set_entry_point("supervisor")

    # Supervisor uses conditional routing to pick the next agent
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "lead_intake":     "lead_intake",
            "property_search": "property_search",
            "policy_faq":      "policy_faq",
            "qualification":   "qualification",
            "scheduling":      "scheduling",
            "application":     "application",
            "decision":        "decision",
            END:               END,
        },
    )

    # After each specialist agent finishes its turn, control returns to supervisor.
    # Supervisor then decides: stay in same stage, advance, or escalate.
    for agent_node in [
        "lead_intake", "property_search", "policy_faq",
        "qualification", "scheduling", "application",
    ]:
        graph.add_edge(agent_node, "supervisor")

    # Decision is a terminal node — once reached, the pipeline is complete
    graph.add_edge("decision", END)

    return graph.compile(
        checkpointer=checkpointer,
        # Interrupt before human-review step (conditional approval cases)
        interrupt_before=["decision"],
    )
