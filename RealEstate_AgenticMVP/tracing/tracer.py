"""
OpenTelemetry tracing setup — exports spans to Google Cloud Trace.

Every agent node and tool call gets a span automatically when you wrap
function calls with the tracer. The trace_id ties all spans in a session together
so you can follow a conversation end-to-end in Cloud Trace.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from config.settings import settings

# Set up once at application startup
def setup_tracing():
    exporter = CloudTraceSpanExporter(project_id=settings.gcp_project_id)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer("leasing-agent")


def trace_agent(agent_name: str, session_id: str, pipeline_stage: str):
    """
    Context manager — wraps an agent node call in a named span.

    Usage:
        with trace_agent("lead_intake", session_id, "intake"):
            result = await lead_intake_node(state)
    """
    return tracer.start_as_current_span(
        name=f"agent.{agent_name}",
        attributes={
            "session.id": session_id,
            "pipeline.stage": pipeline_stage,
            "agent.name": agent_name,
        },
    )


def trace_tool(tool_name: str, session_id: str):
    """Context manager for individual tool calls within an agent."""
    return tracer.start_as_current_span(
        name=f"tool.{tool_name}",
        attributes={
            "session.id": session_id,
            "tool.name": tool_name,
        },
    )
