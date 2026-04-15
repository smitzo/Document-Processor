"""
LangGraph Workflow
==================
Wires all nodes into the claim processing pipeline:

START
  └─► segregator_agent
        ├─► id_agent       ──┐
        ├─► discharge_agent──┤
        └─► bill_agent     ──┘
                              └─► aggregator ──► END

The three extraction agents run in parallel (fan-out → fan-in).
"""

from __future__ import annotations
import logging

from langgraph.graph import StateGraph, END

from app.core.schemas import ClaimState
from app.agents.segregator import segregator_agent
from app.agents.id_agent import id_agent
from app.agents.discharge_agent import discharge_agent
from app.agents.bill_agent import bill_agent
from app.agents.aggregator import aggregator

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and compile the LangGraph claim processing workflow."""

    logger.info("Building LangGraph claim processing workflow")
    graph = StateGraph(ClaimState)

    # -----------------------------------------------------------------------
    # Register nodes
    # -----------------------------------------------------------------------
    graph.add_node("segregator", segregator_agent)
    graph.add_node("id_agent", id_agent)
    graph.add_node("discharge_agent", discharge_agent)
    graph.add_node("bill_agent", bill_agent)
    graph.add_node("aggregator", aggregator)

    # -----------------------------------------------------------------------
    # Edges: START → segregator → (3 agents in parallel) → aggregator → END
    # -----------------------------------------------------------------------
    graph.set_entry_point("segregator")

    # Fan-out: segregator feeds all three agents
    graph.add_edge("segregator", "id_agent")
    graph.add_edge("segregator", "discharge_agent")
    graph.add_edge("segregator", "bill_agent")

    # Fan-in: all agents feed aggregator
    graph.add_edge("id_agent", "aggregator")
    graph.add_edge("discharge_agent", "aggregator")
    graph.add_edge("bill_agent", "aggregator")

    graph.add_edge("aggregator", END)

    return graph.compile()


# Module-level singleton — compiled once, reused across requests
claim_graph = build_graph()
