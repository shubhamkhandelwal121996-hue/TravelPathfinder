import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict
from urllib.parse import urlparse

import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from .Mcp_client import (
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    tavily_mcp_search,
    weather_mcp_search,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
APP_NAME = os.getenv("APP_NAME", "TripMate AI")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
MAX_CONTEXT_CHARS = 6000            # cap on each tool result copied into later prompts
AVIATION_CACHE_SECONDS = 6 * 3600   # airport/airline lists barely change
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, timeout=60, max_retries=2)


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    # Hosted databases need SSL; a local development database usually has none.
    host = urlparse(database_url).hostname or ""
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    if "sslmode=" not in database_url and not is_local:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


# --------------------------------------------------------------------------- #
# Structured output: the supervisor returns validated objects, not free-form JSON
# --------------------------------------------------------------------------- #
AgentName = Literal[
    "flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"
]


class GuardrailDecision(BaseModel):
    """Input guardrail verdict for one user request."""

    allowed: bool = Field(
        description="True if the request is about travel planning or travel information."
    )
    reason: str = Field(
        description="One short, polite sentence explaining a block. Empty string when allowed."
    )


class TripConstraints(BaseModel):
    destination: str = Field(description="Destination city or country, or '' if not stated.")
    origin: str = Field(description="Departure city or country, or '' if not stated.")
    duration: str = Field(description="Trip length such as '5 days', or '' if not stated.")
    budget: str = Field(description="Budget as the user stated it, or '' if not stated.")
    travel_style: str = Field(description="For example budget, luxury, family, adventure; '' if not stated.")
    special_preferences: list[str] = Field(
        description="Specific wishes such as vegetarian food or beach access. Empty list if none."
    )


class SupervisorDecision(BaseModel):
    """Guardrail verdict and routing plan, produced in a single model call."""

    guardrail: GuardrailDecision
    selected_agents: list[AgentName] = Field(
        description="Only the specialist agents this request needs. Empty if the request is blocked."
    )
    trip_constraints: TripConstraints
    reasoning: str = Field(description="One or two sentences explaining the agent selection.")


supervisor_llm = llm.with_structured_output(SupervisorDecision)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def _add_calls(current: int | None, update: int | None) -> int:
    """Counter reducer so parallel agents can each add their own LLM calls.

    A negative update resets the counter. A new request on a reused thread
    sends -1, so the count never carries over from the previous request.
    """
    if update is not None and update < 0:
        return 0
    return (current or 0) + (update or 0)


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str

    # Supervisor + guardrail
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    # Human-in-the-loop
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: Annotated[int, _add_calls]


AGENT_ORDER = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"]
RESEARCH_AGENTS = ("flight_agent", "hotel_agent", "weather_agent")  # independent, so parallel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ask(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return str(response.content)


def _clip(text: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    """Tool results can be huge and get copied into three later prompts."""
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _constraints_text(state: TravelState) -> str:
    constraints = {k: v for k, v in (state.get("trip_constraints") or {}).items() if v}
    return json.dumps(constraints) if constraints else "(none extracted)"


_SECTION_TITLES = {
    "flight_results": "Flight Results",
    "hotel_results": "Hotel Results",
    "weather_results": "Weather Results",
    "budget_results": "Budget Results",
    "itinerary": "Draft Itinerary",
}


def _context(state: TravelState, sections: tuple[str, ...]) -> str:
    """User request, constraints, and the requested result sections (empty ones skipped)."""
    parts = [
        f"User Request:\n{state['user_query']}",
        f"Trip Constraints:\n{_constraints_text(state)}",
    ]
    for key in sections:
        value = _clip(state.get(key))
        if value.strip():
            parts.append(f"{_SECTION_TITLES[key]}:\n{value}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Supervisor: guardrail + routing in one structured call
# --------------------------------------------------------------------------- #
SUPERVISOR_PROMPT = """You are the supervisor of a multi-agent travel-planning system. Do two things.

1. Guardrail. Set guardrail.allowed to true if the request belongs to travel planning or
travel information: destinations, flights, hotels, weather, budgets, visas, transportation,
sightseeing, food, packing, or itineraries. Block clearly unrelated requests and requests for
harmful or illegal instructions. Do not block a valid travel request just because some details
are missing. When you block, give a short, polite reason; otherwise leave the reason empty.

2. Routing. Choose only the specialist agents the request needs:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: builds the integrated plan; always include it
Also extract the trip constraints and explain your choice briefly.
If the request is blocked, select no agents and leave the constraints empty."""


def supervisor_agent(state: TravelState):
    decision: SupervisorDecision | None = None
    try:
        decision = supervisor_llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=state["user_query"]),
            ]
        )
        calls = 1
    except Exception as exc:
        # Fail open, as before: a temporary model or format error must not block travel requests.
        logger.warning("Supervisor fallback used: %s", exc)
        calls = 0

    if decision is not None and not decision.guardrail.allowed:
        reason = decision.guardrail.reason.strip() or (
            f"{APP_NAME} can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": TripConstraints(
                destination="", origin="", duration="", budget="",
                travel_style="", special_preferences=[],
            ).model_dump(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": calls,
        }

    if decision is None:
        selected = AGENT_ORDER.copy()
        constraints: dict[str, Any] = {}
        reasoning = (
            "Supervisor call failed, so the full travel workflow was selected as a safe fallback."
        )
        guardrail_reason = "Guardrail validation fallback allowed the request."
    else:
        requested = set(decision.selected_agents)
        selected = [name for name in AGENT_ORDER if name in requested]
        if "itinerary_agent" not in selected:  # it integrates whatever the others produced
            selected.append("itinerary_agent")
        constraints = decision.trip_constraints.model_dump()
        reasoning = decision.reasoning.strip()
        guardrail_reason = decision.guardrail.reason.strip()

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "llm_calls": calls,
    }


def guardrail_blocked_agent(state: TravelState):
    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the travel input guardrail."
    )
    return {"final_response": reason, "messages": [AIMessage(content=reason)]}


# --------------------------------------------------------------------------- #
# Flight agent
# --------------------------------------------------------------------------- #
FLIGHT_AGENT_PROMPT = """You are a travel flight expert.

User Query:
{query}

Trip Constraints:
{constraints}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

_aviation_lock = threading.Lock()
_aviation_cache: dict[str, Any] = {"at": 0.0, "data": None}


async def _fetch_aviation_reference() -> tuple[str, str]:
    # The two lookups are independent: run them together in one event loop.
    airports, airlines = await asyncio.gather(
        aviation_mcp_call("list_airports"), aviation_mcp_call("list_airlines")
    )
    return str(airports)[:3000], str(airlines)[:3000]


def _aviation_reference() -> tuple[str, str]:
    """Airport and airline lists, cached so each trip request does not re-launch the MCP server."""
    with _aviation_lock:
        fresh = (
            _aviation_cache["data"] is not None
            and time.monotonic() - _aviation_cache["at"] < AVIATION_CACHE_SECONDS
        )
        if not fresh:
            _aviation_cache["data"] = asyncio.run(_fetch_aviation_reference())
            _aviation_cache["at"] = time.monotonic()
        return _aviation_cache["data"]


def flight_agent(state: TravelState):
    try:
        airports, airlines = _aviation_reference()
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=state["user_query"],
            constraints=_constraints_text(state),
            airport_data=airports,
            airline_data=airlines,
        )
        flight_data = _ask("You are an expert travel flight planner.", prompt)
        calls = 1
    except Exception as exc:
        logger.warning("Flight agent error: %s: %s", type(exc).__name__, exc)
        flight_data = f"Flight information unavailable: {exc}"
        calls = 0
    return {"flight_results": flight_data, "llm_calls": calls}


# --------------------------------------------------------------------------- #
# Hotel agent
# --------------------------------------------------------------------------- #
def hotel_agent(state: TravelState):
    try:
        hotel_results = str(asyncio.run(tavily_mcp_search(f"Best hotels for {state['user_query']}")))
    except Exception as exc:
        logger.warning("Hotel agent MCP error: %s: %s", type(exc).__name__, exc)
        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )
    return {"hotel_results": hotel_results}  # no LLM call happens here


# --------------------------------------------------------------------------- #
# Weather agent
# --------------------------------------------------------------------------- #
async def _fetch_weather(city: str) -> tuple[Any, Any]:
    return await asyncio.gather(weather_mcp_search(city), forecast_mcp_search(city))


def weather_agent(state: TravelState):
    calls = 0
    # The supervisor already extracted the destination, so reuse it and skip an LLM call.
    city = ((state.get("trip_constraints") or {}).get("destination") or "").strip()
    if not city:
        try:
            city = extract_destination(state["user_query"])
            calls = 1
        except Exception as exc:
            logger.warning("Destination extraction failed: %s", exc)

    if not city:
        return {
            "weather_results": (
                "The destination could not be determined, so live weather is unavailable. "
                "Give general seasonal guidance and advise the traveler to verify the forecast "
                "before departure."
            ),
            "llm_calls": calls,
        }

    try:
        weather_data, forecast_data = asyncio.run(_fetch_weather(city))
        weather_results = f"Current Weather:\n{weather_data}\n\nForecast:\n{forecast_data}\n"
    except Exception as exc:
        logger.warning("Weather agent MCP error: %s: %s", type(exc).__name__, exc)
        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )
    return {"weather_results": weather_results, "llm_calls": calls}


# --------------------------------------------------------------------------- #
# Budget, itinerary, approval, final
# --------------------------------------------------------------------------- #
def budget_agent(state: TravelState):
    prompt = (
        "Analyze whether this trip is realistic for the user's budget.\n\n"
        + _context(state, ("flight_results", "hotel_results", "weather_results"))
        + "\n\nReturn:\n"
        "1. Estimated cost categories\n"
        "2. Budget risk areas\n"
        "3. Money-saving suggestions\n"
        "4. Overall feasibility\n\n"
        "If exact live prices are unavailable, clearly label estimates as approximate."
    )
    return {
        "budget_results": _ask("You are a practical travel budget analyst.", prompt),
        "llm_calls": 1,
    }


def itinerary_agent(state: TravelState):
    prompt = (
        "Create a complete travel itinerary.\n\n"
        + _context(state, ("flight_results", "hotel_results", "weather_results", "budget_results"))
        + "\n\nMake the itinerary practical, budget-aware, and easy to follow.\n"
        "Create a clear draft that is ready for human review."
    )
    return {
        "itinerary": _ask("You are an expert travel planner.", prompt),
        "approval_request": (
            "Please review the generated draft itinerary. Approve it to create the "
            "final polished plan, or provide feedback for revision."
        ),
        "llm_calls": 1,
    }


def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {"approved": True, "feedback": "Optional revision feedback"},
        }
    )
    if not isinstance(review, dict):
        review = {"approved": bool(review)}

    return {
        "approved": bool(review.get("approved", False)),
        "human_feedback": str(review.get("feedback", "")).strip(),
    }


def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = "The user approved the draft. Preserve its decisions while polishing it."
    else:
        feedback = state.get("human_feedback", "") or "Improve the draft before finalizing it."
        review_instruction = f"The user requested a revision. Apply this feedback carefully:\n{feedback}"

    prompt = (
        "Generate the final travel response for the user.\n\n"
        f"Human Review:\n{review_instruction}\n\n"
        + _context(
            state,
            ("flight_results", "hotel_results", "weather_results", "budget_results", "itinerary"),
        )
        + """

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested."""
    )
    response = llm.invoke(
        [
            SystemMessage(content="You are a professional AI travel booking assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return {"final_response": response.content, "messages": [response], "llm_calls": 1}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> list[str]:
    """Fan out: every selected research agent starts at the same time."""
    if not state.get("guardrail_allowed", True):
        return ["guardrail_blocked"]

    selected = _selected_agents(state)
    research = [agent for agent in RESEARCH_AGENTS if agent in selected]
    if research:
        return research
    return ["budget_agent"] if "budget_agent" in selected else ["itinerary_agent"]


def route_after_research(state: TravelState) -> str:
    """Fan in: runs after each research agent; the next node starts once, after all have finished."""
    return "budget_agent" if "budget_agent" in _selected_agents(state) else "itinerary_agent"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {name: name for name in ("guardrail_blocked", *RESEARCH_AGENTS, "budget_agent", "itinerary_agent")},
)
for research_agent in RESEARCH_AGENTS:
    graph.add_conditional_edges(
        research_agent,
        route_after_research,
        {"budget_agent": "budget_agent", "itinerary_agent": "itinerary_agent"},
    )
graph.add_edge("budget_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


# --------------------------------------------------------------------------- #
# PostgreSQL checkpointer on a connection pool
# --------------------------------------------------------------------------- #
def _build_checkpointer() -> PostgresSaver:
    # One shared connection is not safe once agents run in parallel threads
    # and several requests arrive together; a pool hands each its own.
    pool = ConnectionPool(
        conninfo=get_database_url(),
        max_size=DB_POOL_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


checkpointer = _build_checkpointer()
travel_graph = graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# FastAPI-facing helpers (same names and response shape as before)
# --------------------------------------------------------------------------- #
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    payload = getattr(interrupts[0], "value", interrupts[0])
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    paused = _interrupt_payload(result)

    if paused:
        answer = paused.get("draft_itinerary") or result.get("itinerary", "")
        approval_request = paused.get("approval_request", "")
        itinerary = paused.get("draft_itinerary", "")
    else:
        answer = result.get("final_response") or last_message
        approval_request = result.get("approval_request", "")
        itinerary = result.get("itinerary", "")

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": paused is not None,
        "approval_request": approval_request,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": itinerary,
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def _initial_state(user_input: str) -> dict[str, Any]:
    """Fresh values for every per-request field, so a reused thread starts clean."""
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "trip_constraints": {},
        "supervisor_reasoning": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": -1,  # negative resets the counter (see _add_calls)
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    thread_id = thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(_initial_state(user_input), config=config)
    return _serialize_result(result, thread_id)


def resume_travel_agent(thread_id: str, approved: bool, feedback: str = ""):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback.strip()}),
        config=config,
    )
    return _serialize_result(result, thread_id)