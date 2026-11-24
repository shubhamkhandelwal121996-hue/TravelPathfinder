import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient


# ==========================Environment setup===============================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UVX_COMMAND = shutil.which("uvx") or "uvx"


def _require_env(name: str, value: str | None) -> str:
    """Validate required environment variables."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """Pass necessary system variables and API credentials."""

    allowed_keys = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "UV_CACHE_DIR",
    }

    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed_keys
    }

    for key, value in updates.items():
        if value:
            env[key] = value

    return env


# =======================LLM==================================

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=_require_env("OPENAI_API_KEY", OPENAI_API_KEY),
    timeout=60,
    max_retries=2,
)


# ==========================MCP client==============================

client = MultiServerMCPClient(
    {
        # Remote Tavily MCP server
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },

        # Existing AviationStack MCP server
        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },

        # Existing OpenWeather MCP server
        "weather": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "openweather-mcp==0.1.0",
            ],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


# ==========================MCP tool discovery===============================

async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """Load a specific tool from a specific MCP server."""

    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "aviationstack":
        _require_env(
            "AVIATION_STACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )

    elif server_name == "weather":
        _require_env(
            "OPENWEATHER_API_KEY",
            OPENWEATHER_API_KEY,
        )

    if server_name in ("aviationstack", "weather"):
        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. "
                "Install uv and run `uvx --version`."
            )

    # Discover tools from only the requested server.
    tools = await client.get_tools(
        server_name=server_name,
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =======================MCP connection test==========================

async def get_all_tools() -> None:
    """Test all MCP servers independently."""

    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name,
            )

            tool_names = (
                ", ".join(tool.name for tool in tools)
                or "no tools"
            )

            print(
                f"{server_name}: OK -> {tool_names}"
            )

            # Display tool input schemas.
            for tool in tools:
                print(f"  {tool.name}: {tool.args}")

        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}"
            )


# =========================Tavily MCP=====================

async def tavily_mcp_search(query: str):
    """Search the web using the Tavily MCP server."""

    search_tool = await _get_server_tool(
        "tavily",
        "tavily_search",
    )

    return await search_tool.ainvoke(
        {
            "query": query,
        }
    )


# ============================AviationStack MCP=============

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    """Execute a tool from the AviationStack MCP server."""

    aviation_tool = await _get_server_tool(
        "aviationstack",
        tool_name,
    )

    return await aviation_tool.ainvoke(
        tool_args or {}
    )


# =========================OpenWeather MCP====================

async def weather_mcp_search(city: str):
    """Retrieve current weather information."""

    weather_tool = await _get_server_tool(
        "weather",
        "get_current_weather",
    )

    return await weather_tool.ainvoke(
        {
            "city": city,
        }
    )


async def forecast_mcp_search(city: str):
    """Retrieve weather forecast information."""

    forecast_tool = await _get_server_tool(
        "weather",
        "get_weather_forecast",
    )

    return await forecast_tool.ainvoke(
        {
            "city": city,
        }
    )


# ============================Destination extractor================

def extract_destination(query: str) -> str:
    """Extract the destination using the LLM."""

    prompt = f"""
Extract only the destination city or country
from the travel request.

Travel request:
{query}

Return only the destination name.
Do not add any explanation.
"""

    response = llm.invoke(prompt)

    destination = str(
        response.content
    ).strip()

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination


# =========================Test MCP servers==================

if __name__ == "__main__":
    asyncio.run(get_all_tools())