import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import openai_remote_mcp_bridge as bridge

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "arguments, errors",
    [
        ({"query": "mcp"}, []),
        ({}, ["missing required argument(s): query"]),
        ({"qery": "mcp"}, ["missing required argument(s): query", "unknown argument(s): qery"]),
    ],
)
def test_argument_errors(arguments, errors):
    assert bridge.argument_errors(arguments, SEARCH_SCHEMA) == errors


def test_extra_keys_are_allowed_unless_the_schema_forbids_them():
    schema = {"type": "object", "properties": {"query": {}}, "required": ["query"]}

    assert bridge.argument_errors({"query": "mcp", "limit": 3}, schema) == []


def test_tool_discovery_stops_when_the_server_repeats_a_cursor():
    page = SimpleNamespace(tools=[SimpleNamespace(name="search")], nextCursor="again")
    session = SimpleNamespace(list_tools=lambda cursor: asyncio.sleep(0, result=page))

    with pytest.raises(ValueError, match="repeated the cursor"):
        asyncio.run(bridge.list_all_tools(session))


def test_tool_discovery_follows_distinct_cursors():
    pages = {
        None: SimpleNamespace(tools=["a"], nextCursor="p2"),
        "p2": SimpleNamespace(tools=["b"], nextCursor=None),
    }
    session = SimpleNamespace(list_tools=lambda cursor: asyncio.sleep(0, result=pages[cursor]))

    assert asyncio.run(bridge.list_all_tools(session)) == ["a", "b"]


def test_an_empty_call_is_reported_to_the_model_and_not_sent(monkeypatch):
    calls, requests = [], []
    tool_call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="search", arguments=""))
    replies = iter(
        [
            SimpleNamespace(content=None, tool_calls=[tool_call]),
            SimpleNamespace(content="done", tool_calls=None),
        ]
    )

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            pass

        async def list_tools(self, cursor=None):
            tool = SimpleNamespace(name="search", description="", inputSchema=SEARCH_SCHEMA)
            return SimpleNamespace(tools=[tool], nextCursor=None)

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))

    class Completions:
        async def create(self, **request):
            requests.append([dict(message) for message in request["messages"]])
            message = next(replies)
            message.model_dump = lambda exclude_none: {"role": "assistant"}
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Client:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    @asynccontextmanager
    async def transport(url):
        yield None, None, None

    monkeypatch.setattr(bridge, "streamable_http_client", transport)
    monkeypatch.setattr(bridge, "ClientSession", Session)
    monkeypatch.setattr(bridge, "AsyncOpenAI", Client)

    answer = asyncio.run(bridge.run_bridge("find mcp", "key", "model", "http://mcp", None, 6))

    assert answer == "done"
    assert calls == []
    assert requests[1][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "Error: invalid tool arguments: missing required argument(s): query",
    }
