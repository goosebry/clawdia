"""Tests for ella/tools/registry.py"""
import asyncio
import pytest
from ella.tools.registry import ToolRegistry, ella_tool, get_registry


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_registry() -> ToolRegistry:
    """Return a fresh, isolated registry for each test."""
    return ToolRegistry()


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_and_schema():
    reg = make_registry()

    def add_numbers(a: int, b: int) -> int:
        """a: first number. b: second number."""
        return a + b

    reg.register(add_numbers, "add_numbers", "Add two integers.")
    schemas = reg.get_schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["function"]["name"] == "add_numbers"
    assert schema["function"]["description"] == "Add two integers."
    params = schema["function"]["parameters"]["properties"]
    assert "a" in params
    assert "b" in params
    assert params["a"]["type"] == "integer"
    assert params["b"]["type"] == "integer"


def test_required_vs_optional_params():
    reg = make_registry()

    def greet(name: str, greeting: str = "Hello") -> str:
        """name: person's name. greeting: greeting word."""
        return f"{greeting}, {name}"

    reg.register(greet, "greet", "Greet someone.")
    schema = reg.get_schemas()[0]
    required = schema["function"]["parameters"]["required"]
    assert "name" in required
    assert "greeting" not in required


def test_execute_sync():
    reg = make_registry()

    def double(x: int) -> int:
        """x: number to double."""
        return x * 2

    reg.register(double, "double", "Double a number.")
    result = asyncio.run(reg.execute("double", {"x": 7}))
    assert result == 14


def test_execute_async_fn():
    reg = make_registry()

    async def async_echo(msg: str) -> str:
        """msg: message to echo."""
        return f"echo: {msg}"

    reg.register(async_echo, "async_echo", "Echo a message asynchronously.")
    result = asyncio.run(reg.execute("async_echo", {"msg": "hello"}))
    assert result == "echo: hello"


def test_execute_unknown_tool():
    reg = make_registry()
    result = asyncio.run(reg.execute("nonexistent", {}))
    assert "not found" in result.lower()


def test_execute_raises_returns_error_string():
    reg = make_registry()

    def explode(x: str) -> str:
        """x: input."""
        raise ValueError("boom")

    reg.register(explode, "explode", "Always raises.")
    result = asyncio.run(reg.execute("explode", {"x": "anything"}))
    assert "Error" in result
    assert "explode" in result


def test_register_overwrites_same_name():
    reg = make_registry()

    def v1(x: str) -> str:
        """x: input."""
        return "v1"

    def v2(x: str) -> str:
        """x: input."""
        return "v2"

    reg.register(v1, "my_tool", "Version 1.")
    reg.register(v2, "my_tool", "Version 2.")
    result = asyncio.run(reg.execute("my_tool", {"x": "test"}))
    assert result == "v2"


def test_get_schemas_is_live_snapshot():
    reg = make_registry()
    assert reg.get_schemas() == []

    def noop(x: str) -> str:
        """x: input."""
        return x

    reg.register(noop, "noop", "Does nothing.")
    assert len(reg.get_schemas()) == 1


def test_load_directory_imports_py_files(tmp_path):
    """Drop a .py file into a temp dir, load_directory should pick it up."""
    tool_file = tmp_path / "temp_tool.py"
    tool_file.write_text(
        "from ella.tools.registry import get_registry, ella_tool\n"
        "\n"
        "@ella_tool(name='temp_calc', description='Multiply x by 3.')\n"
        "def temp_calc(x: int) -> int:\n"
        "    '''x: integer input.'''\n"
        "    return x * 3\n"
    )

    reg = get_registry()
    original_count = len(reg.get_schemas())
    reg.load_directory(tmp_path)
    schemas_after = reg.get_schemas()
    names = [s["function"]["name"] for s in schemas_after]
    assert "temp_calc" in names

    # Execute it
    result = asyncio.run(reg.execute("temp_calc", {"x": 5}))
    assert result == 15

    # Cleanup: unregister so it doesn't pollute other tests
    reg._unregister_module("ella_custom.temp_tool")


# ── JSON schema generation ────────────────────────────────────────────────────

def test_schema_string_type():
    from ella.tools.registry import _build_json_schema

    def fn(city: str) -> str:
        """city: name of the city."""
        pass

    schema = _build_json_schema(fn, "fn", "test")
    assert schema["function"]["parameters"]["properties"]["city"]["type"] == "string"


def test_schema_bool_type():
    from ella.tools.registry import _build_json_schema

    def fn(flag: bool) -> str:
        """flag: boolean flag."""
        pass

    schema = _build_json_schema(fn, "fn", "test")
    assert schema["function"]["parameters"]["properties"]["flag"]["type"] == "boolean"


def test_schema_param_description_from_docstring():
    from ella.tools.registry import _build_json_schema

    def fn(query: str) -> str:
        """query: the search query string."""
        pass

    schema = _build_json_schema(fn, "fn", "test")
    desc = schema["function"]["parameters"]["properties"]["query"]["description"]
    assert "search query" in desc
