"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "api_mode": "chat_completion",
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_tldr_returns_response(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_response_mode_maps_max_tokens(llm_params):
    from types import SimpleNamespace

    received_kwargs = {}

    def create_response(**kwargs):
        received_kwargs.update(kwargs)
        return SimpleNamespace(output_text="Summary")

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
    )
    llm_params["api_mode"] = "response"
    paper = make_sample_paper()

    assert paper.generate_tldr(client, llm_params) == "Summary"
    assert received_kwargs["max_output_tokens"] == 16384
    assert "max_tokens" not in received_kwargs


def test_invalid_api_mode_falls_back_to_abstract(llm_params):
    llm_params["api_mode"] = "invalid"
    paper = make_sample_paper()

    assert paper.generate_tldr(make_stub_openai_client(), llm_params) == paper.abstract


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_affiliations_returns_parsed_list(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


# ---------------------------------------------------------------------------
# api_mode: claude_code (Claude Code CLI)
# ---------------------------------------------------------------------------


@pytest.fixture()
def claude_code_params():
    return {
        "api_mode": "claude_code",
        "language": "English",
        "generation_kwargs": {"model": "opus", "max_tokens": 16384},
        "claude_code": {"cli_path": "claude", "fallback_model": "sonnet", "effort": None, "timeout": 600},
    }


def _stub_claude_run(calls, stdout=None, returncode=0, stderr=""):
    """Return a subprocess.run replacement that records its calls and answers like `claude -p --output-format json`."""
    import json
    import subprocess

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if stdout is not None:
            out = stdout
        elif "affiliations" in kwargs["input"]:
            out = json.dumps({"type": "result", "is_error": False, "result": '["TsingHua University","Peking University"]'})
        else:
            out = json.dumps({"type": "result", "is_error": False, "result": "Claude TLDR."})
        return subprocess.CompletedProcess(cmd, returncode, stdout=out, stderr=stderr)

    return _run


def test_claude_code_tldr(claude_code_params, monkeypatch):
    calls = []
    monkeypatch.setattr("zotero_arxiv_daily.protocol.subprocess.run", _stub_claude_run(calls))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")
    paper = make_sample_paper()

    assert paper.generate_tldr(None, claude_code_params) == "Claude TLDR."

    (cmd, kwargs), = calls
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--fallback-model") + 1] == "sonnet"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--effort" not in cmd
    assert "summarizes scientific paper" in cmd[cmd.index("--system-prompt") + 1]
    assert paper.title in kwargs["input"]
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert kwargs["timeout"] == 600


def test_claude_code_affiliations(claude_code_params, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.protocol.subprocess.run", _stub_claude_run([]))
    paper = make_sample_paper()
    assert sorted(paper.generate_affiliations(None, claude_code_params)) == ["Peking University", "TsingHua University"]


def test_claude_code_error_result_falls_back_to_abstract(claude_code_params, monkeypatch):
    import json

    error = json.dumps({"type": "result", "is_error": True, "result": "Claude usage limit reached"})
    monkeypatch.setattr("zotero_arxiv_daily.protocol.subprocess.run", _stub_claude_run([], stdout=error, returncode=1))
    paper = make_sample_paper()
    assert paper.generate_tldr(None, claude_code_params) == paper.abstract


def test_claude_code_non_json_output_falls_back_to_abstract(claude_code_params, monkeypatch):
    monkeypatch.setattr(
        "zotero_arxiv_daily.protocol.subprocess.run",
        _stub_claude_run([], stdout="", returncode=1, stderr="Invalid API key"),
    )
    paper = make_sample_paper()
    assert paper.generate_tldr(None, claude_code_params) == paper.abstract
