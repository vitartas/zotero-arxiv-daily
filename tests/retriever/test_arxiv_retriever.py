"""Tests for ArxivRetriever."""

import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import feedparser

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout, _submission_window
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Full text is only fetched after reranking, never while retrieving
    def _no_download(paper):
        raise AssertionError("retrieve_papers must not download full text")

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", _no_download)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", _no_download)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", _no_download)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert all(p.full_text is None for p in papers)


def test_fetch_full_text_falls_back_tar_html_pdf(config, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    tried = []

    def _extractor(name, result):
        def _extract(paper):
            tried.append(name)
            return result
        return _extract

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", _extractor("tar", None))
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", _extractor("html", None))
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", _extractor("pdf", "PDF text"))
    paper = make_sample_paper(full_text=None)

    ArxivRetriever(config).fetch_full_text(paper)

    assert tried == ["tar", "html", "pdf"]
    assert paper.full_text == "PDF text"


def test_fetch_full_text_stops_at_first_success(config, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: "TeX source")
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: 1 / 0)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: 1 / 0)
    paper = make_sample_paper(full_text=None)

    ArxivRetriever(config).fetch_full_text(paper)

    assert paper.full_text == "TeX source"


def test_extract_text_from_tar_uses_source_url(monkeypatch):
    calls = []
    monkeypatch.setattr(arxiv_retriever, "_run_with_hard_timeout", lambda func, args, **kw: calls.append(args) or "tex")
    paper = make_sample_paper(url="https://arxiv.org/abs/2610.00001", pdf_url="https://arxiv.org/pdf/2610.00001")

    assert arxiv_retriever.extract_text_from_tar(paper) == "tex"
    assert calls == [("https://arxiv.org/src/2610.00001", paper.url, paper.title)]


# ---------------------------------------------------------------------------
# Date-window mode (lookback_days): arXiv API instead of RSS
# ---------------------------------------------------------------------------


def test_submission_window_saturday_run():
    # A Saturday run with lag 2 covers first submissions from the previous Thursday up to this Thursday.
    start, end = _submission_window(date(2026, 10, 10), lookback_days=7, lag_days=2)
    assert start == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 8, tzinfo=timezone.utc)


def test_submission_window_sunday_run():
    # A Sunday run with lag 3 covers the same Thursday-to-Thursday window as a Saturday run with lag 2.
    start, end = _submission_window(date(2026, 10, 11), lookback_days=7, lag_days=3)
    assert start == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 8, tzinfo=timezone.utc)


def test_submission_windows_of_consecutive_runs_are_contiguous():
    _, end_previous = _submission_window(date(2026, 10, 3), lookback_days=7, lag_days=2)
    start_next, _ = _submission_window(date(2026, 10, 10), lookback_days=7, lag_days=2)
    assert end_previous == start_next


def _fake_api_result(paper_id: str, primary_category: str) -> SimpleNamespace:
    return SimpleNamespace(
        entry_id=f"http://arxiv.org/abs/{paper_id}v1",
        title=f"Paper {paper_id}",
        authors=[SimpleNamespace(name="Test Author")],
        summary="Test abstract",
        pdf_url=f"http://arxiv.org/pdf/{paper_id}v1",
        primary_category=primary_category,
    )


def _install_fake_api(monkeypatch, results_by_category):
    searches = []

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            searches.append(search)
            category = search.query.removeprefix("cat:").split(" ")[0]
            outcome = results_by_category[category]
            if isinstance(outcome, Exception):
                raise outcome
            return iter(outcome)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    return searches


def test_api_mode_queries_each_category_in_window(config, monkeypatch):
    config.source.arxiv.lookback_days = 7
    config.source.arxiv.lag_days = 2
    searches = _install_fake_api(monkeypatch, {
        "cs.AI": [_fake_api_result("2610.00001", "cs.AI"), _fake_api_result("2610.00002", "cs.CV")],
        "cs.CV": [_fake_api_result("2610.00002", "cs.CV"), _fake_api_result("2610.00003", "cs.CV")],
    })

    raw = ArxivRetriever(config)._retrieve_raw_papers()

    start, end = _submission_window(datetime.now(timezone.utc).date(), 7, 2)
    window = f"submittedDate:[{start:%Y%m%d}0000 TO {end - timedelta(days=1):%Y%m%d}2359]"
    assert [s.query for s in searches] == [f"cat:cs.AI AND {window}", f"cat:cs.CV AND {window}"]
    assert all(s.max_results is None for s in searches)
    # Deduplicated across categories
    assert [r.entry_id for r in raw] == [
        "http://arxiv.org/abs/2610.00001v1",
        "http://arxiv.org/abs/2610.00002v1",
        "http://arxiv.org/abs/2610.00003v1",
    ]


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]


def test_clean_abstract():
    raw_with_prefix = (
        "arXiv:2609.22090v1 Announce Type: new \n"
        "Abstract: An LLM producing the response pattern associated with a human psychological effect..."
    )
    assert arxiv_retriever._clean_abstract(raw_with_prefix) == (
        "An LLM producing the response pattern associated with a human psychological effect..."
    )

    raw_plain = "Simple abstract without prefix."
    assert arxiv_retriever._clean_abstract(raw_plain) == "Simple abstract without prefix."

    raw_with_html_and_inner_abstract = (
        "<p>arXiv:2609.22090v1 Announce Type: new \n"
        "Abstract: In this work, we introduce Abstract: A Benchmark for evaluation.</p>"
    )
    assert arxiv_retriever._clean_abstract(raw_with_html_and_inner_abstract) == (
        "In this work, we introduce Abstract: A Benchmark for evaluation."
    )

    raw_with_inequalities = (
        "<p>arXiv:2609.22090v1 Announce Type: new \n"
        "Abstract: We prove that x < y > 0 and 0 < a < b under condition C.</p>"
    )
    assert arxiv_retriever._clean_abstract(raw_with_inequalities) == (
        "We prove that x < y > 0 and 0 < a < b under condition C."
    )

    raw_with_br = (
        "<p>arXiv:2609.22090v1 Announce Type: new<br>Abstract: This is the body after br tags.</p>"
    )
    assert arxiv_retriever._clean_abstract(raw_with_br) == "This is the body after br tags."


def test_parse_entry_time():
    from datetime import datetime, timezone
    import time
    st = time.strptime("2026-09-23 12:30:45", "%Y-%m-%d %H:%M:%S")
    dt = arxiv_retriever._parse_entry_time(st)
    assert dt == datetime(2026, 9, 23, 12, 30, 45, tzinfo=timezone.utc)
    assert arxiv_retriever._parse_entry_time(None) == datetime.min.replace(tzinfo=timezone.utc)


def test_extract_authors():
    entry_with_author = SimpleNamespace(author="Alice, Bob, Charlie")
    authors = arxiv_retriever._extract_authors(entry_with_author)
    assert [a.name for a in authors] == ["Alice", "Bob", "Charlie"]

    entry_with_authors_list = SimpleNamespace(
        author="",
        authors=[{"name": "David"}, {"name": "Eva"}]
    )
    authors_from_list = arxiv_retriever._extract_authors(entry_with_authors_list)
    assert [a.name for a in authors_from_list] == ["David", "Eva"]

    entry_empty = SimpleNamespace(author="")
    authors_empty = arxiv_retriever._extract_authors(entry_empty)
    assert [a.name for a in authors_empty] == ["Unknown"]


def test_entry_to_arxiv_result():
    import time
    st = time.strptime("2026-09-23 12:00:00", "%Y-%m-%d %H:%M:%S")
    entry = SimpleNamespace(
        id="oai:arXiv.org:2609.22090v1",
        title="  Recognition, Simulation, \n and Refusal  ",
        author="Joy Bose",
        summary="arXiv:2609.22090v1 Announce Type: new \nAbstract: Test summary.",
        link="https://arxiv.org/abs/2609.22090",
        tags=[{"term": "cs.AI"}, {"term": "cs.CL"}],
        published_parsed=st,
        updated_parsed=st,
        arxiv_comment="10 pages",
        arxiv_journal_ref="Nature",
        arxiv_doi="10.1234/test",
    )
    result = arxiv_retriever._entry_to_arxiv_result(entry)
    assert result.title == "Recognition, Simulation, and Refusal"
    assert [a.name for a in result.authors] == ["Joy Bose"]
    assert result.summary == "Test summary."
    assert result.entry_id == "https://arxiv.org/abs/2609.22090"
    assert result.pdf_url == "https://arxiv.org/pdf/2609.22090v1"
    assert result.source_url() == "https://arxiv.org/src/2609.22090v1"
    assert result.get_short_id() == "2609.22090"
    assert result.primary_category == "cs.AI"
    assert result.categories == ["cs.AI", "cs.CL"]
    assert result.comment == "10 pages"
    assert result.journal_ref == "Nature"
    assert result.doi == "10.1234/test"
    assert result.published.year == 2026
    assert result.published.hour == 12
    assert result.updated.year == 2026
    assert result.updated.hour == 12
    assert len(result.links) == 3

    # Verify authoritative arxiv_primary_category takes precedence over category list order
    entry_with_primary = SimpleNamespace(
        id="oai:arXiv.org:2609.22090v1",
        title="Title",
        author="Author",
        summary="Summary",
        link="https://arxiv.org/abs/2609.22090",
        tags=[{"term": "cs.AI"}, {"term": "cs.LG"}],
        arxiv_primary_category={"term": "cs.LG"},
    )
    result_with_primary = arxiv_retriever._entry_to_arxiv_result(entry_with_primary)
    assert result_with_primary.primary_category == "cs.LG"

    # Verify Atom's arxiv_journal_reference field is mapped to journal_ref
    entry_with_journal_ref = SimpleNamespace(
        id="oai:arXiv.org:2609.22090v1",
        title="Title",
        author="Author",
        summary="Summary",
        link="https://arxiv.org/abs/2609.22090",
        arxiv_journal_reference="Phys. Rev. Lett. 120, 012345 (2025)",
    )
    result_with_journal_ref = arxiv_retriever._entry_to_arxiv_result(entry_with_journal_ref)
    assert result_with_journal_ref.journal_ref == "Phys. Rev. Lett. 120, 012345 (2025)"


def test_retrieve_raw_papers_cross_list(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(config.source.arxiv, "include_cross_list", True)
    retriever = ArxivRetriever(config)
    raw = retriever._retrieve_raw_papers()

    expected_len = len([
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") in {"new", "cross"}
    ])
    assert len(raw) == expected_len


def test_retrieve_raw_papers_debug_mode(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(config.executor, "debug", True)
    monkeypatch.setattr(config.source.arxiv, "include_cross_list", True)
    retriever = ArxivRetriever(config)
    raw = retriever._retrieve_raw_papers()
    assert len(raw) <= 10


def test_retrieve_raw_papers_retries_on_http_failure(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    calls = []

    def mock_parse_flaky(url):
        calls.append(url)
        if len(calls) < 2:
            return SimpleNamespace(
                status=500,
                feed=SimpleNamespace(title="Server Error"),
                entries=[],
                bozo=False,
            )
        return mock_feedparser

    monkeypatch.setattr(arxiv_retriever.feedparser, "parse", mock_parse_flaky)
    retriever = ArxivRetriever(config)
    raw = retriever._retrieve_raw_papers()
    assert len(calls) == 2
    assert len(raw) > 0



def test_api_mode_filters_cross_lists_unless_enabled(config, monkeypatch):
    config.source.arxiv.lookback_days = 7
    results = {
        "cs.AI": [_fake_api_result("2610.00001", "cs.AI"), _fake_api_result("2610.00004", "stat.ML")],
        "cs.CV": [],
    }
    _install_fake_api(monkeypatch, results)
    assert [r.primary_category for r in ArxivRetriever(config)._retrieve_raw_papers()] == ["cs.AI"]

    config.source.arxiv.include_cross_list = True
    _install_fake_api(monkeypatch, results)
    assert [r.primary_category for r in ArxivRetriever(config)._retrieve_raw_papers()] == ["cs.AI", "stat.ML"]


def test_api_mode_keeps_other_categories_when_one_fails(config, monkeypatch):
    config.source.arxiv.lookback_days = 7

    def _partial_then_fail():
        yield _fake_api_result("2610.00001", "cs.AI")
        raise RuntimeError("arXiv API returned an unexpectedly empty page")

    _install_fake_api(monkeypatch, {
        "cs.AI": _partial_then_fail(),
        "cs.CV": [_fake_api_result("2610.00003", "cs.CV")],
    })

    raw = ArxivRetriever(config)._retrieve_raw_papers()

    assert [r.entry_id for r in raw] == ["http://arxiv.org/abs/2610.00001v1", "http://arxiv.org/abs/2610.00003v1"]


def test_api_mode_debug_limits_results(config, monkeypatch):
    config.source.arxiv.lookback_days = 7
    config.executor.debug = True
    searches = _install_fake_api(monkeypatch, {
        "cs.AI": [_fake_api_result(f"2610.{i:05d}", "cs.AI") for i in range(8)],
        "cs.CV": [_fake_api_result(f"2610.{i:05d}", "cs.CV") for i in range(100, 108)],
    })

    raw = ArxivRetriever(config)._retrieve_raw_papers()

    assert all(s.max_results == 10 for s in searches)
    assert len(raw) == 10


def test_api_mode_not_used_without_lookback(config, mock_feedparser, monkeypatch):
    def _no_api(**kw):
        raise AssertionError("RSS mode must not query the arXiv API")

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", _no_api)
    assert len(ArxivRetriever(config)._retrieve_raw_papers()) > 0
