"""/ask/stream sends stage events in pipeline order, the sources, the answer text in pieces,
then the same body /ask returns.
Retrieval and Gemini are faked: no database, models or API key."""

import json

import pytest
from fastapi.testclient import TestClient

from rag_sec import api

PIECES = ["Net revenue ", "was 13,880.", "\nANSWER: 13880"]
CHUNKS = [{"filing_stem": "V_2015_1403161", "chunk_index": 3, "text": "Net revenue $13,880M",
           "score": 0.9}]


def fake_retrieve(query, k=10, on_stage=None):
    report = on_stage or (lambda name, info: None)
    report("resolve", {})
    report("embed", {"tickers": ["V"]})
    report("search", {})
    report("rerank", {"candidates": 1})
    return CHUNKS


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "retrieve", fake_retrieve)
    monkeypatch.setattr(api, "last_call_stats", lambda: {"timings": {"rerank_s": 1.0}})
    monkeypatch.setattr(api, "generate_answer", lambda q, c: "".join(PIECES))
    monkeypatch.setattr(api, "stream_answer", lambda q, c: iter(PIECES))
    return TestClient(api.app)  # no `with`: skips the lifespan model warm-up


def events(text):
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        yield lines["event"], json.loads(lines["data"])


def test_stream_orders_stages_and_matches_ask(client):
    body = {"question": "What was Visa's net revenue in 2015?", "include_text": True}
    got = list(events(client.post("/ask/stream", json=body).text))
    assert [d["stage"] for e, d in got if e == "stage"] == ["resolve", "embed", "search", "rerank",
                                                            "generate"]
    assert got[1][1]["tickers"] == ["V"]
    kinds = [e for e, _ in got]
    assert kinds.index("sources") < kinds.index("text") < kinds.index("answer")
    assert kinds.index("sources") < [i for i, (e, d) in enumerate(got) if d.get("stage") == "generate"][0]
    assert [d["text"] for e, d in got if e == "text"] == PIECES
    final_event, final = got[-1]
    assert final_event == "answer"
    plain = client.post("/ask", json=body).json()
    for key in ("answer", "value", "value_status", "citations"):
        assert final[key] == plain[key]
    assert dict(got)["sources"]["citations"] == plain["citations"]
    assert "generation_first_text_s" in final["stage_latency"]
    assert final["value"] == 13880 and final["value_status"] == "ok"


def test_stream_reports_errors_as_events(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve", lambda q, k=10, on_stage=None: [])
    got = list(events(client.post("/ask/stream", json={"question": "x"}).text))
    assert got[-1] == ("error", {"status": 404, "detail": "no candidates retrieved"})
