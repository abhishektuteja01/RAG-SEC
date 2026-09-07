"""Define the Day 10 Langfuse dashboard as code and push it to the project.

Writes one dashboard ("RAG-SEC -- Arm 6 loop vs static") whose widgets answer the published
trajectory and operational metrics -- retrieval calls, tokens and dollars, wall clock, judge
accuracy, and p50/p95 latency per stage: per-arm cost, per-question and per-stage latency
(the numbers Day 13 gates on), loop trajectory, judge verdicts, prefix-cache share (AGENT-8)
and error level.

Idempotent by name: widgets and the dashboard are looked up by name and PATCHed, so re-running
never duplicates. Nothing is deleted unless --prune (stale placements) or --delete (the whole
dashboard) is passed.

Every write goes through the `unstable/` dashboard API (snapshot 4.16.0). That surface strips
unknown body keys silently instead of erroring, so this script re-reads what it wrote and
compares field by field -- a half-built dashboard that renders is worse than a failed run.
"""

import argparse
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

API_SNAPSHOT = "4.16.0"
DASHBOARD = "RAG-SEC -- Arm 6 loop vs static"
DESCRIPTION = "Day 10 operational view of the agentic loop against its paired static baseline."

# The stages whose p50/p95 are published. embed/search/rerank are the retriever legs;
# plan/judge/generate are the three LLM calls.
STAGES = [
    "embed-query",
    "search-candidates",
    "rerank-candidates",
    "plan-query",
    "judge-sufficiency",
    "generate-answer",
]

LOOP = {"column": "tags", "operator": "any of", "value": ["arm:loop"], "type": "arrayOptions"}
ARMS = {
    "column": "tags",
    "operator": "any of",
    "value": ["arm:loop", "arm:static"],
    "type": "arrayOptions",
}
# The root observation is one per question per arm, so filtering to it turns observation-level
# aggregates into per-question ones for anything that lives on the root (latency, count).
ROOTS = {"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"}


def latency(agg):
    return {"measure": "latency", "agg": agg}


def counter(name, desc, filters):
    return {
        "name": name,
        "description": desc,
        "view": "observations",
        "chartType": "NUMBER",
        "dimensions": [],
        "metrics": [{"measure": "count", "agg": "count"}],
        "filters": filters,
    }


def iteration_tile(n, x):
    return (
        counter(
            f"Loop: questions that reached iteration {n}",
            "plan-query carries the 1-based iteration in metadata; iteration 4 is the cap.",
            [
                {"column": "name", "operator": "=", "value": "plan-query", "type": "string"},
                # metadata numbers compare as their JSON text, so the filter type is stringObject
                # even though `iteration` is an int in the trace.
                {
                    "column": "metadata",
                    "key": "iteration",
                    "operator": "=",
                    "value": str(n),
                    "type": "stringObject",
                },
            ],
        )
        | {"place": (x, 12, 3, 4)}
    )


def verdict_tile(verdict, x):
    return counter(
        f"Judge verdict: {verdict}",
        "judge-sufficiency generations, split by the verdict stored on the observation.",
        [
            {
                "column": "metadata",
                "key": "verdict",
                "operator": "=",
                "value": verdict,
                "type": "stringObject",
            }
        ],
    ) | {"place": (x, 16, 3, 4)}


WIDGETS = [
    {
        "name": "Cost by arm (USD, sum)",
        # Cost cannot be averaged per question here: observation cost is the observation's own,
        # the root AGENT span costs 0, and the unstable API does not serve the traces view. So
        # this is the numerator and "Questions traced by arm" is the denominator.
        "description": "Sum of observation cost. Divide by the question count tile for $/question.",
        "view": "observations",
        "chartType": "VERTICAL_BAR",
        "dimensions": [{"field": "tags"}],
        "metrics": [{"measure": "totalCost", "agg": "sum"}],
        "filters": [ARMS],
        "place": (0, 0, 6, 6),
    },
    {
        "name": "Questions traced by arm",
        "description": "Root answer-question spans: one per question per arm.",
        "view": "observations",
        "chartType": "VERTICAL_BAR",
        "dimensions": [{"field": "tags"}],
        "metrics": [{"measure": "count", "agg": "count"}],
        "filters": [ARMS, ROOTS],
        "place": (6, 0, 6, 6),
    },
    {
        "name": "Question latency by arm (ms)",
        # The loop root encloses the static trace (agent_run.py opens it inside run_one), so the
        # loop bar carries the static arm's generate time too -- ~2s against a ~80s p50.
        "description": "Root-span duration, p50/p95. The loop figure includes the nested static call.",
        "view": "observations",
        "chartType": "PIVOT_TABLE",
        "dimensions": [{"field": "tags"}],
        "metrics": [latency("p50"), latency("p95")],
        "filters": [ARMS, ROOTS],
        "chartConfig": {"type": "PIVOT_TABLE", "row_limit": 20},
        "place": (0, 6, 6, 6),
    },
    {
        "name": "Stage latency p50/p95 (ms), loop arm",
        "description": "Per-stage latency. Day 13's budget is gated on the p95 column.",
        "view": "observations",
        "chartType": "PIVOT_TABLE",
        "dimensions": [{"field": "name"}],
        "metrics": [latency("p50"), latency("p95")],
        "filters": [
            LOOP,
            {"column": "name", "operator": "any of", "value": STAGES, "type": "stringOptions"},
        ],
        "chartConfig": {"type": "PIVOT_TABLE", "row_limit": 20},
        "place": (6, 6, 6, 6),
    },
    *[iteration_tile(n, x) for n, x in zip((1, 2, 3, 4), (0, 3, 6, 9))],
    verdict_tile("finish", 0),
    verdict_tile("loop", 3),
    {
        "name": "Input tokens: cached vs fresh",
        # usageType is groupable but not filterable, so the `total` bar cannot be hidden; read
        # input_cache_read against input, which is already exclusive of the cached read.
        "description": "input vs input_cache_read (AGENT-8). The `total` bar is the sum row.",
        "view": "observations",
        "chartType": "VERTICAL_BAR",
        "dimensions": [{"field": "usageType"}],
        "metrics": [{"measure": "usageByType", "agg": "sum"}],
        "filters": [ARMS],
        "place": (6, 16, 6, 6),
    },
    {
        "name": "Observations by level",
        "description": "Error rate: the ERROR slice over all observations in both arms.",
        "view": "observations",
        "chartType": "PIE",
        "dimensions": [{"field": "level"}],
        "metrics": [{"measure": "count", "agg": "count"}],
        "filters": [ARMS],
        "place": (0, 22, 6, 6),
    },
]

BODY_FIELDS = ("name", "description", "view", "chartType", "dimensions", "metrics", "filters")


class Api:
    def __init__(self, base, public, secret):
        self.base = base.rstrip("/")
        self.http = httpx.Client(auth=(public, secret), timeout=30.0)

    def __call__(self, method, path, **kw):
        # Cloud allows 30 requests/min on this surface and a full build is ~30 calls, so a
        # re-run lands on a 429 mid-way; without this the dashboard is left half-placed.
        for _ in range(4):
            r = self.http.request(method, f"{self.base}/api/public{path}", **kw)
            if r.status_code != 429:
                break
            wait = r.json().get("details", {}).get("retryAfterSeconds", 60) + 1
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
        if r.status_code >= 300:
            raise SystemExit(
                f"{method} {path} -> {r.status_code} (API snapshot {API_SNAPSHOT})\n{r.text}"
            )
        return r.json() if r.content else {}

    def list_all(self, path):
        out, page = [], 1
        while True:
            body = self("GET", path, params={"page": page, "limit": 50})
            out += body["data"]
            if page >= body["meta"]["totalPages"]:
                return out
            page += 1


def body_of(spec):
    body = {k: spec[k] for k in BODY_FIELDS}
    body["chartConfig"] = spec.get("chartConfig", {"type": spec["chartType"]})
    return body


def by_name(items, name):
    hits = [i for i in items if i["name"] == name]
    if len(hits) > 1:
        # Names are not unique server-side; keeping the oldest makes re-runs converge on one
        # object instead of ping-ponging between duplicates.
        hits.sort(key=lambda i: i["createdAt"])
        print(f"  warn: {len(hits)} objects named {name!r}; using the oldest")
    return hits[0] if hits else None


def sync_widgets(api):
    existing = api.list_all("/unstable/dashboard-widgets")
    ids = {}
    for spec in WIDGETS:
        body = body_of(spec)
        found = by_name(existing, spec["name"])
        if found:
            got = api("PATCH", f"/unstable/dashboard-widgets/{found['id']}", json=body)
            action = "updated"
        else:
            got = api("POST", "/unstable/dashboard-widgets", json=body)
            action = "created"
        ids[spec["name"]] = got["id"]
        print(f"  {action} widget {got['id']}  {spec['name']}")
    return ids


def sync_dashboard(api, ids, prune):
    found = by_name(api.list_all("/unstable/dashboards"), DASHBOARD)
    if found:
        dash = api(
            "PATCH",
            f"/unstable/dashboards/{found['id']}",
            json={"name": DASHBOARD, "description": DESCRIPTION},
        )
        print(f"  updated dashboard {dash['id']}")
    else:
        dash = api(
            "POST", "/unstable/dashboards", json={"name": DASHBOARD, "description": DESCRIPTION}
        )
        print(f"  created dashboard {dash['id']}")

    placed = {p["widgetId"]: p for p in dash["definition"]["widgets"]}
    for spec in WIDGETS:
        wid = ids[spec["name"]]
        x, y, w, h = spec["place"]
        geom = {"x": x, "y": y, "width": w, "height": h}
        if wid in placed:
            api(
                "PATCH",
                f"/unstable/dashboards/{dash['id']}/placements/{placed[wid]['id']}",
                json=geom,
            )
        else:
            api(
                "POST",
                f"/unstable/dashboards/{dash['id']}/placements",
                json={"type": "widget", "widgetId": wid, **geom},
            )
    stale = set(placed) - set(ids.values())
    for wid in sorted(stale):
        if prune:
            api("DELETE", f"/unstable/dashboards/{dash['id']}/placements/{placed[wid]['id']}")
            print(f"  pruned placement of {wid}")
        else:
            print(f"  note: placement of {wid} is not in this spec (--prune to remove)")
    return dash["id"]


def verify(api, dash_id, ids):
    dash = api("GET", f"/unstable/dashboards/{dash_id}")
    placed = {p["widgetId"] for p in dash["definition"]["widgets"]}
    # One paginated list instead of a GET per widget: the read-back would otherwise spend the
    # whole 30/min budget on its own verification.
    rows = {w["id"]: w for w in api.list_all("/unstable/dashboard-widgets")}
    bad = 0
    for spec in WIDGETS:
        wid = ids[spec["name"]]
        got = rows.get(wid, {})
        if "metrics" not in got:
            got = api("GET", f"/unstable/dashboard-widgets/{wid}")
        want = {k: v for k, v in body_of(spec).items() if k != "chartConfig"}
        diff = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
        if diff or wid not in placed:
            bad += 1
            print(f"  MISMATCH {spec['name']}: placed={wid in placed} {json.dumps(diff)}")
        else:
            m = ",".join(f"{x['agg']}({x['measure']})" for x in spec["metrics"])
            d = ",".join(x["field"] for x in spec["dimensions"]) or "-"
            print(f"  ok {spec['chartType']:<14} {m:<28} by {d:<10} {spec['name']}")
    if bad:
        raise SystemExit(f"{bad} widget(s) did not read back as written")
    print(f"  {len(WIDGETS)} widgets verified, {len(placed)} placements on the dashboard")


def delete_all(api):
    dash = by_name(api.list_all("/unstable/dashboards"), DASHBOARD)
    if dash:
        api("DELETE", f"/unstable/dashboards/{dash['id']}")
        print(f"  deleted dashboard {dash['id']}")
    for w in api.list_all("/unstable/dashboard-widgets"):
        if any(w["name"] == s["name"] for s in WIDGETS):
            api("DELETE", f"/unstable/dashboard-widgets/{w['id']}")
            print(f"  deleted widget {w['id']}  {w['name']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true", help="remove placements not in this spec")
    ap.add_argument("--delete", action="store_true", help="delete the dashboard and its widgets")
    args = ap.parse_args()

    base = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    public, secret = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not (base and public and secret):
        sys.exit("missing LANGFUSE_BASE_URL/LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY in env or .env")

    api = Api(base, public, secret)
    if args.delete:
        delete_all(api)
        return

    print(f"=== {DASHBOARD} ({base}, API snapshot {API_SNAPSHOT}) ===")
    ids = sync_widgets(api)
    dash_id = sync_dashboard(api, ids, args.prune)
    verify(api, dash_id, ids)
    # /projects is scoped to the API key, so the first (only) entry is this project.
    project = api("GET", "/projects")["data"][0]["id"]
    print(f"\n{base}/project/{project}/dashboards/{dash_id}")


if __name__ == "__main__":
    main()
