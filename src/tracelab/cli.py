"""tracelab CLI: watch / render / sessions / bench."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tracelab.derived.episodes import EpisodeBuilder
from tracelab.detect.detectors import default_detectors
from tracelab.ledger.store import Ledger
from tracelab.state.reducers import StateFolder
from tracelab.views.human_html import render

PROJECTS = Path.home() / ".claude" / "projects"


def _pipeline(source: Path):
    led = Ledger(source)
    folder = StateFolder(str(source), detectors=default_detectors())
    epb = EpisodeBuilder()
    return led, folder, epb


def _consume(led: Ledger, folder: StateFolder, epb: EpisodeBuilder, events) -> None:
    for ev in events:
        folder.apply(ev)
        epb.feed(ev)


def _write_page(out: Path, led, folder, epb, refresh=None) -> None:
    html = render(folder.state, epb.store, marks=led.marks, malformed=led.malformed,
                  trailing=getattr(led, "trailing", 0), refresh=refresh)
    out.write_text(html)


def cmd_render(args) -> int:
    led, folder, epb = _pipeline(Path(args.session))
    led.ingest_available()
    _consume(led, folder, epb, led.events)
    epb.finalize(led.marks.observed_seq + 1)
    out = Path(args.out or (Path(args.session).stem + ".html"))
    _write_page(out, led, folder, epb)
    print(out)
    return 0


def cmd_watch(args) -> int:
    src = Path(args.session)
    out = Path(args.out or (src.stem + ".html"))
    led, folder, epb = _pipeline(src)
    print(f"watching {src} -> {out}  (ctrl-c to stop)", file=sys.stderr)
    try:
        while True:
            n = led.ingest_available()
            if n:
                _consume(led, folder, epb, led.events[-n:])
            _write_page(out, led, folder, epb, refresh=max(2, int(args.poll)))
            time.sleep(args.poll)
    except KeyboardInterrupt:
        return 0


def cmd_sessions(args) -> int:
    files = sorted(PROJECTS.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime,
                   reverse=True)[: args.n]
    for f in files:
        age_min = (time.time() - f.stat().st_mtime) / 60
        print(f"{age_min:7.1f}m  {f.stat().st_size/1e6:6.1f}MB  {f}")
    return 0


def cmd_fleet(args) -> int:
    """Iterate over ALL recent Claude Code sessions: render each observer page and
    build one fleet index linking them, with live state per session."""
    import html as _html
    out_dir = Path(args.out or "demo/fleet")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in PROJECTS.glob("*/*.jsonl") if "subagents" not in p.parts]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[: args.n]
    rows = []
    for f in files:
        try:
            led, folder, epb = _pipeline(f)
            led.ingest_available()
            _consume(led, folder, epb, led.events)
            epb.finalize(led.marks.observed_seq + 1)
            page = out_dir / (f.parent.name[-40:] + "__" + f.stem[:8] + ".html")
            _write_page(page, led, folder, epb)
            s = folder.state
            age_min = (time.time() - f.stat().st_mtime) / 60
            active = [a.kind for a in s.anomalies if getattr(a, "active", True)]
            rows.append(dict(
                page=page.name, age_min=age_min, size_mb=f.stat().st_size / 1e6,
                project=f.parent.name.split("-")[-1] or f.parent.name,
                ask=(s.latest_user_directive or s.goal_text or "")[:140],
                turns=s.n_turns, tools=s.n_tool_calls, cost=s.est_cost_usd,
                anomalies=active))
        except Exception as e:  # noqa: BLE001 — a broken session must not kill the fleet
            rows.append(dict(page=None, age_min=0, size_mb=0, project=f.parent.name,
                             ask=f"(failed to parse: {e})", turns=0, tools=0,
                             cost=0.0, anomalies=["parse-error"]))
    body = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>tracelab fleet — all sessions</title><style>",
            "body{font:15px/1.5 system-ui;margin:32px;background:#f9f9f7;color:#0b0b0b}",
            "table{border-collapse:collapse;width:100%;background:#fcfcfb;"
            "border:1px solid rgba(11,11,11,.1)}",
            "th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #e1e0d9;"
            "font-size:14px}th{color:#52514e}",
            ".live{color:#006300;font-weight:600}.anom{color:#d03b3b;font-weight:600}",
            ".quiet{color:#898781}</style></head><body>",
            f"<h1>tracelab fleet</h1><p class='quiet'>{len(rows)} most recent "
            f"sessions · generated {time.strftime('%Y-%m-%d %H:%M')} · deterministic "
            "fold, no LLM</p>",
            "<table><tr><th>age</th><th>project</th><th>latest ask / goal</th>"
            "<th>turns</th><th>tools</th><th>est. cost</th><th>alerts</th>"
            "<th>size</th></tr>"]
    for r in rows:
        age = ("<span class='live'>LIVE</span>" if r["age_min"] < 5
               else f"{r['age_min']/60:.1f}h" if r["age_min"] > 90
               else f"{r['age_min']:.0f}m")
        anom = (f"<span class='anom'>{', '.join(r['anomalies'])}</span>"
                if r["anomalies"] else "—")
        link = (f"<a href='{r['page']}'>{_html.escape(r['project'])}</a>"
                if r["page"] else _html.escape(r["project"]))
        body.append(
            f"<tr><td>{age}</td><td>{link}</td>"
            f"<td>{_html.escape(r['ask'])}</td><td>{r['turns']}</td>"
            f"<td>{r['tools']}</td><td>${r['cost']:.2f}</td><td>{anom}</td>"
            f"<td>{r['size_mb']:.1f}MB</td></tr>")
    body.append("</table></body></html>")
    idx = out_dir / "index.html"
    idx.write_text("\n".join(body))
    print(idx)
    return 0


def cmd_bench(args) -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "bench" / "scoreboard.json"
    if args.which in ("detect", "all"):
        from tracelab.bench.detect_bench import run as run_detect
        r = run_detect(out=out)
        print("DETECT:", {k: (v["precision"], v["recall"]) for k, v in r["per_kind"].items()})
    if args.which in ("fidelity", "all"):
        from tracelab.bench.fidelity_bench import run as run_fid
        r = run_fid(out=out)
        print("FIDELITY:", r["overall"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tracelab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="one-shot HTML for a session")
    r.add_argument("session")
    r.add_argument("--out")
    r.set_defaults(fn=cmd_render)

    w = sub.add_parser("watch", help="live-follow a session into auto-refreshing HTML")
    w.add_argument("session")
    w.add_argument("--out")
    w.add_argument("--poll", type=float, default=2.0)
    w.set_defaults(fn=cmd_watch)

    s = sub.add_parser("sessions", help="list recent Claude Code sessions")
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(fn=cmd_sessions)

    fl = sub.add_parser("fleet", help="render ALL recent sessions + one index page")
    fl.add_argument("-n", type=int, default=25)
    fl.add_argument("--out")
    fl.set_defaults(fn=cmd_fleet)

    b = sub.add_parser("bench", help="run benchmarks")
    b.add_argument("which", choices=["detect", "fidelity", "all"])
    b.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
