"""L4 view compiler: the observer's human-facing HTML page.

Compiled from (RunState, NodeStore episodes, Ledger stats). Honesty rules:
- watermark banner states what the page knows and through which event;
- every episode line is a drill-down (details/summary) to its digest + evidence pins;
- superseded/hindsight versions are visible in the episode's history dropdown —
  provenance is a feature, not clutter;
- anomaly badges show active issues; cleared ones are greyed in the log.
Static HTML + meta-refresh: no server, works over any file path.
"""

from __future__ import annotations

from datetime import datetime, timezone

from jinja2 import Environment

from tracelab.derived.nodes import NodeKind, NodeStore, Validity
from tracelab.state.runstate import RunState

PAGE = Environment(autoescape=True).from_string(r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
{% if refresh %}<meta http-equiv="refresh" content="{{ refresh }}">{% endif %}
<title>tracelab · {{ s.session_id[:8] }}</title>
<style>
 body{font:14px/1.5 -apple-system,Helvetica,sans-serif;margin:0;background:#11141a;color:#dfe3ea}
 .wrap{max-width:1060px;margin:0 auto;padding:22px}
 h1{font-size:17px;margin:0 0 2px} .dim{color:#8a93a3} .mono{font-family:Menlo,monospace;font-size:12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
 .card{background:#1a1f29;border:1px solid #2a3140;border-radius:8px;padding:10px 12px}
 .card b{font-size:16px} .card .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#8a93a3}
 .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-right:6px}
 .warn{background:#4a3a12;color:#f0c05a} .critical{background:#4a1a1a;color:#ff7b72}
 .ok{background:#15351f;color:#66d18a} .stale{background:#3a2f4a;color:#c0a0f0}
 details{border:1px solid #2a3140;border-radius:8px;margin:6px 0;background:#161a22}
 summary{padding:8px 12px;cursor:pointer;display:flex;gap:8px;align-items:baseline}
 summary .t{flex:1} .body{padding:4px 14px 12px;border-top:1px solid #232a38}
 .frontier{background:#122033;border:1px solid #1f3a5c;border-radius:8px;padding:10px 14px;margin:12px 0}
 table{border-collapse:collapse;font-size:12.5px} td,th{padding:3px 10px 3px 0;text-align:left;color:#b8bfcc}
 .watermark{font-size:11.5px;color:#8a93a3;border-top:1px solid #2a3140;margin-top:18px;padding-top:8px}
 .pin{color:#5aa0f0}
</style></head><body><div class="wrap">

<h1>{{ goal }}</h1>
<div class="dim">{{ s.source_path }} · session {{ s.session_id[:8] }} ·
 {{ "%.1f"|format(s.wall_clock_s/3600) if s.wall_clock_s else "?" }}h wall clock ·
 rendered {{ now }}</div>

<div class="frontier"><b>NOW:</b> {{ s.frontier or "—" }}
{% if s.pending_tools %} · <span class="badge warn">{{ s.pending_tools|length }} tool(s) in flight:
 {{ s.pending_tools|map(attribute='tool_name')|join(', ') }}</span>{% endif %}
{% for a in active_anomalies %}<span class="badge {{ a.severity }}">{{ a.kind }}: {{ a.message }}</span>{% endfor %}
{% if not active_anomalies %}<span class="badge ok">no active anomalies</span>{% endif %}
</div>

<div class="grid">
 <div class="card"><div class="lbl">turns</div><b>{{ s.n_turns }}</b></div>
 <div class="card"><div class="lbl">tool calls</div><b>{{ s.n_tool_calls }}</b> <span class="dim">({{ s.n_tool_errors }} err)</span></div>
 <div class="card"><div class="lbl">est. cost</div><b>${{ "%.2f"|format(s.est_cost_usd) }}</b></div>
 <div class="card"><div class="lbl">output tok</div><b>{{ "%.0fk"|format(s.tokens.output/1000) }}</b></div>
 <div class="card"><div class="lbl">cache read</div><b>{{ "%.1fM"|format(s.tokens.cache_read/1e6) }}</b></div>
 <div class="card"><div class="lbl">files touched</div><b>{{ s.files_touched|length }}</b></div>
 <div class="card"><div class="lbl">episodes</div><b>{{ episodes|length }}</b></div>
</div>

<h1 style="margin-top:20px">Episodes <span class="dim">(newest first — click to drill down)</span></h1>
{% for ep in episodes|reverse %}
<details {% if loop.first %}open{% endif %}>
 <summary>
  {% set st = ep.structured.get('status','?') %}
  <span class="badge {{ 'ok' if st=='ok' else ('critical' if st=='errors' else ('warn' if st=='interrupted' else 'stale')) }}">{{ st }}</span>
  <span class="t">{{ ep.structured.get('ask','')[:110] }}</span>
  <span class="dim mono">seq {{ ep.covered_lo }}–{{ ep.covered_hi }}{% if validity[ep.ref] != 'current' %} · {{ validity[ep.ref] }}{% endif %}</span>
 </summary>
 <div class="body">
  <div>{{ ep.content }}</div>
  <table>
   <tr><th>tools</th><td class="mono">{{ ep.structured.get('tools',{}) }}</td></tr>
   {% if ep.structured.get('conclusion') %}<tr><th>conclusion</th><td>{{ ep.structured['conclusion'] }}</td></tr>{% endif %}
   {% if ep.structured.get('dangling_tools') %}<tr><th>dangling</th><td>{{ ep.structured['dangling_tools']|join(', ') }}</td></tr>{% endif %}
   {% if ep.evidence_pins %}<tr><th>evidence</th><td class="mono pin">seq {{ ep.evidence_pins|join(', ') }}</td></tr>{% endif %}
   <tr><th>provenance</th><td class="mono dim">{{ ep.ref }} · {{ ep.producer }} · {{ history_counts[ep.node_id] }} version(s)</td></tr>
  </table>
 </div>
</details>
{% endfor %}

{% if s.recent_failures %}
<h1 style="margin-top:20px">Recent failures</h1>
{% for fl in s.recent_failures|reverse %}<div class="mono dim">· {{ fl }}</div>{% endfor %}
{% endif %}

<div class="watermark">
 honesty: state materialized at seq {{ s.state_materialized_at_seq }} · observed through seq
 {{ s.observed_seq }} · source line {{ marks.source_line - 1 }} · {{ malformed }} malformed line(s)
 {% if trailing %} · <b>{{ trailing }} bytes of a partial line pending</b>{% endif %}
 · per-tool: <span class="mono">{{ s.per_tool }}</span>
</div>
</div></body></html>""")


def render(state: RunState, store: NodeStore, *, marks=None, malformed: int = 0,
           trailing: int = 0, refresh: int | None = None) -> str:
    episodes = store.heads(NodeKind.EPISODE)
    return PAGE.render(
        s=state,
        goal=(state.goal_text or "(no goal captured yet)")[:160],
        episodes=episodes,
        validity={n.ref: store.validity(n.ref).value for n in episodes},
        history_counts={n.node_id: len(store.history(n.node_id)) for n in episodes},
        active_anomalies=state.active_anomalies,
        now=datetime.now(timezone.utc).strftime("%H:%M:%SZ"),
        marks=marks or type("M", (), {"source_line": 1})(),
        malformed=malformed, trailing=trailing, refresh=refresh,
    )
