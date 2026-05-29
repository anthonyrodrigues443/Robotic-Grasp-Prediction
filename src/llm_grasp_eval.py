"""Frontier-LLM grasp-prediction harness for Phase 5.

Sends a Cornell RGB image to a frontier model via its local CLI and asks for a
single parallel-plate grasp rectangle, parsed to ``(cx, cy, w, h, angle_deg)``
in the 640×480 frame. Mirrors the proven Fraud-Detection ``mark_phase5``
pattern: deterministic stratified sample, append-only JSON cache (idempotent /
resumable), CLI invocation + defensive parsing + latency timing. Metric scoring
(Jiang IoU>0.25 ∧ angle<30°) is done by the *notebook* using ``cornell.py`` so
the LLM and the custom CNN are scored on identical indices with identical code.

Vision input:
  * Claude CLI reads the image from a path embedded in the prompt (it has the
    Read tool in --print mode): ``claude --print --model {opus,haiku}``.
  * Codex CLI attaches the image with ``-i``: ``codex exec -i IMG``.
Both were validated to return a clean parseable single line.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

PROMPT = (
    "You are a robotic grasp-detection expert. The image is a 640x480 RGB photo "
    "of object(s) on a table. Predict the single best parallel-plate gripper grasp.\n"
    "Reply with EXACTLY one line, space-separated, NO other text, NO explanation:\n"
    "cx cy w h angle\n"
    "where cx cy = grasp center in pixels (origin top-left, x right, y down); "
    "w = gripper jaw-opening length in pixels; h = gripper plate thickness in "
    "pixels (~15-30); angle = gripper orientation in degrees in [-90,90], "
    "0 = jaws open horizontally, positive = counterclockwise."
)

# Representative 2026 per-call cost (USD). A 640x480 image is ~1.2-1.4k vision
# tokens; prompt ~150 in, ~10 out. Codex carries agent overhead (~14k tok/call
# observed). Timing in the harness includes CLI startup; the cost math reflects
# equivalent direct-API usage.
COST_PER_CALL_USD = {
    "claude/opus": 0.0225,    # $15/MTok in, $75/MTok out @ ~1.45k in / 10 out
    "claude/haiku": 0.0015,   # $1/MTok in,  $5/MTok out
    "codex/gpt-5.5": 0.0500,  # codex agent overhead ~14k tok/call
    "custom_model": 1e-7,     # MPS/CPU inference, effectively free
}

_LINE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
                      r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def parse_grasp_response(text: str) -> tuple[float, float, float, float, float] | None:
    """Extract the first ``cx cy w h angle`` 5-tuple from CLI output. Returns
    None on parse failure (counted in the parse-success rate, not as wrong)."""
    if not text or text.startswith("__ERROR__"):
        return None
    for line in text.splitlines():
        m = _LINE_RE.search(line.strip())
        if m:
            cx, cy, w, h, ang = (float(v) for v in m.groups())
            if w <= 0 or h <= 0:
                continue
            return cx, cy, w, h, ang
    # fall back to a search over the whole blob
    m = _LINE_RE.search(text.replace("\n", " "))
    if m:
        cx, cy, w, h, ang = (float(v) for v in m.groups())
        if w > 0 and h > 0:
            return cx, cy, w, h, ang
    return None


def _slice_codex(raw: str) -> str:
    """Codex wraps the reply in session metadata. The answer sits between the
    last 'codex' marker line and the 'tokens used' line."""
    if "tokens used" in raw:
        raw = raw.split("tokens used")[0]
    if "\ncodex\n" in raw:
        raw = raw.rsplit("\ncodex\n", 1)[1]
    return raw.strip()


def call_claude(image_path: str, model: str, timeout: int = 120) -> tuple[str, float]:
    prompt = f"{PROMPT}\n\nAnalyze the image at this path and predict the grasp: {image_path}"
    t0 = time.time()
    try:
        r = subprocess.run(
            ["claude", "--print", "--model", model,
             "--no-session-persistence", "--disable-slash-commands"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout.strip() or f"__ERROR__ {r.stderr.strip()[:200]}"
    except subprocess.TimeoutExpired:
        out = "__ERROR__ timeout"
    except Exception as e:  # pragma: no cover - defensive
        out = f"__ERROR__ {e}"
    return out, time.time() - t0


def call_codex(image_path: str, timeout: int = 180) -> tuple[str, float]:
    t0 = time.time()
    try:
        r = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
             "-i", image_path, "-"],
            input=PROMPT, capture_output=True, text=True, timeout=timeout,
        )
        out = _slice_codex(r.stdout) if r.stdout else f"__ERROR__ {r.stderr.strip()[:200]}"
    except subprocess.TimeoutExpired:
        out = "__ERROR__ timeout"
    except Exception as e:  # pragma: no cover - defensive
        out = f"__ERROR__ {e}"
    return out, time.time() - t0


def run_llm_eval(jobs: list[dict], cache_path: str | Path) -> list[dict]:
    """``jobs`` is a list of {idx, pcd_id, image_path, llm, model}. Appends each
    completed call to ``cache_path`` (JSON list) and skips (idx, llm, model)
    triples already present. Returns the full cache list."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: list[dict] = []
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text())
    done = {(c["idx"], c["llm"], c["model"]) for c in cache}
    for job in jobs:
        key = (job["idx"], job["llm"], job["model"])
        if key in done:
            continue
        if job["llm"] == "claude":
            raw, lat = call_claude(job["image_path"], job["model"])
        elif job["llm"] == "codex":
            raw, lat = call_codex(job["image_path"])
        else:
            raise ValueError(job["llm"])
        parsed = parse_grasp_response(raw)
        rec = {
            "idx": job["idx"], "pcd_id": job["pcd_id"],
            "llm": job["llm"], "model": job["model"],
            "raw_text": raw[:500], "parsed": list(parsed) if parsed else None,
            "latency_s": round(lat, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        cache.append(rec)
        done.add(key)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"  [{job['llm']}/{job['model']}] idx={job['idx']} "
              f"parsed={rec['parsed']} {lat:.1f}s", flush=True)
    return cache
