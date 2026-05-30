"""Robotic Grasp Prediction — interactive Streamlit demo.

Upload a tabletop image (or pick a Cornell example) and the Phase-4 champion
(global ResNet-18 regressor) predicts a single oriented parallel-jaw grasp:
where to place the gripper, how wide to open it, and how to orient the jaws.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cornell import GraspRect, load_samples  # noqa: E402
from data_pipeline import load_config  # noqa: E402
from predict import GraspPredictor  # noqa: E402
from viz import render_grasp  # noqa: E402

st.set_page_config(page_title="Robotic Grasp Prediction", page_icon="🦾", layout="wide")


@st.cache_resource(show_spinner="Loading champion model…")
def get_predictor():
    cfg = load_config()
    return GraspPredictor(config=cfg), cfg


@st.cache_data(show_spinner=False)
def load_eval_metrics() -> dict:
    p = REPO_ROOT / "results" / "phase6_evaluation.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data(show_spinner=False)
def load_llm_table() -> pd.DataFrame | None:
    p = REPO_ROOT / "results" / "llm_vs_custom.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def list_examples() -> dict:
    """Map a friendly label -> CornellSample for a few folder-03 (held-out
    test) images, so the demo defaults to images the model never trained on."""
    cfg = load_config()
    samples = load_samples(REPO_ROOT / cfg["data"]["cornell_root"] / "03")
    out = {}
    for s in samples[:8]:
        out[f"Cornell test pcd{s.pcd_id:04d}  (held-out object)"] = s
    return out


predictor, cfg = get_predictor()
ref = cfg["reference"]
eval_metrics = load_eval_metrics()

# ---------------------------------------------------------------------------
# Sidebar — model card at a glance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Champion model")
    st.caption("Phase-4 tuned global ResNet-18 regressor")
    acc = eval_metrics.get("accuracy", ref["champion_accuracy"])
    miou = eval_metrics.get("median_iou", ref["champion_median_iou"])
    st.metric("Jiang accuracy (folder-03, n=100)", f"{acc:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("Median IoU", f"{miou:.3f}")
    c2.metric("Median angle err", f"{eval_metrics.get('median_angle_err', ref['champion_median_angle_err_deg']):.1f}°")

    bench = eval_metrics.get("latency_benchmark", {})
    if bench:
        fastest = min(bench.items(), key=lambda kv: kv[1]["ms_per_image"])
        st.metric(f"Inference ({fastest[0]})", f"{fastest[1]['ms_per_image']:.1f} ms/img")

    st.divider()
    st.subheader("vs. published Cornell (object-wise)")
    st.table(pd.DataFrame({
        "Model": ["Lenz 2014", "This model", "Redmon 2015", "Cao 2023 (SOTA, img-wise)"],
        "Acc": [ref["lenz_2014_object_wise"], acc,
                ref["redmon_2015_object_wise"], ref["cao_2023_sota_image_wise"]],
    }).set_index("Model"))
    st.caption("Beats Lenz 2014; 6.9 pp under Redmon 2015. SOTA shown for scale.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🦾 Robotic Grasp Prediction")
st.markdown(
    "Predict a **parallel-jaw grasp** — center, jaw width, plate depth, and "
    "orientation — for a tabletop object, from a single RGB image. Trained on the "
    "**Cornell Grasping Dataset** and evaluated with the field-standard "
    "**Jiang metric** (IoU > 0.25 **and** angle error < 30°)."
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1 · Choose an image")
    examples = list_examples()
    mode = st.radio("Source", ["Cornell example (held-out)", "Upload your own"],
                    horizontal=True, label_visibility="collapsed")

    image: Image.Image | None = None
    gt_rects: list[GraspRect] | None = None
    show_gt = st.checkbox("Overlay ground-truth grasps (Cornell examples only)", value=True)

    if mode.startswith("Cornell"):
        choice = st.selectbox("Example", list(examples.keys()))
        sample = examples[choice]
        image = Image.open(sample.rgb_path).convert("RGB")
        gt_rects = sample.positives if show_gt else None
    else:
        up = st.file_uploader("Upload a tabletop RGB image", type=["png", "jpg", "jpeg"])
        if up is not None:
            image = Image.open(up).convert("RGB")

with right:
    st.subheader("2 · Predicted grasp")
    if image is None:
        st.info("Pick an example or upload an image to see a predicted grasp.")
    else:
        t0 = time.perf_counter()
        pred = predictor.predict(image)
        wall_ms = 1000.0 * (time.perf_counter() - t0)
        fig = render_grasp(image, pred.to_grasp_rect(), gt_rects=gt_rects)
        st.pyplot(fig, width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Angle", f"{pred.angle_deg:+.1f}°")
        m2.metric("Jaw opening", f"{pred.width:.0f} px")
        m3.metric("Plate depth", f"{pred.height:.0f} px")
        m4, m5, m6 = st.columns(3)
        m4.metric("Center x,y", f"{pred.cx:.0f}, {pred.cy:.0f}")
        m5.metric("Confidence", f"{pred.confidence:.2f}")
        m6.metric("Latency", f"{pred.latency_ms:.1f} ms")
        st.caption(
            "Confidence is a heuristic — the norm of the predicted (sin2θ, cos2θ) "
            "orientation vector, not a calibrated probability. The mint box is the "
            "prediction; red edges are the gripper plates; amber boxes (if shown) "
            "are human-annotated valid grasps."
        )

st.divider()

# ---------------------------------------------------------------------------
# Frontier-LLM head-to-head
# ---------------------------------------------------------------------------
st.subheader("This 45 MB CNN vs. frontier LLMs (Phase-5 head-to-head, n=40)")
llm = load_llm_table()
if llm is not None:
    df = llm.copy()
    df = df.rename(columns={
        "model": "Model", "accuracy": "Jiang acc", "median_iou": "Median IoU",
        "latency_s": "Latency (s)", "cost_per_1k_usd": "Cost / 1k ($)",
    })
    df = df[["Model", "Jiang acc", "Median IoU", "Latency (s)", "Cost / 1k ($)"]]
    st.dataframe(df.set_index("Model"), width="stretch")
    st.markdown(
        "- **Claude Opus / Haiku fail the spatial task** (median IoU 0.0 — right angle, "
        "wrong place). Grasping is a localisation problem language models reason past.\n"
        "- **GPT-5.5 ties the custom CNN on accuracy (0.75)** and edges it on IoU — but at "
        "**~2,400× the latency** and **effectively infinite cost-per-call** vs a free CPU "
        "forward pass.\n"
        "- The headline: on a precise spatial task, a tiny purpose-built CNN matches a "
        "frontier model at a fraction of a millisecond and zero marginal cost."
    )

with st.expander("How it works"):
    st.markdown(
        "1. **Input** — the RGB image is resized to 224×224 and ImageNet-normalised.\n"
        "2. **Backbone** — an ImageNet-pretrained ResNet-18 extracts features.\n"
        "3. **Head** — a linear layer regresses 6 numbers: "
        "`(cx, cy, w, h, sin2θ, cos2θ)`. The doubled-angle `(sin2θ, cos2θ)` "
        "encoding handles a grasp's 180° symmetry without a wrap-around "
        "discontinuity at ±90°.\n"
        "4. **Decode** — the 6-vector is projected back to the 640×480 frame and "
        "turned into an oriented rectangle.\n\n"
        "**Why a global regressor?** Phase 5 tested the popular per-pixel paradigm "
        "(GG-CNN / GR-ConvNet) on the *same* pretrained backbone and it **lost by "
        "27 pp** — the per-pixel decoder has no ImageNet-pretrained equivalent, "
        "forfeiting this dataset's single strongest lever exactly where it localises."
    )
