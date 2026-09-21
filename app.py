from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from assemblyvision.config import load_config
from assemblyvision.pipeline import AssemblyPipeline
from assemblyvision.synthetic import generate_demo

ROOT = Path(__file__).parent
CONFIG = ROOT / "configs" / "default.yaml"
SAMPLE_VIDEO = ROOT / "data" / "samples" / "assembly_demo.mp4"
SAMPLE_LABELS = ROOT / "data" / "samples" / "assembly_demo.json"

st.set_page_config(page_title="AssemblyVision", page_icon="🏭", layout="wide")
st.title("AssemblyVision")
st.caption("Manual assembly activity recognition — industrial computer-vision research prototype")

with st.sidebar:
    st.header("Input")
    source = st.radio("Video source", ["Built-in demonstration", "Upload video"])
    uploaded = st.file_uploader("MP4/AVI video", type=["mp4", "avi", "mov"], disabled=source != "Upload video")
    regenerate = st.button("Regenerate demo video", use_container_width=True)
    run = st.button("Run recognition", type="primary", use_container_width=True)
    st.divider()
    st.info("The built-in detector uses colored markers. It is a transparent baseline intended to be replaced with a trained detector for real factory footage.")

if regenerate or not SAMPLE_VIDEO.exists():
    generate_demo(SAMPLE_VIDEO, SAMPLE_LABELS)
    if regenerate:
        st.success("Demo video generated.")

video_path = SAMPLE_VIDEO
if source == "Upload video" and uploaded:
    suffix = Path(uploaded.name).suffix
    target = Path(tempfile.gettempdir()) / f"assemblyvision_upload{suffix}"
    target.write_bytes(uploaded.getbuffer())
    video_path = target

st.video(str(video_path))

if run:
    with st.spinner("Analyzing video..."):
        output = Path(tempfile.gettempdir()) / "assemblyvision_annotated.mp4"
        result = AssemblyPipeline(load_config(CONFIG)).process(video_path, output, keep_frames=True)
    a, b, c, d = st.columns(4)
    a.metric("Process complete", "Yes" if result["completed"] else "No")
    b.metric("Detected events", len(result["events"]))
    c.metric("Mean latency", f"{result['mean_latency_ms']:.1f} ms")
    d.metric("Processing speed", f"{result['processing_fps']:.1f} FPS")
    st.subheader("Recognized activity timeline")
    event_rows = [event.to_dict() for event in result["events"]]
    if event_rows:
        frame = pd.DataFrame(event_rows)
        frame["activity"] = frame.pop("step").str.replace("_", " ").str.title()
        frame["confidence"] = frame["confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(frame[["activity", "timestamp_s", "frame", "confidence"]], use_container_width=True, hide_index=True)
    else:
        st.warning("No complete activity was recognized. Uploaded footage must match the configured objects and sequence.")
    st.subheader("Annotated samples")
    previews = result["preview_frames"]
    for start in range(0, len(previews), 4):
        columns = st.columns(4)
        for column, image in zip(columns, previews[start:start + 4]):
            column.image(image, use_container_width=True)
    with output.open("rb") as handle:
        st.download_button("Download annotated video", handle, file_name="assemblyvision_annotated.mp4", mime="video/mp4")

with st.expander("Research and responsible-use notes"):
    st.markdown("""
    A production study should split evaluation data by operator and recording session, compare the rule baseline
    with object detection, pose, and temporal video models, and test lighting, camera angle, occlusion, latency,
    and compute requirements. Employee participation, purpose limitation, data minimization, access control,
    retention periods, and applicable privacy/labor requirements must be addressed before collecting workplace video.
    """)
