from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from analysis_utils import audio_energy_summary, make_suggestions, summarize_presence

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")
st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Offline-first classroom analytics prototype · Recorded media · Anonymous presence estimates")

with st.expander("Privacy and interpretation notice", expanded=True):
    st.warning(
        "This prototype does not identify students by name or measure attention directly. "
        "Face presence, movement, and audio activity are imperfect proxies. Use only media you are authorized "
        "to process, keep files local, and do not use these estimates alone for grading or discipline."
    )

video_file = st.file_uploader("Upload classroom video", type=["mp4", "mov", "avi", "mkv"])
audio_file = st.file_uploader("Optional audio recording", type=["wav", "mp3", "m4a", "ogg"])
col1, col2 = st.columns(2)
with col1:
    sample_interval = st.slider("Sample one frame every (seconds)", min_value=1, max_value=15, value=3)
with col2:
    min_observations = st.slider("Minimum face-positive samples for sustained presence", min_value=1, max_value=10, value=3)

def analyze_video(raw: bytes, interval: int, threshold: int):
    suffix = Path(video_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(raw)
        temp.flush()
        cap = cv2.VideoCapture(temp.name)
        if not cap.isOpened():
            raise ValueError("Could not open this video. Try a standard MP4 or AVI file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps > 0 else 0
        step = max(1, int((fps if fps > 0 else 1) * interval))
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        rows = []
        previous_gray = None
        frame_index = 0
        progress = st.progress(0)
        status = st.empty()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                # Resize large frames for quicker local processing.
                h, w = frame.shape[:2]
                scale = min(1.0, 720 / max(w, h))
                if scale < 1:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
                motion = 0.0
                if previous_gray is not None and previous_gray.shape == gray.shape:
                    diff = cv2.absdiff(previous_gray, gray)
                    motion = float(np.mean(diff) / 255.0)
                previous_gray = gray
                seconds = frame_index / fps if fps > 0 else float(frame_index)
                rows.append({
                    "timestamp_seconds": round(seconds, 2),
                    "faces_detected": int(len(faces)),
                    "visual_activity_proxy": round(motion, 5),
                })
            frame_index += 1
            if frame_count > 0 and frame_index % max(1, int(frame_count / 100)) == 0:
                progress.progress(min(1.0, frame_index / frame_count))
                status.caption(f"Processed {frame_index:,} / {int(frame_count):,} frames")
        cap.release()
        progress.empty()
        status.empty()
    if not rows:
        raise ValueError("No frames could be analyzed.")
    df = pd.DataFrame(rows)
    positive = int((df["faces_detected"] > 0).sum())
    summary = summarize_presence(df["faces_detected"].tolist(), threshold)
    metrics = {
        "duration": float(duration),
        "samples": len(df),
        "positive_samples": positive,
        "presence_ratio": summary.presence_ratio,
        "sustained_presence": summary.sustained_presence,
        "visual_activity": float(df["visual_activity_proxy"].mean()),
    }
    return df, metrics

if st.button("Analyze session", type="primary", disabled=video_file is None):
    try:
        with st.spinner("Analyzing locally…"):
            video_df, metrics = analyze_video(video_file.getvalue(), sample_interval, min_observations)
            audio_metrics = None
            if audio_file is not None:
                try:
                    audio_data, sr = sf.read(io.BytesIO(audio_file.getvalue()), dtype="float32", always_2d=True)
                    mono = np.mean(audio_data, axis=1)
                    audio_metrics = audio_energy_summary(mono, sr)
                except Exception as exc:
                    st.info(f"Audio could not be decoded by the installed audio reader: {exc}")

        st.session_state["video_df"] = video_df
        st.session_state["metrics"] = metrics
        st.session_state["audio_metrics"] = audio_metrics
        st.session_state["suggestions"] = make_suggestions(
            metrics["visual_activity"],
            audio_metrics["activity_ratio"] if audio_metrics else None,
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

if "metrics" in st.session_state:
    m = st.session_state["metrics"]
    a = st.session_state.get("audio_metrics")
    st.subheader("Session summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Video duration", f"{m['duration'] / 60:.1f} min")
    c2.metric("Sampled frames", f"{m['samples']:,}")
    c3.metric("Samples with face(s)", f"{m['presence_ratio']:.0%}")
    c4.metric("Visual activity proxy", f"{m['visual_activity']:.3f}")
    st.caption("Face-positive sample ratio is not a student attendance percentage. It only describes sampled frames.")
    df = st.session_state["video_df"]
    st.subheader("Sample timeline")
    st.line_chart(df.set_index("timestamp_seconds")[["faces_detected", "visual_activity_proxy"]])
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download video analysis CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="session_video_analysis.csv",
        mime="text/csv",
    )
    if a:
        st.subheader("Audio activity (optional)")
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Audio duration", f"{a['duration_seconds'] / 60:.1f} min")
        ac2.metric("Estimated activity windows", f"{a['activity_ratio']:.0%}")
        ac3.metric("Mean RMS energy", f"{a['mean_rms']:.4f}")
        audio_df = pd.DataFrame(a["windows"])
        if not audio_df.empty:
            st.line_chart(audio_df.set_index("start_seconds")[["rms"]])
            st.download_button(
                "Download audio activity CSV",
                data=audio_df.to_csv(index=False).encode("utf-8"),
                file_name="session_audio_activity.csv",
                mime="text/csv",
            )
    st.subheader("Local teaching suggestions")
    for suggestion in st.session_state["suggestions"]:
        st.write(f"- {suggestion}")
    st.caption("Suggestions are generated by transparent local rules, not a cloud AI model.")
