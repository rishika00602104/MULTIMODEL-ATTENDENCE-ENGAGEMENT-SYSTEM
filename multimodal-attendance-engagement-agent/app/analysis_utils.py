"""Local, explainable utilities for the classroom analytics prototype."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class PresenceSummary:
    total_samples: int
    positive_samples: int
    presence_ratio: float
    sustained_presence: bool


def summarize_presence(face_counts: Iterable[int], minimum_observations: int = 3) -> PresenceSummary:
    """Summarize face detections across sampled frames (not person identity)."""
    counts = [max(0, int(c)) for c in face_counts]
    positives = sum(c > 0 for c in counts)
    ratio = positives / len(counts) if counts else 0.0
    return PresenceSummary(
        total_samples=len(counts),
        positive_samples=positives,
        presence_ratio=ratio,
        sustained_presence=positives >= max(1, int(minimum_observations)),
    )


def audio_energy_summary(samples: np.ndarray, sample_rate: int, frame_seconds: float = 0.5) -> dict:
    """Compute RMS energy windows and an adaptive activity estimate; no transcription."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0 or sample_rate <= 0:
        return {"duration_seconds": 0.0, "activity_ratio": 0.0, "mean_rms": 0.0, "windows": []}

    window_size = max(1, int(sample_rate * frame_seconds))
    rms_values = []
    for start in range(0, audio.size, window_size):
        chunk = audio[start:start + window_size]
        rms_values.append(float(np.sqrt(np.mean(np.square(chunk))) if chunk.size else 0.0))

    values = np.asarray(rms_values, dtype=float)
    # Simple adaptive threshold; this is only a rough audio-activity proxy.
    threshold = max(0.008, float(np.percentile(values, 35)) * 1.8)
    active = values > threshold
    return {
        "duration_seconds": float(audio.size / sample_rate),
        "activity_ratio": float(np.mean(active)) if active.size else 0.0,
        "mean_rms": float(np.mean(values)) if values.size else 0.0,
        "windows": [
            {"start_seconds": i * frame_seconds, "rms": float(v), "active": bool(a)}
            for i, (v, a) in enumerate(zip(values, active))
        ],
    }


def make_suggestions(visual_activity: float | None, audio_activity: float | None) -> list[str]:
    """Transparent local rules; these are prompts for human review, not decisions."""
    suggestions = []
    if visual_activity is not None and visual_activity < 0.03:
        suggestions.append("Movement was low in sampled frames. Consider a brief check-in or an interactive question; low movement does not necessarily mean low attention.")
    elif visual_activity is not None and visual_activity > 0.20:
        suggestions.append("There was frequent visual change. Review the relevant segment to determine whether it reflects discussion, movement, or camera noise.")
    if audio_activity is not None and audio_activity < 0.08:
        suggestions.append("Audio activity was limited. If appropriate, invite questions or a short peer discussion; quiet listening may still be engaged learning.")
    elif audio_activity is not None and audio_activity > 0.60:
        suggestions.append("Audio activity was frequent. Review the session context to distinguish productive discussion from background noise.")
    if not suggestions:
        suggestions.append("No strong rule-based signal was detected. Review the session timeline and use instructor judgment before changing teaching strategy.")
    suggestions.append("Treat all indicators as rough proxies and confirm any interpretation with classroom context.")
    return suggestions
