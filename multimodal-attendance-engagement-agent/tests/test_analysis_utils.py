import unittest
import numpy as np
from app.analysis_utils import summarize_presence, audio_energy_summary, make_suggestions


class AnalysisUtilsTests(unittest.TestCase):
    def test_presence_summary(self):
        result = summarize_presence([0, 1, 0, 2, 1], minimum_observations=3)
        self.assertEqual(result.total_samples, 5)
        self.assertEqual(result.positive_samples, 3)
        self.assertAlmostEqual(result.presence_ratio, 0.6)
        self.assertTrue(result.sustained_presence)

    def test_empty_presence(self):
        result = summarize_presence([], minimum_observations=2)
        self.assertEqual(result.total_samples, 0)
        self.assertEqual(result.presence_ratio, 0.0)
        self.assertFalse(result.sustained_presence)

    def test_audio_summary(self):
        audio = np.zeros(16000, dtype=np.float32)
        result = audio_energy_summary(audio, 16000)
        self.assertEqual(result["duration_seconds"], 1.0)
        self.assertEqual(result["activity_ratio"], 0.0)

    def test_suggestions_always_include_caution(self):
        suggestions = make_suggestions(0.01, 0.02)
        self.assertTrue(any("rough proxies" in s for s in suggestions))


if __name__ == "__main__":
    unittest.main()
