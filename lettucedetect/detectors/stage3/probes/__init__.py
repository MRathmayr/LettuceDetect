"""Probe models for Stage 3 uncertainty quantification."""

from lettucedetect.detectors.stage3.probes.grounding_probe import GroundingProbe
from lettucedetect.detectors.stage3.probes.hidden_state_extractor import HiddenStateExtractor

__all__ = ["GroundingProbe", "HiddenStateExtractor"]
