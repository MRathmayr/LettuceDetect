"""Factory function for creating detector instances."""

from __future__ import annotations

import logging

from lettucedetect.detectors.base import BaseDetector

__all__ = ["make_detector"]

logger = logging.getLogger(__name__)


def make_detector(method: str, **kwargs) -> BaseDetector:
    """Create a detector of the requested type with the given parameters.

    :param method: One of "transformer", "llm", "rag_fact_checker", or "cascade".
    :param kwargs: Passed to the concrete detector constructor.
    :return: A concrete detector instance.
    :raises ValueError: If method is not supported.
    """
    if method == "transformer":
        augmentations = kwargs.pop("augmentations", None)
        if augmentations:
            # Stage1Detector wraps TransformerDetector with augmentations
            from lettucedetect.detectors.stage1 import Stage1Detector

            return Stage1Detector(augmentations=augmentations, **kwargs)
        from lettucedetect.detectors.transformer import TransformerDetector

        return TransformerDetector(**kwargs)
    elif method == "llm":
        from lettucedetect.detectors.llm import LLMDetector

        return LLMDetector(**kwargs)
    elif method == "rag_fact_checker":
        from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector

        return RAGFactCheckerDetector(**kwargs)
    elif method == "cascade":
        from lettucedetect.configs import CascadeConfig
        from lettucedetect.detectors.cascade import CascadeDetector

        config = kwargs.get("config")
        config_path = kwargs.get("config_path")

        if config_path:
            config = CascadeConfig.from_json(config_path)
            logger.info(f"Loaded cascade config from {config_path}")
        elif isinstance(config, dict):
            config = CascadeConfig.model_validate(config)
        elif config is None:
            raise ValueError(
                "cascade method requires 'config' (CascadeConfig or dict) or 'config_path'"
            )

        # Log warning for missing stage configs
        for stage_num in config.stages:
            stage_key = f"stage{stage_num}"
            stage_config = getattr(config, stage_key)
            if stage_config == getattr(CascadeConfig(), stage_key):
                logger.warning(f"No {stage_key} config provided, using defaults")

        return CascadeDetector(config=config)
    else:
        raise ValueError(
            f"Unknown detector method: {method}. "
            f"Valid: transformer, llm, rag_fact_checker, cascade"
        )
