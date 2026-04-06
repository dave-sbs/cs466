"""
interpretability.py — Line-level attribution for CLIP retrieval.

Given a stanza (joined with " / ") and a set of retrieved image embeddings,
computes per-line cosine similarity scores showing which lines drove each match.
"""

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


def encode_line(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    device: str,
) -> np.ndarray:
    """Encode a single line of text with CLIP, returning an L2-normalized embedding."""
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        features = model.text_projection(text_outputs.pooler_output)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy().astype("float32").squeeze(0)


def compute_line_attributions(
    model: CLIPModel,
    processor: CLIPProcessor,
    stanza_text: str,
    image_embeddings: np.ndarray,
    device: str,
) -> tuple[list[list[float]], list[str]]:
    """
    Compute per-line similarity scores against each retrieved image.

    Args:
        model: Loaded CLIPModel
        processor: Loaded CLIPProcessor
        stanza_text: Stanza text with lines joined by " / "
        image_embeddings: Array of shape [K, 512] — L2-normalized image embeddings
        device: torch device string

    Returns:
        line_scores: list of K lists, each containing per-line cosine similarities
                     (line_scores[image_idx][line_idx])
        line_texts: list of individual line strings
    """
    line_texts = [line.strip() for line in stanza_text.split(" / ") if line.strip()]

    if not line_texts:
        return [], []

    # Encode each line independently
    line_embeddings = np.stack(
        [encode_line(model, processor, line, device) for line in line_texts]
    )  # shape [num_lines, 512]

    # Dot product (cosine sim since both are L2-normalized)
    # image_embeddings: [K, 512], line_embeddings: [num_lines, 512]
    # Result: [K, num_lines]
    scores_matrix = image_embeddings @ line_embeddings.T

    line_scores = scores_matrix.tolist()
    return line_scores, line_texts


def normalize_scores(line_scores: list[list[float]]) -> list[list[float]]:
    """
    Min-max normalize per image so the strongest line → 1.0, weakest → 0.0.

    Args:
        line_scores: list of K lists, each with per-line raw cosine similarities

    Returns:
        Normalized scores in the same shape, values in [0, 1].
    """
    normalized = []
    for scores in line_scores:
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s
        if span > 0:
            normalized.append([(s - min_s) / span for s in scores])
        else:
            normalized.append([1.0] * len(scores))
    return normalized
