"""Per-point feature extraction with the GeoPT pre-trained encoder."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

from .config import GEOPT_8LAYERS, FEATURE_DIM, ExtractConfig
from .preprocess import PreparedSample, pad_batch, prepare_sample


class GeoPTFeatureExtractor:
    """Maps ``(coordinates, 0/1 mask)`` to per-point GeoPT features.

    The encoder is fully point-count agnostic: ``Physics_Attention_Irregular_Mesh``
    projects N points onto ``slice_num`` physical states, attends among those
    states and de-slices back to N, so nothing in the forward pass depends on N.
    Variable-length clouds are therefore batched by right-padding and masking.
    """

    def __init__(self, cfg: ExtractConfig, root: Optional[str] = None):
        self.cfg = cfg
        self.root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == 'cpu' else 'cpu')
        self.dtype = torch.float16 if cfg.dtype == 'float16' else torch.float32
        self.model = self._build_model()

    # ------------------------------------------------------------------ setup
    def _build_model(self):
        from models.model_factory import get_model

        model = get_model(GEOPT_8LAYERS)
        path = self.cfg.checkpoint
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'GeoPT checkpoint not found: {path}')

        state = torch.load(path, map_location='cpu')
        # Strict on purpose: a silently partial load would produce features from
        # randomly initialized layers, which is far worse than a hard failure.
        model.load_state_dict(state)
        print(f'[GeoPT] loaded {len(state)} tensors from {path}')

        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        return model

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def encode_prepared(self, samples: Sequence[PreparedSample]) -> list[np.ndarray]:
        """Run the encoder on already-prepared samples, returning one array each."""
        x_np, fx_np, mask_np = pad_batch(list(samples))
        x = torch.from_numpy(x_np).to(device=self.device, dtype=self.dtype)
        fx = torch.from_numpy(fx_np).to(device=self.device, dtype=self.dtype)
        mask = torch.from_numpy(mask_np).to(device=self.device, dtype=self.dtype)

        _, point_feature = self.model(x, fx, mask=mask, return_point_feature=True)

        out = []
        for i, sample in enumerate(samples):
            rows = torch.from_numpy(sample.export_index).to(self.device)
            out.append(point_feature[i].index_select(0, rows).float().cpu().numpy())
        return out

    def extract(
            self,
            points: np.ndarray,
            mask: Optional[np.ndarray] = None,
            normals: Optional[np.ndarray] = None,
            sdf: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, PreparedSample]:
        """Extract features for one cloud.

        Returns ``(features, prepared)`` where ``features`` has shape
        ``(mask.sum(), FEATURE_DIM)`` and rows follow the ascending order of the
        selected point indices.
        """
        prepared = prepare_sample(points, mask, self.cfg, normals=normals, sdf=sdf)
        features = self.encode_prepared([prepared])[0]
        assert features.shape[1] == FEATURE_DIM
        return features, prepared
