"""Configuration for GeoPT per-point feature extraction.

The architecture constants below are fixed by the released pre-trained weights
(``checkpoints/GeoPT_8layers.pt``) and must not be changed: the checkpoint's
``preprocess.linear_pre.0.weight`` has shape ``(512, 14)``, i.e. the encoder
consumes exactly ``space_dim + fun_dim == 3 + 11 == 14`` input channels.

Channel layout expected by the pre-trained encoder (see exp/GeoPT_finetune.py,
where ``model(x[:, :, :3], cat(pos7, v4))`` is called):

    0:3    point coordinates                (space_dim, passed as ``x``)
    3:6    point coordinates, again         |
    6:7    signed distance, 0 on a surface  |
    7:10   surface normal                   | fun_dim == 11, passed as ``fx``
    10:13  dynamics direction (unit vector) |
    13:14  dynamics magnitude scalar        |
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Architecture of the released GeoPT 8-layer pre-trained model.
# Mirrors checkpoints/config.json -> models["8-layers"] and the --fun_dim 11 /
# --space_dim 3 used by every script under scripts/finetune/.
# ---------------------------------------------------------------------------
GEOPT_8LAYERS = SimpleNamespace(
    model='Transolver',
    geotype='unstructured',
    space_dim=3,
    fun_dim=11,
    out_dim=9,  # dynamics_trajectory_dim of the pre-training head
    n_hidden=256,
    n_heads=8,
    n_layers=8,
    mlp_ratio=2,
    slice_num=32,
    act='gelu',
    dropout=0.0,
    checkpoint=0,
    shapelist=None,
)

#: Dimensionality of the exported per-point feature vectors.
FEATURE_DIM = GEOPT_8LAYERS.n_hidden

#: GeoPT normalizes every geometry to this extent along the length axis.
GEOPT_TARGET_LENGTH = 5.0


@dataclass
class ExtractConfig:
    """Everything that influences the exported features.

    Persisted verbatim into the output manifest so a feature dump can always be
    traced back to the settings that produced it.
    """

    checkpoint: str = 'checkpoints/GeoPT_8layers.pt'

    # --- geometry alignment -------------------------------------------------
    # GeoPT is pre-trained in a normalized domain, so raw point clouds have to be
    # brought into it or the features are out of distribution.
    normalize_geometry: bool = True
    #: Permutation applied to the input axes before normalization. GeoPT expects
    #: axis 0 = length, axis 1 = up, axis 2 = lateral. Z-up data needs 'xzy'.
    axis_order: str = 'xyz'
    target_length: float = GEOPT_TARGET_LENGTH

    # --- input channels -----------------------------------------------------
    #: How to fill channels 7:10 when the input file has no ``normals`` array.
    normal_source: str = 'estimate'  # 'estimate' | 'zeros'
    normal_neighbors: int = 16
    #: Channel 6. Zero is correct for points sampled on a surface.
    sdf_value: float = 0.0
    #: Channels 10:13. The dynamics "prompt" GeoPT was pre-trained to condition
    #: on; the released downstream configs all feed a unit direction here.
    dynamics_direction: tuple = (1.0, 0.0, 0.0)
    #: Channel 13. 0.3 matches the default used by exp/dynamics_config.py's
    #: drivAerML prompt.
    dynamics_magnitude: float = 0.3

    # --- masking ------------------------------------------------------------
    #: False: mask==0 points are invalid and excluded from the encoder.
    #: True:  mask==0 points still shape the geometry, they are just not exported.
    encode_all_points: bool = False

    # --- runtime ------------------------------------------------------------
    device: str = 'cuda'
    batch_size: int = 1
    dtype: str = 'float32'  # 'float32' | 'float16'

    def as_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            out[key] = list(value) if isinstance(value, tuple) else value
        return out
