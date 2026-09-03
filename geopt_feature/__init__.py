"""Per-point feature extraction with the GeoPT pre-trained encoder.

Typical use::

    from geopt_feature import ExtractConfig, GeoPTFeatureExtractor

    extractor = GeoPTFeatureExtractor(ExtractConfig())
    features, _ = extractor.extract(points, mask)   # (mask.sum(), 256)
"""

from .config import FEATURE_DIM, GEOPT_8LAYERS, GEOPT_TARGET_LENGTH, ExtractConfig
from .extractor import GeoPTFeatureExtractor
from .preprocess import estimate_normals, normalize_geometry, pad_batch, prepare_sample

__all__ = [
    'ExtractConfig',
    'FEATURE_DIM',
    'GEOPT_8LAYERS',
    'GEOPT_TARGET_LENGTH',
    'GeoPTFeatureExtractor',
    'estimate_normals',
    'normalize_geometry',
    'pad_batch',
    'prepare_sample',
]
