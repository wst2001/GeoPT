"""Turn a raw point cloud + 0/1 mask into the 14-channel tensor GeoPT expects."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import ExtractConfig

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def permute_axes(array: np.ndarray, axis_order: str) -> np.ndarray:
    """Reorder the last dimension of an (N, 3) array.

    ``axis_order='xzy'`` maps a Z-up cloud onto the Y-up convention GeoPT uses
    (see the ``(x,y,z) -> (x,z,y)`` transform in
    ``data_preprocess/DrivAerML_process.py``).
    """
    if axis_order == 'xyz':
        return array
    if len(axis_order) != 3 or set(axis_order) != set('xyz'):
        raise ValueError(f"axis_order must be a permutation of 'xyz', got {axis_order!r}")
    perm = [_AXIS_INDEX[c] for c in axis_order]
    return array[:, perm]


def normalize_geometry(points: np.ndarray, target_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a cloud into GeoPT's normalized domain.

    Reproduces ``data_preprocess/DrivAerML_process.py::transform``: vertical axis
    (1) starts at zero, the length axis (0) is scaled to ``target_length`` and
    centred. The result is a single affine map, returned alongside the points so
    downstream consumers can invert it.

    Returns:
        normalized points (N, 3), scale (scalar array), offset (3,)
        such that ``normalized == points * scale + offset``.
    """
    bound_min = points.min(axis=0)
    bound_max = points.max(axis=0)

    length = float(bound_max[0] - bound_min[0])
    if length <= 1e-12:
        raise ValueError(f'degenerate extent along the length axis: {length}')
    scale = float(target_length / length)

    offset = np.zeros(3, dtype=np.float64)
    offset[1] = -scale * float(bound_min[1])
    offset[0] = -scale * float(points[:, 0].mean())

    normalized = points * scale + offset
    return normalized, np.asarray(scale, dtype=np.float64), offset


def estimate_normals(points: np.ndarray, k: int = 16, chunk: int = 200_000) -> np.ndarray:
    """Estimate unit normals by local PCA, oriented away from the centroid.

    The smallest-eigenvalue eigenvector of each point's kNN covariance is the
    local surface normal up to sign; PCA cannot recover that sign, so we resolve
    it by pointing every normal outward from the cloud's centroid. That is exact
    for star-convex shapes and a reasonable default otherwise.
    """
    from scipy.spatial import cKDTree

    points = np.ascontiguousarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)

    k = int(min(k, n - 1))
    if k < 2:
        return np.zeros((n, 3), dtype=np.float32)

    tree = cKDTree(points)
    normals = np.empty((n, 3), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # k + 1 because the query point is its own nearest neighbour.
        _, idx = tree.query(points[start:stop], k=k + 1, workers=-1)
        nb = points[idx]                                   # (chunk, k+1, 3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', nb, nb)
        _, eigvec = np.linalg.eigh(cov)                    # ascending eigenvalues
        normals[start:stop] = eigvec[:, :, 0]

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-12)

    outward = points - points.mean(axis=0, keepdims=True)
    flip = np.einsum('ni,ni->n', normals, outward) < 0.0
    normals[flip] *= -1.0
    return normals.astype(np.float32)


class PreparedSample:
    """A single cloud, ready for the encoder."""

    __slots__ = ('x', 'fx', 'export_index', 'point_index', 'points', 'scale', 'offset', 'num_encoded')

    def __init__(self, x, fx, export_index, point_index, points, scale, offset):
        self.x = x                        # (N, 3)  float32, encoder coordinates
        self.fx = fx                      # (N, 11) float32, encoder functions
        self.export_index = export_index  # (M,) int64, rows of x whose features we keep
        self.point_index = point_index    # (M,) int64, rows of the *input* array they came from
        self.points = points              # (M, 3) float32, original coords of those rows
        self.scale = scale
        self.offset = offset
        self.num_encoded = x.shape[0]


def prepare_sample(
        points: np.ndarray,
        mask: Optional[np.ndarray],
        cfg: ExtractConfig,
        normals: Optional[np.ndarray] = None,
        sdf: Optional[np.ndarray] = None,
) -> PreparedSample:
    """Build the encoder inputs for one point cloud.

    Args:
        points: (N, 3) coordinates.
        mask: (N,) 0/1 selector. ``None`` means every point is valid.
        normals: optional (N, 3) surface normals, overriding ``cfg.normal_source``.
        sdf: optional (N,) signed distances, overriding ``cfg.sdf_value``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points must have shape (N, 3), got {points.shape}')
    n_input = points.shape[0]
    if n_input == 0:
        raise ValueError('points is empty')

    if mask is None:
        keep = np.ones(n_input, dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1)
        if mask.shape[0] != n_input:
            raise ValueError(f'mask has {mask.shape[0]} entries but points has {n_input}')
        keep = mask.astype(bool)
    if not keep.any():
        raise ValueError('mask selects no points')

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != points.shape:
            raise ValueError(f'normals must have shape {points.shape}, got {normals.shape}')
    if sdf is not None:
        sdf = np.asarray(sdf, dtype=np.float64).reshape(-1)
        if sdf.shape[0] != n_input:
            raise ValueError(f'sdf has {sdf.shape[0]} entries but points has {n_input}')

    # With the default masking semantics, mask==0 points are invalid: drop them
    # up front so they influence neither the normalization statistics, nor the
    # normal estimation, nor the encoder.
    if cfg.encode_all_points:
        source_index = np.arange(n_input, dtype=np.int64)
    else:
        source_index = np.flatnonzero(keep).astype(np.int64)
    original_points = points[source_index].astype(np.float32)

    work = permute_axes(points[source_index], cfg.axis_order)
    if normals is not None:
        normals = permute_axes(normals[source_index], cfg.axis_order)

    if cfg.normalize_geometry:
        work, scale, offset = normalize_geometry(work, cfg.target_length)
    else:
        scale, offset = np.asarray(1.0, dtype=np.float64), np.zeros(3, dtype=np.float64)

    if normals is None:
        if cfg.normal_source == 'estimate':
            normals = estimate_normals(work, cfg.normal_neighbors)
        elif cfg.normal_source == 'zeros':
            normals = np.zeros_like(work)
        else:
            raise ValueError(f'unknown normal_source {cfg.normal_source!r}')

    if sdf is None:
        sdf_col = np.full((work.shape[0], 1), cfg.sdf_value, dtype=np.float64)
    else:
        sdf_col = sdf[source_index][:, None]

    direction = np.asarray(cfg.dynamics_direction, dtype=np.float64).reshape(1, 3)
    direction = np.repeat(direction, work.shape[0], axis=0)
    magnitude = np.full((work.shape[0], 1), cfg.dynamics_magnitude, dtype=np.float64)

    x = work.astype(np.float32)
    # Channel order must match the pre-trained preprocess layer, see config.py.
    fx = np.concatenate([work, sdf_col, normals, direction, magnitude], axis=-1).astype(np.float32)

    if cfg.encode_all_points:
        export_index = np.flatnonzero(keep).astype(np.int64)
    else:
        export_index = np.arange(source_index.shape[0], dtype=np.int64)
    # Where each exported feature came from in the caller's original array.
    point_index = source_index[export_index]

    return PreparedSample(
        x=x,
        fx=fx,
        export_index=export_index,
        point_index=point_index,
        points=original_points[export_index],
        scale=scale,
        offset=offset,
    )


def pad_batch(samples: list[PreparedSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad a list of variable-length samples into a dense batch.

    Returns ``(x, fx, mask)`` with shapes ``(B, N_max, 3)``, ``(B, N_max, 11)``
    and ``(B, N_max)``. Padded slots are marked invalid in ``mask`` so the
    encoder ignores them.
    """
    b = len(samples)
    n_max = max(s.num_encoded for s in samples)
    x = np.zeros((b, n_max, 3), dtype=np.float32)
    fx = np.zeros((b, n_max, samples[0].fx.shape[1]), dtype=np.float32)
    mask = np.zeros((b, n_max), dtype=np.float32)
    for i, sample in enumerate(samples):
        n = sample.num_encoded
        x[i, :n] = sample.x
        fx[i, :n] = sample.fx
        mask[i, :n] = 1.0
    return x, fx, mask
