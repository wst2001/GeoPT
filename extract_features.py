"""Export per-point GeoPT features for a directory of point clouds.

Input: one ``.npz`` per sample, containing
    points   (N, 3)  float   required -- point coordinates
    mask     (N,)    0/1     optional -- which points to export, default all ones
    normals  (N, 3)  float   optional -- overrides --normal-source
    sdf      (N,)    float   optional -- overrides --sdf-value

Output: one ``.npz`` per sample in --output, containing
    features     (M, 256) float32 -- per-point features from the pre-trained encoder
    point_index  (M,)     int64   -- row in the input ``points`` each feature belongs to
    points       (M, 3)   float32 -- original coordinates of those rows
plus a ``manifest.json`` recording the exact configuration used.

Example::

    python extract_features.py --input ./my_clouds --output ./results/geopt_features
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export per-point GeoPT features for point clouds with a 0/1 mask.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', required=True, help='directory of .npz point clouds, or a single .npz file')
    parser.add_argument('--output', required=True, help='directory to write feature .npz files into')
    parser.add_argument('--checkpoint', default='checkpoints/GeoPT_8layers.pt',
                        help='GeoPT pre-trained weights, relative to the GeoPT root')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16'])
    parser.add_argument('--batch-size', type=int, default=1,
                        help='samples per forward pass; variable lengths are padded and masked')

    geo = parser.add_argument_group('geometry alignment')
    geo.add_argument('--no-normalize-geometry', action='store_true',
                     help='feed raw coordinates instead of mapping them into GeoPT\'s normalized domain')
    geo.add_argument('--axis-order', default='xyz',
                     help="axis permutation applied before normalization; use 'xzy' for Z-up data")
    geo.add_argument('--target-length', type=float, default=5.0,
                     help='extent along axis 0 after normalization')

    chan = parser.add_argument_group('input channels')
    chan.add_argument('--normal-source', default='estimate', choices=['estimate', 'zeros'],
                      help='how to fill the normal channels when the input has no normals')
    chan.add_argument('--normal-neighbors', type=int, default=16, help='kNN size for normal estimation')
    chan.add_argument('--sdf-value', type=float, default=0.0,
                      help='value for the signed-distance channel; 0 is correct for surface points')
    chan.add_argument('--dynamics-direction', default='1,0,0',
                      help='comma-separated unit vector used as the dynamics prompt')
    chan.add_argument('--dynamics-magnitude', type=float, default=0.3,
                      help='scalar magnitude channel of the dynamics prompt')

    mask = parser.add_argument_group('masking')
    mask.add_argument('--encode-all-points', action='store_true',
                      help='let mask==0 points shape the geometry instead of discarding them; '
                           'features are still exported only for mask==1 points')

    parser.add_argument('--overwrite', action='store_true', help='recompute samples that already have output')
    return parser.parse_args()


def build_config(args):
    from geopt_feature import ExtractConfig

    direction = tuple(float(v) for v in args.dynamics_direction.split(','))
    if len(direction) != 3:
        raise ValueError(f'--dynamics-direction needs 3 comma-separated values, got {args.dynamics_direction!r}')
    return ExtractConfig(
        checkpoint=args.checkpoint,
        normalize_geometry=not args.no_normalize_geometry,
        axis_order=args.axis_order,
        target_length=args.target_length,
        normal_source=args.normal_source,
        normal_neighbors=args.normal_neighbors,
        sdf_value=args.sdf_value,
        dynamics_direction=direction,
        dynamics_magnitude=args.dynamics_magnitude,
        encode_all_points=args.encode_all_points,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )


def list_inputs(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f'input not found: {path}')
    files = sorted(f for f in os.listdir(path) if f.endswith('.npz'))
    if not files:
        raise FileNotFoundError(f'no .npz files in {path}')
    return [os.path.join(path, f) for f in files]


def read_sample(path: str):
    with np.load(path) as data:
        if 'points' not in data:
            raise KeyError(f"{path}: missing required array 'points'")
        points = data['points']
        mask = data['mask'] if 'mask' in data else None
        normals = data['normals'] if 'normals' in data else None
        sdf = data['sdf'] if 'sdf' in data else None
    return points, mask, normals, sdf


def main():
    args = parse_args()
    # Must precede the first torch import for the device selection to take effect.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)

    from geopt_feature import GeoPTFeatureExtractor
    from geopt_feature.preprocess import prepare_sample

    cfg = build_config(args)
    paths = list_inputs(args.input)
    os.makedirs(args.output, exist_ok=True)

    extractor = GeoPTFeatureExtractor(cfg)

    manifest = {'config': cfg.as_dict(), 'feature_dim': None, 'samples': []}
    pending: list[tuple[str, str, object]] = []

    def flush():
        if not pending:
            return
        features = extractor.encode_prepared([item[2] for item in pending])
        for (name, out_path, prepared), feat in zip(pending, features):
            np.savez(
                out_path,
                features=feat,
                point_index=prepared.point_index,
                points=prepared.points,
                geometry_scale=prepared.scale,
                geometry_offset=prepared.offset,
            )
            manifest['feature_dim'] = int(feat.shape[1])
            manifest['samples'].append({
                'name': name,
                'output': os.path.basename(out_path),
                'num_encoded_points': int(prepared.num_encoded),
                'num_exported_points': int(feat.shape[0]),
            })
            print(f'  {name}: encoded {prepared.num_encoded} points -> features {feat.shape}')
        pending.clear()

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output, f'{name}_features.npz')
        if os.path.exists(out_path) and not args.overwrite:
            print(f'  {name}: exists, skipping (use --overwrite to recompute)')
            continue

        points, mask, normals, sdf = read_sample(path)
        prepared = prepare_sample(points, mask, cfg, normals=normals, sdf=sdf)
        pending.append((name, out_path, prepared))
        # Padding to the longest cloud in a batch wastes compute when lengths vary
        # a lot, so keep batches small and flush eagerly.
        if len(pending) >= cfg.batch_size:
            flush()
    flush()

    manifest_path = os.path.join(args.output, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'wrote {len(manifest["samples"])} feature files + {manifest_path}')


if __name__ == '__main__':
    main()
