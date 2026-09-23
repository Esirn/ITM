#!/usr/bin/env python3
"""Render a fixed-seed HY-Motion control comparison from evaluated outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from itm.visualization.skeleton import save_motion_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--motion-id', required=True)
    parser.add_argument('--sensor', choices=('head', 'wrists'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1 or 30 % args.stride:
        raise ValueError('stride must be a positive divisor of 30')
    motions, labels = [], []
    for mode in ('text_only', 'paired', 'zero', 'shuffled'):
        path = args.evaluation / f'{args.motion_id}_{args.sensor}_{mode}.npz'
        with np.load(path, allow_pickle=False) as data:
            motions.append(data['joints'][::args.stride])
            metadata = json.loads(data['metadata'].item())
        labels.append(mode)
    save_motion_comparison(args.output, motions, labels, metadata['caption'], fps=30 // args.stride)
    print(args.output)


if __name__ == '__main__':
    main()
