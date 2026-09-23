#!/usr/bin/env python3
"""Export selected four-way HY-Motion evaluation outputs for browser review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True,
                        help='motion_id:sensor:short_label; repeat for each review case')
    args = parser.parse_args()

    cases = []
    for value in args.case:
        try:
            motion_id, sensor, label = value.split(':', 2)
        except ValueError as error:
            raise ValueError(f'Invalid --case {value!r}; expected motion_id:sensor:label') from error
        if sensor not in ('head', 'wrists'):
            raise ValueError(f'Unsupported sensor {sensor!r}')
        outputs = {}
        metadata = None
        for mode in ('text_only', 'paired', 'zero', 'shuffled'):
            path = args.evaluation / f'{motion_id}_{sensor}_{mode}.npz'
            with np.load(path, allow_pickle=False) as data:
                outputs[mode] = np.asarray(data['joints'], dtype=np.float32).round(4).tolist()
                metadata = json.loads(data['metadata'].item())
        cases.append({
            'label': label,
            'motion_id': motion_id,
            'sensor': sensor,
            'caption': metadata['caption'],
            'frames': len(outputs['paired']),
            'outputs': outputs,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'fps': 30, 'cases': cases}, separators=(',', ':')), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
