"""Cache a neutral SMPL surface fitted to one demo panel."""

import argparse
import json
from itm.demo.mesh_worker import fit_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    metadata = fit_mesh(args.motion, args.model, args.output, args.device)
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
