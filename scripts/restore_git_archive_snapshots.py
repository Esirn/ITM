#!/usr/bin/env python
"""Restore ignored duplicate snapshots from Git-versioned canonical files.

Default is a dry run; existing differing files are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def restore(root,entries,apply=False):
    root=Path(root).resolve()
    actions=[]
    for e in entries:
        source=(root/e['canonical_path']).resolve()
        target=(root/e['archive_path']).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root) or source==target:
            raise ValueError('Snapshot paths must be distinct and inside this checkout')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=e['sha256']:
            raise ValueError('Canonical source SHA mismatch: '+e['canonical_path'])
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=e['sha256']:
                raise ValueError('Existing snapshot differs: '+e['archive_path'])
            action='already_present'
        else:action='copy' if apply else 'would_copy'
        actions.append(dict(path=e['archive_path'],action=action,source=str(source),target=str(target)))
    # Validate the entire inventory before any write.
    if apply:
        for a in actions:
            if a['action']=='copy':
                target=Path(a['target']);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as dst,Path(a['source']).open('rb') as src:
                    shutil.copyfileobj(src,dst)
    return actions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--inventory',type=Path,default=Path(__file__).resolve().parents[1]/'docs/itm_rework/version_control_snapshot_20261010_v1.json')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();d=json.loads(a.inventory.read_text());actions=restore(a.root,d['duplicate_snapshots'],a.apply)
    print(json.dumps(dict(total=len(actions),already_present=sum(x['action']=='already_present' for x in actions),
        missing=sum(x['action']!='already_present' for x in actions),applied=a.apply)))
