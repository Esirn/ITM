"""User-authorized contended GPU admission, with raw read-only NVML query evidence."""
import csv
import io
import json
from pathlib import Path
import subprocess
import time

LIMITS=dict(minimum_free_MiB=4096,samples=3,sample_interval_seconds=2)


def preflight_gpu(uuid,*,evidence_dir,check=lambda:None,sleep=time.sleep):
    if not uuid.startswith('GPU-') or any(c not in 'GPU-0123456789abcdefABCDEF' for c in uuid):
        raise ValueError('Name one physical GPU UUID')
    evidence_dir=Path(evidence_dir);evidence_dir.mkdir(parents=True,exist_ok=False)
    report=dict(version='itm-contended-preflight-v1',gpu_uuid=uuid,limits=LIMITS,
                status='running',samples=[],compute_processes=[],queries_attempted=0)
    def publish():
        (evidence_dir/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def query(name,fields):
        check();report['queries_attempted']+=1;publish()
        args=['nvidia-smi','--id='+uuid,'--'+fields,'--format=csv,noheader,nounits']
        try:
            result=subprocess.run(args,capture_output=True,text=True,check=True,timeout=10)
        except subprocess.CalledProcessError as error:
            (evidence_dir/(name+'.stdout')).write_text(error.stdout or '')
            (evidence_dir/(name+'.stderr')).write_text(error.stderr or '')
            raise
        (evidence_dir/(name+'.stdout')).write_text(result.stdout)
        (evidence_dir/(name+'.stderr')).write_text(result.stderr)
        check();return list(csv.reader(io.StringIO(result.stdout)))
    publish()
    try:
        for index in range(LIMITS['samples']):
            if index:sleep(LIMITS['sample_interval_seconds']);check()
            rows=query(f'gpu_{index+1:02d}','query-gpu=uuid,memory.total,memory.free,utilization.gpu,utilization.memory')
            if len(rows)!=1 or len(rows[0])!=5 or rows[0][0].strip()!=uuid:
                raise RuntimeError('Shared GPU identity/shape preflight failed')
            try:total,free,gpu,memory=map(int,(v.strip() for v in rows[0][1:]))
            except ValueError as error:raise RuntimeError('Shared GPU resource telemetry unavailable') from error
            if not (0<=free<=total and 0<=gpu<=100 and 0<=memory<=100):
                raise RuntimeError('Invalid GPU resource telemetry')
            sample=dict(total_MiB=total,free_MiB=free,GPU_utilization_percent=gpu,memory_controller_utilization_percent=memory)
            report['samples'].append(sample);publish()
            if free<LIMITS['minimum_free_MiB']:raise RuntimeError('Shared GPU free memory below 4096 MiB')
            # Activity percentages are telemetry, not admission gates.
        rows=query('compute_processes','query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')
        for fields in rows:
            if not fields or not fields[0].strip():continue
            if len(fields)!=4 or fields[0].strip()!=uuid:raise RuntimeError('Unexpected compute-process query identity/shape')
            report['compute_processes'].append(dict(pid=int(fields[1].strip()),process_name=fields[2].strip(),used_memory_MiB=fields[3].strip()))
        report.update(status='passed',compute_process_count=len(report['compute_processes']))
        publish();return report
    except BaseException as error:
        report.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)));publish();raise
