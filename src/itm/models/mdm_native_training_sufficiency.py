"""Native Q batch5 and trajectory batch4, frozen amendment only."""
from pathlib import Path
from itm.models.mdm_correction import ROOT,read,verify
from itm.models.mdm_sampling_alignment import AlignmentPackage
from itm.models.mdm_state_diagnostic import StatePackage
from itm.models.mdm_run_limits import sha256
from itm.models.mdm_training_sufficiency import (TrainingPackage as BasePackage,TrainingGuard as BaseGuard,
    SnapshotCache,BRANCHES,SNAPSHOTS,ledger,stop_rows,compact_json,read_receipt)

PACKAGE=ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v4/package.json'
PACKAGE_SHA='95283f4c9744c6624ca3d4e6f0d2a199427f85e50edb2c975039512208b0416a'
CAPS=dict(mdm=5184,encoder=432,caption=72)
CALL_CAPS=dict(mdm=1152,encoder=432,caption=24)


class TrainingPackage:
    def __init__(self,path=PACKAGE):
        self.path=Path(path).resolve()
        if sha256(self.path)!=PACKAGE_SHA:raise ValueError('Exact native sufficiency v3 required')
        self.files={Path(a['path']).stem:verify(a) for a in read(self.path)['artifacts']}
        self.source_paths=[verify(a) for a in read(self.files['sources'])]
        self.design=read(self.files['design']);self.common=dict(self.design['common'],environment=self.design['environment'])
        self.alignment=AlignmentPackage(verify(self.design['original_Q_package']))
        self.mean,self.std=self.alignment.mean,self.alignment.std
        self.states=read(self.files['states']);self.jobs=read(self.files['jobs']);self.checks=read(self.files['software_checks'])
        self.cases=read(self.files['input_cases']);self.snapshots=read(self.files['checkpoints'])
        if (len(self.states),len(self.jobs),len(self.checks),len(self.cases))!=(192,1152,2880,12):raise ValueError('Native population changed')
    inputs=StatePackage.inputs
    state=BasePackage.state
    saved=BasePackage.saved


class TrainingGuard(BaseGuard):
    def __init__(self,context,package,**kwargs):
        super().__init__(context,package,package_sha=PACKAGE_SHA,caps=CAPS,call_caps=CALL_CAPS,native_batches=True,**kwargs)
