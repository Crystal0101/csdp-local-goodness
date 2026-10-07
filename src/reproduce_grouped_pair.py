"""Re-run the fixed first G1/G8 pair in a fresh directory using the frozen runner."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
import numpy as np
D=Path(__file__).resolve().parents[1];P=D.parent
SEED=12001;GROUPS=(1,8)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def verified_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Manifest path outside source')
    return path
def verify_inputs(study,data,runner):
    protocol=read(study/'protocol.json');manifest=read(study/'source_manifest.json')
    decision=read(study/'freeze_decision.json')
    assert sha(runner)==protocol['code_sha256'],'Frozen runner hash mismatch'
    assert sha(study/'source_manifest.json')==protocol['source_manifest_sha256']
    assert sha(study/'protocol.json')==decision['protocol_sha256']
    assert protocol['groups']==[1,8] and SEED in protocol['seeds']
    assert set(protocol['data_files'])=={'trainX.npy','trainY.npy','validX.npy','validY.npy'}
    for name,digest in manifest.items():assert sha(verified_path(study/'source',name))==digest,name
    for name,digest in protocol['data_files'].items():assert sha(data/name)==digest,name
    return protocol,manifest
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=P/'grouped_pair_reproduction')
    ap.add_argument('--data',type=Path,default=P/'references/upstream_csdp/data/digits_smoke')
    ap.add_argument('--study',type=Path,default=D/'results/grouped_csdp_replication')
    ap.add_argument('--runner',type=Path,default=D/'scripts/run_grouped_csdp_replication.py')
    ap.add_argument('--prepare-only',action='store_true',help='Verify and copy frozen files; do not train')
    ap.add_argument('--_worker-group',type=int,choices=GROUPS,help=argparse.SUPPRESS)
    args=ap.parse_args();out=args.output.resolve();data=args.data.resolve();study=args.study.resolve();runner=args.runner.resolve()
    if args._worker_group:
        # Separate OS processes prevent ngcsimlib same-name mutable context collisions.
        verify_inputs(out,data,runner)
        metadata=read(out/'reproduction.json');assert metadata['adapter_sha256']==sha(Path(__file__))
        spec=importlib.util.spec_from_file_location('frozen_grouped_replication',runner)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.O=out;module.S=out/'source';module.DATA=data
        module.verify() # Retain the exact frozen runner's own verification as well.
        module.worker(args._worker_group,SEED)
        return
    if out.exists():raise FileExistsError(f'Refusing existing output: {out}')
    protocol,manifest=verify_inputs(study,data,runner)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','source_manifest.json','freeze_decision.json'):
        shutil.copy2(study/name,out/name)
    for name in manifest:
        dest=verified_path(out/'source',name);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(verified_path(study/'source',name),dest)
    metadata={'seed':SEED,'groups':list(GROUPS),'epochs':protocol['epochs'],
        'adapter_sha256':sha(Path(__file__)),'runner_sha256':sha(runner),
        'study_protocol_sha256':sha(study/'protocol.json'),
        'scope':'Execution check of prespecified first pair; not a new20seed study or significance test',
        'test_data_access':False,'training_requested':not args.prepare_only}
    (out/'reproduction.json').write_text(json.dumps(metadata,indent=2))
    verify_inputs(out,data,runner)
    if args.prepare_only:
        print(json.dumps({'status':'prepared only; no training','output':str(out)}));return
    for group in GROUPS:
        command=[sys.executable,str(Path(__file__).resolve()),'--output',str(out),
                 '--data',str(data),'--study',str(study),'--runner',str(runner),
                 '--_worker-group',str(group)]
        with (out/f'G{group}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    a,b=[read(out/'runs'/f'G{g}_{SEED}'/'result.json') for g in GROUPS]
    assert a['initial_weight_hashes']==b['initial_weight_hashes']
    assert a['stream']==b['stream'] and a['weight_count']==b['weight_count']
    assert all(r['all_finite_checks_passed'] and r['inference_labels_zero'] for r in (a,b))
    summary={'status':'fixed pair rerun completed','seed':SEED,
       'G1_accuracy':a['final_validation_accuracy'],'G8_accuracy':b['final_validation_accuracy'],
       'difference_pp':100*(b['final_validation_accuracy']-a['final_validation_accuracy']),
       'initial_weights_explicit_streams_parameter_counts_matched':True,
       'scope':'one paired rerun; no accuracy-retention statistical decision from one seed'}
    (out/'pair_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
