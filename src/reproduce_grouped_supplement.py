#!/usr/bin/env python3
"""Portable frozen supplementary reproduction. Default: prepare only, NO training.
Use --first-pair for seed13001/G1,G8 at20epochs, or explicit --all for all50 models.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
D=Path(__file__).resolve().parents[1]
P=D.parent
GROUPS=(1,2,4,8,16)
SEEDS=tuple(range(13001,13011))
METADATA=('protocol.json','source_manifest.json','source_change_record.json','analysis_plan.json','freeze_decision.json')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def safe(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Manifest path outside source')
    return path

def verify_inputs(study,original,data,runner):
    protocol=read(study/'protocol.json');decision=read(study/'freeze_decision.json')
    manifest=read(study/'source_manifest.json')
    assert sha(runner)==protocol['code_sha256'],'Frozen runner hash mismatch'
    assert sha(study/'protocol.json')==decision['protocol_sha256'],'Decision/protocol mismatch'
    assert sha(study/'source_manifest.json')==protocol['source_manifest_sha256']
    assert sha(study/'source_change_record.json')==protocol['source_change_record_sha256']
    assert sha(study/'analysis_plan.json')==protocol['analysis_plan_sha256']==decision['analysis_plan_sha256']
    assert sha(original/'protocol.json')==protocol['original_protocol_sha256']
    assert sha(original/'source_manifest.json')==protocol['original_source_manifest_sha256']
    assert protocol['groups']==list(GROUPS) and protocol['seeds']==list(SEEDS)
    assert protocol['epochs_by_group']=={str(g):(20 if g in (1,8) else 5) for g in GROUPS}
    assert set(protocol['data_files'])=={'trainX.npy','trainY.npy','validX.npy','validY.npy'}
    for name,digest in protocol['data_files'].items():assert sha(safe(data,name))==digest,(name,'data hash')
    for name,digest in manifest.items():assert sha(safe(study/'source',name))==digest,(name,'source hash')
    passed=[]
    for result in sorted((study/'preflight').glob('*/comparison.json')):
        if sha(result)!=decision['preflight_comparison_sha256']:continue
        registration=read(result.parent/'protocol.json')
        assert read(result)['passed']
        assert registration['supplement_protocol_sha256']==sha(study/'protocol.json')
        assert registration['runner_sha256']==sha(runner)
        passed.append(result.parent)
    assert passed,'Reviewed historical preflight metadata missing'
    return protocol,manifest,passed

def import_frozen(runner,out,original,data):
    spec=importlib.util.spec_from_file_location('frozen_grouped_supplement',runner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.O=out;module.S=out/'source';module.DATA=data;module.ORIGINAL=original
    module.ORIGIN=original/'source'
    module.verify(require_decision=True)
    return module

def validate_results(out,tasks,mode):
    rows={(g,s):read(out/'runs'/f'G{g}_{s}'/'result.json') for g,s in tasks}
    for g,s in tasks:
        r=rows[g,s]
        assert r['group']==g and r['seed']==s and not r['is_preflight']
        assert r['epochs']==(20 if g in (1,8) else 5)
        assert r['all_finite_checks_passed'] and r['inference_labels_zero']
        assert len(r['epoch_results'])==r['epochs']
    for seed in sorted({s for _,s in tasks}):
        reference=rows[1,seed]
        for g,s in tasks:
            if s!=seed:continue
            row=rows[g,s]
            for k in ('initial_weight_hashes','initial_encoder_keys','labels','weight_count'):
                assert row[k]==reference[k],(seed,g,k)
            assert row['stream'][:5]==reference['stream'][:5]
            assert row['encoder_key_log'][:5]==reference['encoder_key_log'][:5]
        assert rows[8,seed]['stream']==reference['stream']
        assert rows[8,seed]['encoder_key_log']==reference['encoder_key_log']
    return {'status':'completed','mode':mode,'models':len(tasks),
      'all_paired_rng_weight_and_parameter_checks_passed':True,
      'rows':[{'group':g,'seed':s,'epochs':r['epochs'],
               'epoch5_accuracy':r['curves'][4]['validation_accuracy'],
               'last_accuracy':r['final_validation_accuracy']} for (g,s),r in rows.items()],
      'scope':('Fixed first pair execution check; NOT the complete ten-pair supplement or a statistical conclusion'
               if mode=='first-pair' else 'All50 fixed supplementary models rerun; apply the original supplementary analysis plan without selecting G')}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,default=D/'results/grouped_supplement')
    ap.add_argument('--original-study',type=Path,default=D/'results/grouped_csdp_replication')
    ap.add_argument('--data',type=Path,default=P/'references/upstream_csdp/data/digits_smoke')
    ap.add_argument('--runner',type=Path,default=D/'scripts/run_grouped_supplement.py')
    ap.add_argument('--output',type=Path,default=P/'grouped_supplement_reproduction')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true',help='Verify and copy only; this is also the default')
    mode.add_argument('--first-pair',action='store_true',help='Actually rerun seed13001 G1/G8, 20epochs each')
    mode.add_argument('--all',action='store_true',help='Actually rerun all50 fixed models, 550epochs, using2workers')
    ap.add_argument('--_worker-group',type=int,choices=GROUPS,help=argparse.SUPPRESS)
    ap.add_argument('--_worker-seed',type=int,choices=SEEDS,help=argparse.SUPPRESS)
    args=ap.parse_args()
    study=args.study.resolve();original=args.original_study.resolve();data=args.data.resolve()
    runner=args.runner.resolve();out=args.output.resolve()
    if args._worker_group is not None or args._worker_seed is not None:
        assert args._worker_group is not None and args._worker_seed is not None
        meta=read(out/'reproduction.json')
        assert meta['adapter_sha256']==sha(Path(__file__)),'Adapter changed since preparation'
        assert [args._worker_group,args._worker_seed] in meta['tasks']
        assert meta['mode'] in ('first-pair','all')
        frozen_runner=out/'frozen_runner.py'
        assert sha(frozen_runner)==meta['runner_sha256']
        verify_inputs(out,original,data,frozen_runner)
        module=import_frozen(frozen_runner,out,original,data)
        module.worker(args._worker_group,args._worker_seed)
        return
    if out.exists():raise FileExistsError('Refusing existing output: '+str(out))
    protocol,manifest,passed=verify_inputs(study,original,data,runner)
    out.mkdir(parents=True,exist_ok=False)
    for name in METADATA:shutil.copy2(study/name,out/name)
    for name in manifest:
        target=safe(out/'source',name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(safe(study/'source',name),target)
    for source_dir in passed:
        target=out/'preflight'/source_dir.name;target.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','comparison.json'):shutil.copy2(source_dir/name,target/name)
    frozen_runner=out/'frozen_runner.py';shutil.copy2(runner,frozen_runner)
    assert sha(frozen_runner)==protocol['code_sha256']
    action='all' if args.all else ('first-pair' if args.first_pair else 'prepare-only')
    tasks=([(g,s) for s in SEEDS for g in GROUPS] if action=='all' else
           ([(1,13001),(8,13001)] if action=='first-pair' else []))
    metadata={'mode':action,'tasks':tasks,'workers':2 if tasks else 0,
      'adapter_sha256':sha(Path(__file__)),'runner_sha256':sha(runner),
      'supplement_protocol_sha256':sha(study/'protocol.json'),
      'original_protocol_sha256':sha(original/'protocol.json'),
      'test_data_access':False,'training_requested':bool(tasks),
      'scope':'Frozen supplementary reproduction only; original study and main primary claim unchanged'}
    dump(out/'reproduction.json',metadata)
    verify_inputs(out,original,data,frozen_runner)
    import_frozen(frozen_runner,out,original,data) # Exact frozen verification, no model created.
    if not tasks:
        print(json.dumps({'status':'prepared only; no training','output':str(out)}));return
    logs=out/'logs';logs.mkdir(exist_ok=False)
    def execute(task):
        g,seed=task
        command=[sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--original-study',str(original),
                 '--data',str(data),'--_worker-group',str(g),'--_worker-seed',str(seed)]
        with (logs/f'G{g}_{seed}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
              env={**os.environ,'MPLBACKEND':'Agg','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        return task
    dump(out/'reproduction_progress.json',{'status':'running','total':len(tasks),'completed':0,'workers':2})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(execute,task):task for task in tasks}
        try:
            for completed,future in enumerate(as_completed(futures),1):
                group,seed=future.result()
                dump(out/'reproduction_progress.json',{'status':'running','total':len(tasks),'completed':completed,'last':[group,seed],'workers':2})
        except Exception:
            for future in futures:future.cancel()
            dump(out/'reproduction_progress.json',{'status':'failed; preserve outputs','total':len(tasks)})
            raise
    summary=validate_results(out,tasks,action)
    dump(out/'reproduction_summary.json',summary)
    dump(out/'reproduction_progress.json',{'status':'complete','total':len(tasks),'completed':len(tasks)})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
