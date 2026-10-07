"""Run the fixed pair by default; --all explicitly requests a complete study."""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def call(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True,env={**os.environ,'MPLBACKEND':'Agg','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',choices=['primary','supplement'],default='primary')
    ap.add_argument('--all',action='store_true',help='40 primary or 50 supplementary models; otherwise one fixed pair')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--_seed',type=int,help=argparse.SUPPRESS)
    ap.add_argument('--_group',type=int,help=argparse.SUPPRESS)
    a=ap.parse_args()
    out=(a.output or ROOT/'fresh_runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    src=ROOT/'src';study=ROOT/'studies'/a.study;primary=ROOT/'studies/primary'
    if a._seed is not None:
        assert a.study=='primary' and a._seed in range(12001,12021) and a._group in (1,8)
        runner=src/'run_grouped_csdp_replication.py'
        spec=importlib.util.spec_from_file_location('frozen_primary',runner);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m.O=out/'training';m.S=m.O/'source';m.DATA=out/'data';m.verify()
        m.worker(a._group,a._seed);return
    if out.exists():raise FileExistsError('Use a new output directory: '+str(out))
    out.mkdir(parents=True)
    call(src/'prepare_grouped_data.py','--output',out/'data','--study',primary)
    mechanism='check_grouped_delivery_mechanism.py' if a.study=='primary' else 'check_grouped_supplement_mechanism.py'
    call(src/mechanism,'--study',study,'--output',out/'mechanism')
    if a.study=='supplement':
        call(src/'reproduce_grouped_supplement.py','--study',study,'--original-study',primary,
             '--runner',src/'run_grouped_supplement.py','--data',out/'data','--output',out/'training',
             '--prepare-only' if a.prepare_only else '--all' if a.all else '--first-pair')
    else:
        args=[src/'reproduce_grouped_pair.py','--study',study,'--runner',src/'run_grouped_csdp_replication.py','--data',out/'data','--output',out/'training']
        if a.all or a.prepare_only:args+=['--prepare-only']
        call(*args)
        if a.all and not a.prepare_only:
            # Keep each ngcsimlib model in its own process. No replacement seeds.
            for seed in range(12001,12021):
                for group in (1,8):
                    call(Path(__file__),'--output',out,'--_seed',seed,'--_group',group)
    print('Results:',out)
if __name__=='__main__':main()
