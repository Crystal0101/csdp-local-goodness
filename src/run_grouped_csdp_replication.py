"""Locked complete-CSDP conditional seed replication, G=1/G=8; historical train/valid only.
Different processes isolate same-name ngcsimlib resolver registries.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time,subprocess,traceback,shutil
import numpy as np
D=Path(__file__).resolve().parents[1];P=D.parent;O=D/'results/grouped_csdp_replication';S=O/'source';ORIGIN=D/'results/grouped_csdp_development/source';DATA=P/'references/upstream_csdp/data/digits_smoke'
GROUPS=(1,8);SEEDS=tuple(range(12001,12021));NAMES=('W1','W2','V2','V2y','V3y','M1','M2','C2','C3','R1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def freeze():
 if (O/'protocol.json').exists():raise RuntimeError('No protocol overwrite')
 O.mkdir(parents=True,exist_ok=True)
 origin_manifest=json.loads((ORIGIN.parent/'source_manifest.json').read_text())
 for f,h in origin_manifest.items():
  assert sha(ORIGIN/f)==h;dest=S/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ORIGIN/f,dest)
 dump(O/'source_manifest.json',origin_manifest)
 protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'locked-method conditional new-seed replication on reused data; not independent-data confirmation','question':'On this fixed reused split, is the lower endpoint of the paired seed-difference 95 percent t interval above minus 1 percentage point?','groups':list(GROUPS),'seeds':list(SEEDS),'architecture':[64,256,64,10],'batch':98,'T':50,'dt':3.,'epochs':5,'learn_recon':True,'source':'grouped_csdp_scope G1 reproduces original one-batch complete learning bitwise; no Dale/no compensation','formula':'G*sum(z_group^2)-theta; mean over batch and groups BCE, unchanged theta=10; derivative 2z/B*(sigmoid-group-label). G1 exact original branch. Objective changes for G8.','partition':'fixed contiguous equal-size groups, no selection','dependency_units':{'G1':[256,64],'G8':[32,8]},'dependency_boundary':'only instantaneous goodness modulator; shared labels, recurrent connections and readout errors remain. No whole-network communication/energy claim.','endpoints':'validation accuracy epoch 5; all epoch curves retained; no checkpoint selection','primary_analysis':{'paired_difference':'G8 minus G1 final epoch validation accuracy percentage points','n':20,'interval':'two-sided Student t paired seed mean 95 percent','success':'lower endpoint > -1.0 pp and all integrity/mechanism checks passed','sensitivity':'paired-seed percentile bootstrap, fixed RNG 87123, 20000 resamples; not alternative primary decision'},'sample_size_rationale':'fixed 20, conservative assumed paired SD 1.5 pp gives t19 half-width about 0.70 pp; no guaranteed power','excluded_development_seeds':[951,952,953],'data_files':{n:sha(DATA/n) for n in ('trainX.npy','trainY.npy','validX.npy','validY.npy')},'forbidden':'test data, group/hyperparameter search, additional seeds or epochs based on results','code_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(O/'source_manifest.json'),'caveats':['historical validation reused','fixed 5-epoch budget not convergence','group/layer helper has published precedent, no novelty claim','uncertainty conditional on same reused dataset; no independent population evidence']}
 dump(O/'protocol.json',protocol);print('frozen',O/'protocol.json')
def verify():
 pr=json.loads((O/'protocol.json').read_text());assert sha(Path(__file__))==pr['code_sha256'];assert sha(O/'source_manifest.json')==pr['source_manifest_sha256']
 for f,h in json.loads((O/'source_manifest.json').read_text()).items():assert sha(S/f)==h
 for n,h in pr['data_files'].items():assert sha(DATA/n)==h
 return pr

def worker(g,seed):
 verify();assert (O/'freeze_decision.json').exists(),'Root decision required before training'
 out=O/'runs'/f'G{g}_{seed}';out.mkdir(parents=True,exist_ok=False);os.chdir(out);start=time.perf_counter()
 try:
  sys.path.insert(0,str(S))
  from jax import random,numpy as jnp
  from csdp_model import CSDP_SNN
  data={n:jnp.asarray(np.load(DATA/(n+'.npy'))) for n in ('trainX','trainY','validX','validY')}
  key=random.PRNGKey(seed);key,*sub=random.split(key,4)
  model=CSDP_SNN(sub[0],in_dim=64,out_dim=10,hid_dim=256,hid_dim2=64,batch_size=98,eta=.002,T=50,dt=3.,algo_type='supervised',learn_recon=True,exp_dir='model',group_count=g,compensate=False,nonneg_w=False)
  # No extra initialization clip: preserve complete original native behavior.
  initial={n:ah(np.asarray(getattr(model,n).weights.value)) for n in NAMES}
  dump(out/'initial.json',{'group_count':g,'seed':seed,'weight_hashes':initial,'protocol_sha256':sha(O/'protocol.json'),'decision_sha256':sha(O/'freeze_decision.json'),'dependency_units':[256//g,64//g]})
  def audit():
   for n in NAMES:assert np.isfinite(np.asarray(getattr(model,n).weights.value)).all(),n
   observations={}
   for i in (1,2):
    cell=getattr(model,'z'+str(i));trace=getattr(model,'tr'+str(i));gc=getattr(model,'g'+str(i))
    a=np.asarray(cell.s.value);r=np.asarray(trace.trace.value);m=np.asarray(gc.modulator.value)
    v=np.asarray(cell.v.value);thr=np.asarray(cell.thr.value)
    assert np.isfinite(v).all() and np.isfinite(thr).all()
    assert np.isfinite(a).all() and np.isfinite(r).all() and np.isfinite(m).all()
    observations['layer'+str(i)]={'last_step_spike_mean':float(a.mean()),'last_step_trace_mean':float(r.mean()),'last_step_modulator_l2':float(np.linalg.norm(m)),'last_step_v_min':float(v.min()),'last_step_v_max':float(v.max()),'last_step_thr_min':float(thr.min()),'last_step_thr_max':float(thr.max())}
   for name in model.lifs:
    cell=getattr(model,name)
    for field in ('v','thr'):
     if hasattr(cell,field): assert np.isfinite(np.asarray(getattr(cell,field).value)).all(), (name,field)
   return observations
  curves=[];batch_audits=[];streams=[];epoch_results=[]
  for ep in range(1,6):
   key,pkey,*sub=random.split(key,3);perm=random.permutation(pkey,len(data['trainX']));X=data['trainX'][perm];Y=data['trainY'][perm]
   streams.append({'epoch':ep,'permutation_sha256':ah(np.asarray(perm)),'batch_keys':[]})
   for b in range(0,len(X),98):
    key,*sub=random.split(key,2);streams[-1]['batch_keys'].append(np.asarray(sub[0]).tolist())
    output,*_=model.process(X[b:b+98],Y[b:b+98],dkey=sub[0],adapt_synapses=True);assert np.isfinite(np.asarray(output)).all()
    batch_audits.append({'epoch':ep,'start':b,'finite':True,**audit()})
   probs=[];pred=[];zero_checks=[]
   for b in range(0,len(data['validX']),98):
    output,*_=model.process(data['validX'][b:b+98],data['validY'][b:b+98],dkey=key,adapt_synapses=False)
    a=np.asarray(output);assert np.isfinite(a).all();assert np.all(np.asarray(model.z3.inputs.value)==0),'labels not zero at inference'
    zero_checks.append(True);pred.extend(a.argmax(1).tolist());probs.extend(a.tolist())
   labels=np.asarray(data['validY']).argmax(1);acc=float((np.asarray(pred)==labels).mean())
   curves.append({'epoch':ep,'validation_accuracy':acc});epoch_results.append({'epoch':ep,'predictions':pred,'probabilities':probs,'inference_labels_zero':zero_checks})
   dump(out/'progress.json',{'group':g,'seed':seed,'completed_epoch':ep,'curves':curves,'elapsed':time.perf_counter()-start});print('G'+str(g),seed,ep,acc,flush=True)
  r={'group':g,'seed':seed,'final_validation_accuracy':acc,'curves':curves,'epoch_results':epoch_results,'predictions':pred,'labels':labels.tolist(),'probabilities':probs,'initial_weight_hashes':initial,'stream':streams,'batch_audits':batch_audits,'all_finite_checks_passed':True,'inference_labels_zero':True,'weight_count':sum(np.asarray(getattr(model,n).weights.value).size for n in NAMES),'dependency_units':[256//g,64//g],'seconds':time.perf_counter()-start,'scope':'locked 20-seed conditional replication, historical train/valid only; no test or best-checkpoint choice'}
  dump(out/'result.json',r)
 except Exception:(out/'failure.txt').write_text(traceback.format_exc());raise

def all_runs():
 verify();assert (O/'freeze_decision.json').exists()
 decision=json.loads((O/'freeze_decision.json').read_text())
 assert decision['protocol_sha256']==sha(O/'protocol.json')
 from concurrent.futures import ThreadPoolExecutor,as_completed
 logs=O/'logs';logs.mkdir(exist_ok=True)
 tasks=[]
 for seed in SEEDS:
  for g in GROUPS:
   target=O/'runs'/f'G{g}_{seed}'
   if (target/'result.json').exists():continue
   if target.exists():raise RuntimeError('Incomplete run preserved; explicit amendment required')
   tasks.append((g,seed))
 def execute(task):
  g,seed=task
  with (logs/f'G{g}_{seed}.log').open('w') as f:
   subprocess.run([sys.executable,str(Path(__file__).resolve()),'--group',str(g),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  return task
 dump(O/'progress.json',{'status':'running','workers_max':2,'fixed_tasks':40,'completed':40-len(tasks)})
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(execute,t):t for t in tasks}
  try:
   for fut in as_completed(futures):
    g,seed=fut.result();done=len(list((O/'runs').glob('*/result.json')))
    dump(O/'progress.json',{'status':'running','workers_max':2,'last_completed':[g,seed],'completed':done,'total':40})
    print('completed',g,seed,done,'/40',flush=True)
  except Exception:
   for fut in futures:fut.cancel()
   dump(O/'progress.json',{'status':'failed, pending cancelled; preserve existing results','completed':len(list((O/'runs').glob('*/result.json')))})
   raise
 rows=[json.loads((O/'runs'/f'G{g}_{s}'/'result.json').read_text()) for s in SEEDS for g in GROUPS]
 for seed in SEEDS:
  a,b=[r for r in rows if r['seed']==seed];assert a['initial_weight_hashes']==b['initial_weight_hashes'];assert a['stream']==b['stream'];assert a['weight_count']==b['weight_count']
 summary={'status':'all locked conditional seed replication runs complete','rows':[{'group':r['group'],'seed':r['seed'],'validation_accuracy':r['final_validation_accuracy']} for r in rows],'initial_weights_explicit_streams_parameter_counts_matched':True,'interpretation':'Apply prespecified paired t interval; same reused data, not independent-data confirmation; no optional seeds'}
 dump(O/'summary.json',summary);dump(O/'progress.json',{'status':'complete','completed':40,'total':40});print(json.dumps(summary,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--all',action='store_true');ap.add_argument('--group',type=int,choices=GROUPS);ap.add_argument('--seed',type=int,choices=SEEDS);a=ap.parse_args()
 if a.freeze:freeze()
 elif a.all:all_runs()
 elif a.group and a.seed:worker(a.group,a.seed)
 else:ap.error('Choose action')
