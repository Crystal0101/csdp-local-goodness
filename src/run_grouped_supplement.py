"""Post-hoc fixed-budget supplement; does not replace primary analysis or select G.
Only the source group guard is broadened; original source/results are immutable.
--freeze snapshots code/source/data, --preflight checks historical predictions,
--all uses exactly two subprocess workers. No implicit freezing or running.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time,subprocess,traceback,shutil
import numpy as np
D=Path(__file__).resolve().parents[1];P=D.parent;O=D/'results/grouped_supplement';S=O/'source';ORIGINAL=D/'results/grouped_csdp_replication';ORIGIN=ORIGINAL/'source';DATA=P/'references/upstream_csdp/data/digits_smoke'
GROUPS=(1,2,4,8,16);SEEDS=tuple(range(13001,13011));NAMES=('W1','W2','V2','V2y','V3y','M1','M2','C2','C3','R1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def epochs_for(g):return 20 if g in (1,8) else 5

def freeze():
 if (O/'protocol.json').exists() or S.exists():raise RuntimeError('No frozen supplement overwrite')
 analysis_plan=json.loads((O/'analysis_plan.json').read_text())
 assert analysis_plan['groups']==list(GROUPS) and analysis_plan['seeds']==list(SEEDS)
 assert analysis_plan['epochs_by_group']=={str(g):epochs_for(g) for g in GROUPS}
 original_protocol=json.loads((ORIGINAL/'protocol.json').read_text())
 original_manifest=json.loads((ORIGINAL/'source_manifest.json').read_text())
 original_decision=json.loads((ORIGINAL/'freeze_decision.json').read_text())
 assert sha(ORIGINAL/'protocol.json')==original_decision['protocol_sha256']
 for f,h in original_manifest.items():assert sha(ORIGIN/f)==h,(f,'original source mismatch')
 for f,h in original_protocol['data_files'].items():assert sha(DATA/f)==h,(f,'original data mismatch')
 O.mkdir(parents=True,exist_ok=True)
 changed='custom/goodnessModCell.py'
 for f,h in original_manifest.items():
  dest=S/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ORIGIN/f,dest)
  if f==changed:
   text=dest.read_text()
   old='if group_count not in (1, 8) or n_units % group_count:'
   assert text.count(old)==1
   text=text.replace(old,'if group_count not in (1, 2, 4, 8, 16) or n_units % group_count:')
   # Guard diagnostic changes with the allowed values; all numerical code untouched.
   assert text.count('Only G=1 or fixed equal G=8 groups allowed')==1
   text=text.replace('Only G=1 or fixed equal G=8 groups allowed','Only fixed equal G=1/2/4/8/16 groups allowed')
   dest.write_text(text)
  else:assert sha(dest)==h
 manifest={f:sha(S/f) for f in original_manifest};dump(O/'source_manifest.json',manifest)
 dump(O/'source_change_record.json',{'original_manifest':original_manifest,'supplement_manifest':manifest,'changed_files':[changed],
  'change':'Only allowed-group guard and its diagnostic message; no numerical source changes'})
 protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'stage':'post-hoc supplementary sensitivity on reused data; never a replacement primary analysis or group selection',
  'groups':list(GROUPS),'seeds':list(SEEDS),'epochs_by_group':{str(g):epochs_for(g) for g in GROUPS},
  'model_count':50,'training_epoch_count':550,'architecture':[64,256,64,10],'batch':98,'effective_contrastive_batch':196,
  'T':50,'dt':3.,'constructor_eta_requested':.002,'actual_eta_w':.001,'actual_weight_decay':.00007,'learning_rate_basis':'Unchanged csdp_model.py constructor heuristic for raw batch98 overrides requested eta','learn_recon':True,'theta':10.,'partition':'fixed equal contiguous groups',
  'endpoints':'epoch5 for all groups; epochs5 and20 for G1/G8; all epoch curves/predictions retained, no best-epoch/group selection',
  'encoder_logging':'read-only z0.key.value and z3.key.value at creation and before training/before validation/after validation each epoch; no extra process or key reset',
  'original_protocol_sha256':sha(ORIGINAL/'protocol.json'),'original_source_manifest_sha256':sha(ORIGINAL/'source_manifest.json'),
  'data_files':{n:sha(DATA/n) for n in ('trainX.npy','trainY.npy','validX.npy','validY.npy')},
  'code_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(O/'source_manifest.json'),
  'source_change_record_sha256':sha(O/'source_change_record.json'),'analysis_plan_sha256':sha(O/'analysis_plan.json'),
  'preflight':'Before supplementary training: independent seed12001 G1/G8 five-epoch runs compared to historical predictions at all five epochs',
  'run_gate':'Separate root-reviewed freeze_decision.json required and must match protocol hash; statistical plan belongs to that reviewed decision',
  'forbidden':'test split, replacing old results, picking bestG, changing epochs/seeds in reaction to results',
  'caveats':['same historically reused split','post-hoc supplement, not preregistered original hypothesis','epoch20 need not imply convergence','only modulator dependence, no whole-network cost claim']}
 dump(O/'protocol.json',protocol);print('supplement frozen, not started',O/'protocol.json')

def verify(require_decision=False):
 pr=json.loads((O/'protocol.json').read_text());assert sha(Path(__file__))==pr['code_sha256'];assert sha(O/'source_manifest.json')==pr['source_manifest_sha256']
 assert sha(O/'source_change_record.json')==pr['source_change_record_sha256']
 assert sha(O/'analysis_plan.json')==pr['analysis_plan_sha256']
 for f,h in json.loads((O/'source_manifest.json').read_text()).items():assert sha(S/f)==h,(f,'supplement source mismatch')
 for n,h in pr['data_files'].items():assert sha(DATA/n)==h,(n,'supplement data mismatch')
 assert sha(ORIGINAL/'protocol.json')==pr['original_protocol_sha256']
 assert sha(ORIGINAL/'source_manifest.json')==pr['original_source_manifest_sha256']
 if require_decision:
  decision=json.loads((O/'freeze_decision.json').read_text())
  assert decision['protocol_sha256']==sha(O/'protocol.json'),'decision does not cover frozen protocol'
  approved_preflights=[]
  for result_path in (O/'preflight').glob('*/comparison.json'):
   registration=json.loads((result_path.parent/'protocol.json').read_text())
   result=json.loads(result_path.read_text())
   if registration['supplement_protocol_sha256']==sha(O/'protocol.json') and registration['runner_sha256']==sha(Path(__file__)) and result.get('passed'):
    approved_preflights.append(result_path)
  assert approved_preflights,'A passed preflight for this exact frozen runner is required'
 return pr

def worker(g,seed,preflight_dir=None):
 is_preflight=preflight_dir is not None
 verify(require_decision=not is_preflight)
 if is_preflight:
  assert seed==12001 and g in (1,8)
  preflight_dir=Path(preflight_dir).resolve()
  assert preflight_dir.is_relative_to((O/'preflight').resolve())
  assert (preflight_dir/'protocol.json').exists(),'preflight parent protocol missing'
  registration=json.loads((preflight_dir/'protocol.json').read_text())
  assert registration['supplement_protocol_sha256']==sha(O/'protocol.json') and registration['runner_sha256']==sha(Path(__file__))
  budget=5;out=preflight_dir/'runs'/f'G{g}_{seed}'
 else:
  assert g in GROUPS and seed in SEEDS
  budget=epochs_for(g);out=O/'runs'/f'G{g}_{seed}'
 out.mkdir(parents=True,exist_ok=False);os.chdir(out);start=time.perf_counter()
 try:
  sys.path.insert(0,str(S))
  from jax import random,numpy as jnp
  from csdp_model import CSDP_SNN
  data={n:jnp.asarray(np.load(DATA/(n+'.npy'))) for n in ('trainX','trainY','validX','validY')}
  key=random.PRNGKey(seed);key,*sub=random.split(key,4)
  model=CSDP_SNN(sub[0],in_dim=64,out_dim=10,hid_dim=256,hid_dim2=64,batch_size=98,eta=.002,T=50,dt=3.,algo_type='supervised',learn_recon=True,exp_dir='model',group_count=g,compensate=False,nonneg_w=False)
  # No extra initialization clip: preserve complete original native behavior.
  initial={n:ah(np.asarray(getattr(model,n).weights.value)) for n in NAMES}
  dump(out/'initial.json',{'group_count':g,'seed':seed,'weight_hashes':initial,'protocol_sha256':sha(O/'protocol.json'),'decision_sha256':None if is_preflight else sha(O/'freeze_decision.json'),'dependency_units':[256//g,64//g]})
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
  def encoder_keys():
   return {name:np.asarray(getattr(model,name).key.value).copy().tolist() for name in ('z0','z3')}
  initial_encoder_keys=encoder_keys()
  curves=[];batch_audits=[];streams=[];epoch_results=[];encoder_key_log=[]
  for ep in range(1,budget+1):
   epoch_keys={'epoch':ep,'before_training':encoder_keys()}
   key,pkey,*sub=random.split(key,3);perm=random.permutation(pkey,len(data['trainX']));X=data['trainX'][perm];Y=data['trainY'][perm]
   streams.append({'epoch':ep,'permutation_sha256':ah(np.asarray(perm)),'batch_keys':[]})
   for b in range(0,len(X),98):
    key,*sub=random.split(key,2);streams[-1]['batch_keys'].append(np.asarray(sub[0]).tolist())
    output,*_=model.process(X[b:b+98],Y[b:b+98],dkey=sub[0],adapt_synapses=True);assert np.isfinite(np.asarray(output)).all()
    batch_audits.append({'epoch':ep,'start':b,'finite':True,**audit()})
   epoch_keys['before_validation']=encoder_keys()
   probs=[];pred=[];zero_checks=[]
   for b in range(0,len(data['validX']),98):
    output,*_=model.process(data['validX'][b:b+98],data['validY'][b:b+98],dkey=key,adapt_synapses=False)
    a=np.asarray(output);assert np.isfinite(a).all();assert np.all(np.asarray(model.z3.inputs.value)==0),'labels not zero at inference'
    zero_checks.append(True);pred.extend(a.argmax(1).tolist());probs.extend(a.tolist())
   epoch_keys['after_validation']=encoder_keys();encoder_key_log.append(epoch_keys)
   labels=np.asarray(data['validY']).argmax(1);acc=float((np.asarray(pred)==labels).mean())
   curves.append({'epoch':ep,'validation_accuracy':acc});epoch_results.append({'epoch':ep,'predictions':pred,'probabilities':probs,'inference_labels_zero':zero_checks})
   dump(out/'progress.json',{'group':g,'seed':seed,'completed_epoch':ep,'curves':curves,'elapsed':time.perf_counter()-start});print('G'+str(g),seed,ep,acc,flush=True)
  r={'group':g,'seed':seed,'epochs':budget,'is_preflight':is_preflight,'initial_encoder_keys':initial_encoder_keys,'encoder_key_log':encoder_key_log,'final_validation_accuracy':acc,'curves':curves,'epoch_results':epoch_results,'predictions':pred,'labels':labels.tolist(),'probabilities':probs,'initial_weight_hashes':initial,'stream':streams,'batch_audits':batch_audits,'all_finite_checks_passed':True,'inference_labels_zero':True,'weight_count':sum(np.asarray(getattr(model,n).weights.value).size for n in NAMES),'dependency_units':[256//g,64//g],'seconds':time.perf_counter()-start,'scope':'post-hoc fixed supplementary sensitivity; historical train/valid only; no test, best-checkpoint or group choice'}
  dump(out/'result.json',r)
 except Exception:(out/'failure.txt').write_text(traceback.format_exc());raise

def preflight():
 verify(require_decision=False)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 target=O/'preflight'/stamp;target.mkdir(parents=True,exist_ok=False)
 prereg={'seed':12001,'groups':[1,8],'epochs':5,'supplement_protocol_sha256':sha(O/'protocol.json'),
  'original_protocol_sha256':sha(ORIGINAL/'protocol.json'),'runner_sha256':sha(Path(__file__)),
  'historical_result_sha256':{str(g):sha(ORIGINAL/'runs'/f'G{g}_12001'/'result.json') for g in (1,8)},
  'endpoint':'Exact saved predictions and probability arrays at every epoch; identical explicit random streams and initial weights',
  'no_tuning':'Run once, retain any mismatch; this does not license additional supplementary runs'}
 dump(target/'protocol.json',prereg)
 for g in (1,8):
  with (target/f'G{g}.log').open('x') as log:
   subprocess.run([sys.executable,str(Path(__file__).resolve()),'--preflight-worker','--preflight-dir',str(target),'--group',str(g),'--seed','12001'],
    stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 comparisons=[]
 for g in (1,8):
  actual=json.loads((target/'runs'/f'G{g}_12001'/'result.json').read_text())
  historical_path=ORIGINAL/'runs'/f'G{g}_12001'/'result.json'
  assert sha(historical_path)==prereg['historical_result_sha256'][str(g)]
  historical=json.loads(historical_path.read_text())
  assert len(actual['epoch_results'])==len(historical['epoch_results'])==5
  checks={'group':g,'initial_weights_equal':actual['initial_weight_hashes']==historical['initial_weight_hashes'],
   'explicit_stream_equal':actual['stream']==historical['stream'],'labels_equal':actual['labels']==historical['labels'],
   'parameters_equal':actual['weight_count']==historical['weight_count'],
   'epoch_prediction_equal':[a['predictions']==b['predictions'] for a,b in zip(actual['epoch_results'],historical['epoch_results'])],
   'epoch_probability_equal':[a['probabilities']==b['probabilities'] for a,b in zip(actual['epoch_results'],historical['epoch_results'])],
   'curves_equal':actual['curves']==historical['curves']}
  checks['passed']=all(checks[k] for k in ('initial_weights_equal','explicit_stream_equal','labels_equal','parameters_equal','curves_equal')) and all(checks['epoch_prediction_equal']) and all(checks['epoch_probability_equal'])
  comparisons.append(checks)
 left=json.loads((target/'runs/G1_12001/result.json').read_text());right=json.loads((target/'runs/G8_12001/result.json').read_text())
 keymatch=left['initial_encoder_keys']==right['initial_encoder_keys'] and left['encoder_key_log']==right['encoder_key_log']
 result={'passed':all(c['passed'] for c in comparisons) and keymatch,'comparisons':comparisons,'encoder_boundaries_equal_across_G':keymatch,'historical_results_unchanged':True}
 dump(target/'comparison.json',result);print(json.dumps(result,indent=2))
 if not result['passed']:raise RuntimeError('Preflight mismatch retained; do not run supplementary experiment')

def all_runs():
 verify(require_decision=True)
 from concurrent.futures import ThreadPoolExecutor,as_completed
 logs=O/'logs';logs.mkdir(exist_ok=True)
 tasks=[]
 for seed in SEEDS:
  for g in GROUPS:
   target=O/'runs'/f'G{g}_{seed}'
   if (target/'result.json').exists():
    complete=json.loads((target/'result.json').read_text())
    assert complete['group']==g and complete['seed']==seed and complete['epochs']==epochs_for(g) and not complete['is_preflight']
    continue
   if target.exists():raise RuntimeError('Incomplete run preserved; explicit amendment required')
   if (logs/f'G{g}_{seed}.log').exists():raise RuntimeError('Existing log preserved; refusing overwrite')
   tasks.append((g,seed))
 def execute(task):
  g,seed=task
  with (logs/f'G{g}_{seed}.log').open('x') as f:
   subprocess.run([sys.executable,str(Path(__file__).resolve()),'--group',str(g),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True,
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  return task
 dump(O/'progress.json',{'status':'running','workers_max':2,'fixed_tasks':50,'fixed_training_epochs':550,'completed':50-len(tasks)})
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(execute,t):t for t in tasks}
  try:
   for fut in as_completed(futures):
    g,seed=fut.result();done=len(list((O/'runs').glob('*/result.json')))
    dump(O/'progress.json',{'status':'running','workers_max':2,'last_completed':[g,seed],'completed':done,'total':50})
    print('completed',g,seed,done,'/50',flush=True)
  except Exception:
   for fut in futures:fut.cancel()
   dump(O/'progress.json',{'status':'failed, pending cancelled; preserve existing results','completed':len(list((O/'runs').glob('*/result.json')))})
   raise
 rows={(g,s):json.loads((O/'runs'/f'G{g}_{s}'/'result.json').read_text()) for s in SEEDS for g in GROUPS}
 for seed in SEEDS:
  ref=rows[1,seed]
  for g in GROUPS:
   row=rows[g,seed]
   assert row['epochs']==epochs_for(g) and len(row['curves'])==epochs_for(g)
   for k in ['initial_weight_hashes','initial_encoder_keys','labels','weight_count']:assert row[k]==ref[k],(seed,g,k)
   assert row['stream'][:5]==ref['stream'][:5]
   assert row['encoder_key_log'][:5]==ref['encoder_key_log'][:5]
  assert rows[8,seed]['stream']==ref['stream']
  assert rows[8,seed]['encoder_key_log']==ref['encoder_key_log']
 summary={'status':'all 50 fixed post-hoc supplementary runs complete','rows':[{'group':r['group'],'seed':r['seed'],'epochs':r['epochs'],'epoch5_validation_accuracy':r['curves'][4]['validation_accuracy'],'last_validation_accuracy':r['final_validation_accuracy']} for r in rows.values()],
  'paired_initial_states_explicit_streams_encoder_keys_and_parameters_matched':True,'interpretation':'Supplementary sensitivity only; do not replace primary criterion or choose a best group'}
 dump(O/'summary.json',summary);dump(O/'progress.json',{'status':'complete','completed':50,'total':50});print(json.dumps(summary,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 modes=ap.add_mutually_exclusive_group()
 modes.add_argument('--freeze',action='store_true');modes.add_argument('--all',action='store_true');modes.add_argument('--preflight',action='store_true');modes.add_argument('--preflight-worker',action='store_true',help=argparse.SUPPRESS)
 ap.add_argument('--preflight-dir',type=Path,help=argparse.SUPPRESS)
 ap.add_argument('--group',type=int,choices=GROUPS);ap.add_argument('--seed',type=int);a=ap.parse_args()
 if a.freeze:freeze()
 elif a.all:all_runs()
 elif a.preflight:preflight()
 elif a.preflight_worker:
  if a.preflight_dir is None or a.group not in (1,8) or a.seed!=12001:ap.error('Invalid internal preflight specification')
  worker(a.group,a.seed,a.preflight_dir)
 elif a.group is not None and a.seed in SEEDS:worker(a.group,a.seed)
 else:ap.error('Choose --freeze, --preflight, --all, or a valid --group/--seed pair')
