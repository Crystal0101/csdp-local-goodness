"""Portable real-source goodness checks; invoke in a separate Python process."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,traceback
import numpy as np
D=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,default=D/'results/grouped_csdp_replication')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();study=args.study.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Refusing existing output: {out}')
    study_protocol=json.loads((study/'protocol.json').read_text())
    assert sha(study/'source_manifest.json')==study_protocol['source_manifest_sha256']
    source=study/'source';manifest=json.loads((study/'source_manifest.json').read_text())
    for name,digest in manifest.items():
        path=(source/name).resolve();assert path.is_relative_to(source.resolve())
        assert sha(path)==digest,name
    module_path=source/'custom/goodnessModCell.py'
    out.mkdir(parents=True,exist_ok=False)
    config={'widths':[64,256],'batches':[1,2],'groups':[1,8],'theta':10.,
        'target_logit':.5,'perturbation':.001,'support_tolerance':1e-10,
        'analytic_atol':1e-10,'analytic_rtol':1e-9}
    protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_sha256':sha(Path(__file__)),'study_protocol_sha256':sha(study/'protocol.json'),
        'source_sha256':sha(module_path),'config':config,
        'scope':'Instantaneous goodness only; no dataset, training or whole-network locality check',
        'G1_reference':'Original calc_mod_signal in the same frozen module, not a fresh download of author source'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    try:
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        spec=importlib.util.spec_from_file_location('portable_grouped_goodness',module_path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        records=[];reference_checks=[]
        for n in config['widths']:
            for b in config['batches']:
                z=jnp.full((b,n),np.sqrt(10.5/n),dtype=jnp.float64)
                labels=jnp.asarray((np.arange(b)%2==0).astype(float).reshape(b,1))
                original=mod.calc_mod_signal(z,labels,10.,False)
                grouped1=mod.grouped_mod_signal(z,labels,10.,1)
                assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(original,grouped1))
                reference_checks.append({'N':n,'B':b,'same_module_original_G1_bitwise_equal':True})
                for g in config['groups']:
                    fn=lambda trace:mod.grouped_mod_signal(trace,labels,10.,g)[1]
                    gradient=np.asarray(fn(z));jacobian=np.asarray(jax.jacfwd(fn)(z))
                    probability=1/(1+np.exp(-.5))
                    expected_gradient=2*np.asarray(z)*(probability-np.asarray(labels))/b
                    assert np.allclose(gradient,expected_gradient,atol=1e-10,rtol=1e-9)
                    expected=np.zeros_like(jacobian);support=np.zeros_like(jacobian,dtype=bool)
                    width=n//g
                    for sample in range(b):
                        for group in range(g):
                            ids=np.arange(group*width,(group+1)*width)
                            cross=4*g/b*float(z[sample,0])**2*probability*(1-probability)
                            expected[sample,ids[:,None],sample,ids[None,:]]=cross
                            expected[sample,ids,sample,ids]+=2*(probability-float(labels[sample,0]))/b
                            support[sample,ids[:,None],sample,ids[None,:]]=True
                    observed=np.abs(jacobian)>config['support_tolerance']
                    assert np.array_equal(observed,support)
                    assert np.allclose(jacobian,expected,atol=1e-10,rtol=1e-9)
                    index=width+1 if g>1 else n//2
                    delta=np.asarray(fn(z.at[0,index].add(.001))-fn(z))
                    affected=np.abs(delta)>config['support_tolerance']
                    wanted=np.zeros((b,n),bool);group=index//width
                    wanted[0,group*width:(group+1)*width]=True
                    assert np.array_equal(affected,wanted)
                    records.append({'N':n,'B':b,'G':g,'support_per_modulator':width,
                        'gradient_max_abs_error':float(np.max(np.abs(gradient-expected_gradient))),
                        'jacobian_max_abs_error':float(np.max(np.abs(jacobian-expected))),
                        'exact_support_passed':True,'cross_batch_support_zero':True,
                        'affected_outputs_after_single_trace_perturbation':int(affected.sum()),
                        'perturbation_support_passed':True})
        result={'passed':True,'source_sha256':sha(module_path),'reference_checks':reference_checks,
            'records':records,'jax_version':jax.__version__,'float64':True,
            'scope':'Real frozen goodness source, non-saturated synthetic traces only. No training or dataset access.',
            'limits':['G1 comparison is original function retained in same module',
                'Whole-network recurrent/shared-label information routes remain',
                'No communication, memory, energy or accuracy result follows from this check']}
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'passed':True,'cases':len(records),'G1_reference_checks':len(reference_checks),'output':str(out)},indent=2))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
