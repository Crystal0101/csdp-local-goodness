"""Recreate only the four frozen public train/validation arrays, without test files."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
D=Path(__file__).resolve().parents[1];P=D.parent
FILES=('trainX.npy','trainY.npy','validX.npy','validY.npy')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=P/'grouped_reproduction_data')
    ap.add_argument('--study',type=Path,default=D/'results/grouped_csdp_replication')
    args=ap.parse_args();out=args.output.resolve();study=args.study.resolve()
    if out.exists():raise FileExistsError(f'Refusing existing output: {out}')
    protocol=json.loads((study/'protocol.json').read_text());expected=protocol['data_files']
    assert set(expected)==set(FILES),'Expected exactly four train/validation files'
    digits=load_digits()
    train,remainder=train_test_split(np.arange(len(digits.target)),test_size=.4,
        stratify=digits.target,random_state=271890601)
    valid,_unused=train_test_split(remainder,test_size=.5,
        stratify=digits.target[remainder],random_state=271890602)
    # The complete public dataset is needed to reproduce the already defined split.
    # No held-out feature arrays or test files are constructed, saved or evaluated.
    out.mkdir(parents=True,exist_ok=False)
    actual={}
    for label,indices in [('train',train),('valid',valid)]:
        for suffix,array in [('X',(digits.data[indices]/16).astype('float32')),
                             ('Y',np.eye(10,dtype='float32')[digits.target[indices]])]:
            path=out/f'{label}{suffix}.npy';np.save(path,array)
            actual[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual[path.name]==expected[path.name],f'Frozen data mismatch: {path}'
    assert {x.name for x in out.iterdir()}==set(FILES)
    print(json.dumps({'status':'all four frozen data hashes match','output':str(out),
        'rows':{'train':len(train),'valid':len(valid)},'test_files_created_or_read':False,
        'numpy':np.__version__,'sha256':actual},indent=2))
if __name__=='__main__':main()
