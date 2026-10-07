from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, re
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
out=root/'fresh_runs'/('executed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
out.mkdir(parents=True)
ks=out/'kernels/toy';ks.mkdir(parents=True)
(ks/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Toy','language':'python'}))
os.environ['JUPYTER_PATH']=str(out)
n=nbformat.read(root/'demo.ipynb',as_version=4)
NotebookClient(n,timeout=1200,kernel_name='toy',resources={'metadata':{'path':str(root)}}).execute()
# Remove workstation paths from display text; generated run files keep the raw logs.
def clean_output(value):
    if isinstance(value, str):
        return re.sub(r'/(?:Users|home)/[^\s"<>]+', '<local-path>', value)
    if isinstance(value, list):
        return [clean_output(x) for x in value]
    if isinstance(value, dict):
        return {k: clean_output(v) for k, v in value.items()}
    return value
for cell in n.cells:
    if cell.cell_type == 'code':
        cell.outputs = [nbformat.from_dict(clean_output(o)) for o in cell.outputs]
nbformat.write(n,out/'demo.ipynb')
print(out/'demo.ipynb')
