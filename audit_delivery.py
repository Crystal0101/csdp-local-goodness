"""Check the review candidate without training, publishing or modifying files."""
from pathlib import Path
import argparse
import ast
import json
import re
import shutil
import subprocess

import nbformat
import numpy as np

from src.config import ROOT, load_config, verify_sources
from src.results import load_saved_results
from src.stats import analyse_results


def git(*arguments: str) -> str:
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def check_language(value, filename: str) -> None:
    if isinstance(value, str):
        if re.search(r'[\u3400-\u9fff]', value):
            raise ValueError(f'Non-English CJK text in {filename}')
        if re.search(r'/(?:Users|home|var/folders)/', value):
            raise ValueError(f'Workstation path in {filename}')
    elif isinstance(value, dict):
        for item in value.values():
            check_language(item, filename)
    elif isinstance(value, list):
        for item in value:
            check_language(item, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-clean', action='store_true')
    args = parser.parse_args()
    if args.require_clean and git('status', '--porcelain'):
        raise ValueError('The reviewed working tree is not clean')
    tracked = set(git('ls-files').splitlines())
    untracked = set(filter(None, git('ls-files', '--others', '--exclude-standard').splitlines()))
    files = sorted(tracked | untracked)
    for filename in files:
        path = ROOT / filename
        if path.stat().st_size > 1024 * 1024:
            raise ValueError(f'File exceeds the 1 MiB review limit: {filename}')
        if path.suffix in {'.zip', '.pt', '.pth', '.ckpt', '.npy', '.npz', '.pyc'}:
            raise ValueError(f'Generated binary in review candidate: {filename}')
        if path.suffix in {'.png', '.pdf'}:
            continue
        text = path.read_text()
        check_language(text, filename)
        if path.suffix in {'.json', '.ipynb'}:
            check_language(json.loads(text), filename)
        if path.suffix == '.py':
            ast.parse(text, filename=filename)
    for name in ('primary', 'supplement'):
        verify_sources(load_config(name))
    evidence = load_saved_results()
    summaries = analyse_results(evidence)
    # Guard the published numerical endpoints against accidental analysis drift.
    references = {
        'primary': (-.153203, -.765369, .458962),
        'G8-G1': (-.640669, -1.682539, .401202),
        'epoch20': (-.250696, -.695768, .194376),
        'budget_interaction': (.389972, -.314218, 1.094163),
    }
    for name, reference in references.items():
        result = summaries[name]
        np.testing.assert_allclose(
            [result.mean_pp, result.lower_pp, result.upper_pp], reference, atol=1e-5,
        )
    notebook = nbformat.read(ROOT / 'demo.ipynb', as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if [cell.execution_count for cell in cells] != list(range(1, len(cells) + 1)):
        raise ValueError('Notebook was not saved after an ordered complete run')
    for cell in cells:
        if any(output.output_type == 'error' for output in cell.outputs):
            raise ValueError('Notebook contains an execution error')
        if len(cell.source.splitlines()) > 15:
            raise ValueError('A presentation cell exceeds 15 lines')
    identities = git('log', '--format=%an|%cn').splitlines()
    if any(identity != 'Ning Yang|Ning Yang' for identity in identities):
        raise ValueError('Commit author or committer identity differs from Ning Yang')
    pdf_checked = bool(shutil.which('pdftotext'))
    if pdf_checked:
        text = subprocess.check_output(
            ['pdftotext', str(ROOT / 'docs/appendix.pdf'), '-'], text=True,
        )
        check_language(text, 'docs/appendix.pdf')
    print(json.dumps({
        'status': 'passed', 'files': len(files), 'saved_epochs': len(evidence.accuracy),
        'executed_notebook_cells': len(cells), 'pdf_text_checked': pdf_checked,
        'scope': 'Static checks and saved evidence only; no fresh training or publication.',
    }, indent=2))


if __name__ == '__main__':
    main()
