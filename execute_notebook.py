"""Execute from a fresh kernel and save a reviewed copy under fresh_runs/."""
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

from src.config import ROOT
from src.utils import clean_display_paths, display_path, fresh_directory


def main() -> None:
    output = fresh_directory("executed")
    kernel = output / "kernels/toy"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Toy", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(output)
    notebook = nbformat.read(ROOT / "demo.ipynb", as_version=4)
    # Discard saved state; this must run in order rather than inherit previous outputs.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=1200, kernel_name="toy",
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, output / "failed_notebook.ipynb")
        raise
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = [nbformat.from_dict(clean_display_paths(item)) for item in cell.outputs]
    nbformat.write(notebook, output / "demo.ipynb")
    print(display_path(output / "demo.ipynb"))


if __name__ == "__main__":
    main()
