# Hybrid Structural-Semantic Alignment

Research code for **Hybrid Structural-Semantic Alignment of Graph-Based Knowledge Representations**.

## Files

- `final_ga.py`: proposed graph-alignment method and experiment sweeps.
- `gnn1.py`: PyTorch Geometric GNN alignment baselines.
- `REGAL.py`: REGAL/xNetMF baseline adapted to the noisy graph-view protocol.
- `Our_model_DAWAK.ipynb`: notebook version of the proposed method.

## Setup and execution

Install the Python dependencies in your preferred environment:

```sh
python -m pip install -r requirements.txt
```

Run the scripts from this directory:

```sh
python final_ga.py
python gnn1.py
python REGAL.py
```

Dataset lists, noise sweeps, seeds, and output directories are configured inside each script. Dataset loaders download the required datasets when run. Review these settings before launching the full experiments; `final_ga.py` defaults to 32 parallel jobs.

Open `Our_model_DAWAK.ipynb` in Jupyter or a compatible notebook environment to run the notebook.

The dependency list is unpinned. Python source syntax and notebook JSON were checked when preparing this repository; the full experiments and dependency compatibility have not been validated in this environment.
