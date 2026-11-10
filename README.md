# graph-pooling

Reference implementation for **Adaptive Graph Coarsening with Learned Pooling Operators** (NeurIPS 2025).

The repository contains the full training and evaluation pipeline used for the paper: learned pooling operators, the coarsening scheduler, and scripts to reproduce all main-table results.

## Layout

- `src/model.py` - encoder and pooling blocks
- `src/pooling.py` - differentiable coarsening operators
- `train.py` / `evaluate.py` - experiment entry points
- `configs/default.yaml` - default hyper-parameters

## Getting started

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
```

Released under the MIT license.
