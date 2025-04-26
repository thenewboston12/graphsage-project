# GraphSAGE Extensions: Attention and Skip Connections

### Author: Yermakhan Magzym (hq4793)

This project extends the original GraphSAGE implementation by introducing:

- Attention-based neighbor aggregation
- Skip connections between layers
- Combined attention + skip architecture

### Requirements

pytorch >0.2 is required.

## How to Run

From the project root directory, run:

```bash
python train.py --dataset cora --model baseline
```

### Options:

- `--dataset`: `cora` or `pubmed`
- `--model`: `baseline`, `attention`, `skip`, or `skip_and_attention`

### Example:

```bash
python train.py --dataset pubmed --model skip_and_attention
```

## Dependencies

- Python 3.9+
- PyTorch 2.x
- NumPy
- scikit-learn
- matplotlib
