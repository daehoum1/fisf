# FSIF (ICML 2025)
This repository is the official PyTorch implementation of "Propagate and Inject: Revisiting Propagation-Based Feature Imputation for Graphs with Partially Observed Features" [[Paper](https://openreview.net/forum?id=QfKrcgyase)]. The codes are built on [Feature Propagation](https://github.com/twitter-research/feature-propagation).

## Requirements
python >= 3.9 <br />
torch == 1.10.2 <br />
pyg == 2.0.3

## To run the code
Semi-supervised node classification
```
python run_node.py
```
Link prediction
```
python run_link.py
```


## Citation
```
@inproceedings{umpropagate,
  title={Propagate and Inject: Revisiting Propagation-Based Feature Imputation for Graphs with Partially Observed Features},
  author={Um, Daeho and Kim, Sunoh and Park, Jiwoong and Lim, Jongin and Ahn, Seong Jin and Park, Seulki},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
