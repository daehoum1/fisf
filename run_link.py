"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import numpy as np
import scipy.sparse
from tqdm import tqdm
import argparse
import torch
from data_loading import get_dataset
from utils import get_missing_feature_mask
from seeds import seeds
from utils_link import train, test
import random
from torch_geometric.utils import train_test_split_edges
from models import GCNEncoder
from torch_geometric.nn import GAE
from fisf import fisf

parser = argparse.ArgumentParser("Setting for graphs with partially known features")
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of dataset",
    default="PubMed",
    choices=[
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers",
    ],
)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="structural", choices=["uniform", "structural"],
)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.0)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=100,
)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=200)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=5)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")

def run(args):
    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset, evaluator = get_dataset(name=args.dataset_name)
    n_nodes, n_features = dataset.data.x.shape
    aucs, aps = [], []

    i = 0
    for seed in tqdm(seeds[: args.n_runs]):

        dataset, evaluator = get_dataset(name=args.dataset_name)
        data = dataset.data
        data = data.to(device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.7).to(device)
        missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed = seed, type=args.mask_type,
        ).to(device)
        i += 1
        x = data.x.clone()
        x[~missing_feature_mask] = float("nan")
        filled_features = fisf(data.train_pos_edge_index, x, missing_feature_mask, args.num_iterations, args.mask_type,
                                    args.alpha, args.beta, args.gamma)

        data = data.to(device)
        x = torch.where(missing_feature_mask, data.x, filled_features)
        data.train_mask = data.val_mask = data.test_mask = None

        model = GAE(GCNEncoder(dataset.num_features, out_channels=16))
        model.to(device)
        train_pos_edge_index = data.train_pos_edge_index
        optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)

        epochs = 200
        for epoch in range(1, epochs + 1):
            loss = train(model, x, train_pos_edge_index, optimizer)
            auc, ap = test(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
        aucs.append(auc)
        aps.append(ap)
    test_auc_mean, test_auc_std = np.mean(aucs), np.std(aucs)
    test_ap_mean, test_ap_std = np.mean(aps), np.std(aps)
    print(f"AUC Accuracy: {test_auc_mean * 100:.2f}% +- {test_auc_std * 100:.2f}")
    print(f"AP Accuracy: {test_ap_mean * 100:.2f}% +- {test_ap_std * 100:.2f}")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run(args)
