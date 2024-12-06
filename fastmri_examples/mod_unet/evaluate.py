"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from huggingface_hub import hf_hub_download

import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fastmri.data import transforms
import fastmri


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            value = func(target,recons)
            if isinstance(value, np.ndarray):  # Ensure we push scalars
                value = value.item()
            self.metrics[metric].push(value)
            

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    txt_filepath=""
    if args.repo_id is None:
        pred_files = args.target_path.iterdir()
    else:
        txt_filepath = f'hf/knee_singlecoil_val.txt'
        pred_files = []
        with open(txt_filepath, 'r') as f:
            for line in f:
                fname = f'singlecoil_val/{line.strip()}'
                pred_files.append(fname)

        if args.max_len:
            pred_files = pred_files[:args.max_len]


    for tgt_file in pred_files:
        if args.repo_id is None:
            tgt_file = args.target_path / tgt_file
            recon_file = args.predictions_path / tgt_file.name
        else:
            
            tgt_file = hf_hub_download(repo_id=args.repo_id, filename=tgt_file, repo_type="dataset")
            t_name = pathlib.Path(tgt_file).name
            recon_file = args.predictions_path / t_name
            
        with h5py.File(tgt_file, "r") as target, h5py.File(
            recon_file, "r"
        ) as recons:
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue

            if args.acceleration and target.attrs["acceleration"] != args.acceleration:
                continue
            

            if args.repo_id is not None and 'test' in txt_filepath:
                target = fastmri.ifft2c(transforms.to_tensor(target["kspace"][()]))
                target = transforms.complex_center_crop(target, (320, 320))
                target = fastmri.complex_abs(target)
                target, _, _= transforms.normalize_instance(target, eps=1e-11)
                target = target.clamp(-6, 6)
                target= target.cpu().numpy()
            elif (args.repo_id is not None and "val" in txt_filepath) or args.repo_id is None:
                target = target[recons_key][()]
                target = transforms.center_crop(
                target, (target.shape[-1], target.shape[-1])
            )
           
            recons = recons["reconstruction"][()]
            recons = transforms.center_crop(
                recons, (target.shape[-1], target.shape[-1])
            )
            recons = recons.squeeze()
            metrics.push(target, recons)
            
    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        type=str,
        help="Repo ID to use huggingface data",
    )
    parser.add_argument(
        "--max_len",
        default=None,
        type=int,
        help="Maximum length of input images with hf data for quick debugging",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        required=True,
        help="Which challenge",
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args = parser.parse_args()

    recons_key = (
        "reconstruction_rss" if args.challenge == "multicoil" else "reconstruction_esc"
    )
    metrics = evaluate(args, recons_key)
    print(metrics)
