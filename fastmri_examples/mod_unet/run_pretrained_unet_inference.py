"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.models import Unet
import os

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

    progress_bar.close()


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, device, repo_id, max_len):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    # download the state_dict if we don't have it
    if state_dict_file is None or not state_dict_file.exists():
        if not Path(MODEL_FNAMES[challenge]).exists():
            url_root = UNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]

    model.load_state_dict(torch.load(state_dict_file, weights_only=True, map_location=device))
    model = model.eval()

    # data loader setup
    if "_mc" in challenge:
        data_transform = T.UnetDataTransform(which_challenge="multicoil")
    else:
        data_transform = T.UnetDataTransform(which_challenge="singlecoil")

    if repo_id is not None:
        # Data is stored in huggingface
        data_path = "singlecoil_val"

    if "_mc" in challenge:
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            challenge="multicoil",
        )
    else:
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            challenge="singlecoil",
            repo_id=repo_id,
            max_len=max_len
        )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)

        outputs[fname].append((slice_num, output))

    # save outputs
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])


    #make sure the output dirs exist
    
    fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="unet_knee_sc",
        choices=(
            "unet_knee_sc",
            "unet_knee_mc",
            "unet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    check_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument(
        "--device",
        default=check_device,
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=Path('knee_sc_leaderboard_state_dict.pt'),
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
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

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
        args.repo_id,
        args.max_len
    )
