"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd
from torch.nn import functional as F

from fastmri.models import Unet

from fastmri.pl_modules import MriModule
from fastmri.data.transforms import batched_central_weight_mask
from fastmri.data.mri_data import AnnotatedSliceDataset

class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, weight):
        return torch.mean(weight * (output - target) ** 2)


class ModUnetModule(MriModule):
    """
    Modified Unet training module to use upweighting of the center k-space

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )
        self.loss_func = WeightedMSELoss()

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        output = self(batch.image)

        # In case of publication, fix hardcoded variables below
        datamodule = self.trainer.datamodule
        train_dataloader = datamodule.train_dataloader()
        dataset = train_dataloader.dataset  # This will be an instance of AnnotatedSliceDataset
        pd_dataset = pd.DataFrame(dataset.raw_samples)

        file_name = "singlecoil_train/"+ batch.fname[0]
        slice_num = batch.slice_num[0].item()

        # Only taking first label on the slice
        labels_for_slice = pd_dataset[(pd_dataset['fname'] == file_name) & (pd_dataset['slice_ind'] == slice_num)]['metadata'].values.tolist()
        shape = batch.image.shape
        if len(labels_for_slice) > 0:
            # Initialize the weight mask with the outer weight
            if len(shape) < 4:
                slices, height, width = shape
            else:
                _, slices, height, width, _ = shape
            weight_mask = torch.full((slices, height, width), 1.0)

            for label in labels_for_slice:
        
                
                _, _, _, x0, y0, w, h, label_txt = label['annotation'].values()
                x1 = x0 + w
                y1 = y0 + h


                # Apply central weight to the central region of the batches
                weight_mask[:, y0:y1, x0:x1] = 2.0

            weight = weight_mask
        else:
            # Get the batched center weights for the loss
            weight = batched_central_weight_mask(shape)

        weight.to("mps")
    
        if torch.cuda.is_available():
            weight.to('cuda')
      
        # Compute the loss
        loss = self.loss_func(output, batch.target, weight)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        weight = batched_central_weight_mask(batch.image.shape)
        
        if torch.cuda.is_available():
            weight.to('cuda')

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss":self.loss_func(output, batch.target, weight),
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
