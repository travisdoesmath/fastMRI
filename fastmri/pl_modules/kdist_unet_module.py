"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from torch.nn import functional as F

from fastmri.models import Unet

from fastmri.pl_modules import MriModule
from fastmri.data.transforms import batched_central_weight_mask

class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, weight):
        return torch.mean(weight * (output - target) ** 2)
    
class KDistLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_output, teacher_output, target, weight):
        return torch.mean(weight * (student_output - target) ** 2) + torch.mean(weight * (teacher_output - target) ** 2)


class KDistUnetModule(MriModule):
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
        ds_layer_distil=[],
        us_layer_distil=[],
        teacher_unet: Unet = None,
        recon_weight: float = 0.75,
        train_mode: bool = True,
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
            ds_layer_distil (list[int], optional): Block numbers to choose for
                knowledge distillation during the Downsampling part of UNet.
                e.g. [1, 2, 3] will distil the output of the first, second and
                third downsample block. Defaults to [].
            us_layer_distil (list[int], optional): Block numbers to choose for
                knowledge distillation during the Upsampling part of UNet.
                e.g. [1, 2, 3] will distil the output of the first, second and
                third upsample block. Defaults to [].
            student_unet (nn.Module, optional): Student U-Net model. 
                Defaults to KDistUnet.
            teacher_unet (nn.Module): Teacher U-Net model. Required for
                knowledge distillation.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['teacher_unet', 'recon_weight', 'train_mode'],)

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.recon_weight = recon_weight
        self.kdist_weight = 1 - recon_weight

        self.student_unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )
        
        if teacher_unet is not None:
            self.teacher_unet = teacher_unet.eval()
            for param in self.teacher_unet.parameters():
                param.requires_grad = False

        self.recon_loss = WeightedMSELoss()

        if train_mode:
            self._init_layer_distil(ds_layer_distil, us_layer_distil)
            self.student_features = []  # to store teacher features
            self.teacher_features = []  # to store student features
            
            # Pre-calculate feature map dimensions for distillation layers (down-sampling, bottleneck, up-sampling)
            self.student_feature_map_dims, self.student_feature_channels = self.calculate_feature_map_dims(self.student_unet, self.ds_layer_distil, self.us_layer_distil)
            self.teacher_feature_map_dims, self.teacher_feature_channels = self.calculate_feature_map_dims(self.teacher_unet, self.ds_layer_distil, self.us_layer_distil)

        # Initialize projection layers using pre-calculated dimensions
            self.projection_layers = torch.nn.ModuleList([
                torch.nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
                for student_channels, teacher_channels in zip(self.student_feature_channels, self.teacher_feature_channels)
            ])
            print(self.ds_layer_distil, self.us_layer_distil)

    def calculate_feature_map_dims(self, model, ds_layer_distil, us_layer_distil):
        """
        Pre-calculate the dimensions of the feature maps at specified layers for distillation.
        Args:
            model (nn.Module): The model (teacher or student).
            ds_layer_distil (list[int]): Indices of the down-sampling layers to consider.
            us_layer_distil (list[int]): Indices of the up-sampling layers to consider.
        Returns:
            Tuple of feature map dimensions for teacher or student model at each layer.
        """
        H, W = 320, 320  # Input Size is 320x320
        feature_map_dims = []
        feature_channels = []

        # Calculate dimensions for down-sampling layers
        current_channels = self.in_chans
        for idx in ds_layer_distil:
            layer = model.down_sample_layers[idx]
            H = (H - 1) // 2 + 1  # Pooling halving the height
            W = (W - 1) // 2 + 1  # Pooling halving the width
            feature_map_dims.append((current_channels, H, W))
            feature_channels.append(current_channels)
            current_channels *= 2  # Channels double after each down-sampling layer

        # Calculate dimensions for bottleneck
        bottleneck_layer = model.conv
        H = (H - 1) // 2 + 1  # Final ConvBlock layer's height
        W = (W - 1) // 2 + 1  # Final ConvBlock layer's width
        feature_map_dims.append((current_channels, H, W))
        feature_channels.append(current_channels)

        # Calculate dimensions for up-sampling layers
        for idx in us_layer_distil:
            layer = model.up_conv[idx]
            H = (H - 1) // 2 + 1  # Upsample halving the height
            W = (W - 1) // 2 + 1  # Upsample halving the width
            feature_map_dims.append((current_channels, H, W))
            feature_channels.append(current_channels)
            current_channels //= 2  # Channels halve after each up-sampling layer

        return feature_map_dims, feature_channels


    def _init_layer_distil(self, ds_layer_distil, us_layer_distil):
        """
        Initialize the layers selected for distillation.
        """
        def _index_choices(block):
            block_len = len(block)
            if block_len <= 4:
                return [0, block_len-1]
            else:
                return [0, block_len//2, block_len-1]

        # Set default layers for distillation if not provided
        self.ds_layer_distil = ds_layer_distil or _index_choices(self.teacher_unet.down_sample_layers)
        self.us_layer_distil = us_layer_distil or _index_choices(self.teacher_unet.up_conv)

        # Register hooks for the teacher and student networks
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks for feature distillation from the teacher to the student.
        """
        # Hook for down-sampling layers
        for idx in self.ds_layer_distil:
            # Hook for teacher's down-sampling layers
            layer = self.teacher_unet.down_sample_layers[idx]
            layer.register_forward_hook(self._save_teacher_feature)
            # Hook for student's down-sampling layers
            layer = self.student_unet.down_sample_layers[idx]
            layer.register_forward_hook(self._save_student_feature)

        # Hook for the Teacher's bottleneck
        self.teacher_unet.conv.register_forward_hook(self._save_teacher_feature)

        # Hook for the Student's bottleneck
        self.student_unet.conv.register_forward_hook(self._save_student_feature)

        # Hook for Up-sampling layers
        for idx in self.us_layer_distil:
            # Hook for teacher's up-sampling layers
            layer = self.teacher_unet.up_conv[idx]
            layer.register_forward_hook(self._save_teacher_feature)
            # Hook for student's up-sampling layers
            layer = self.student_unet.up_conv[idx]
            layer.register_forward_hook(self._save_student_feature)


    def _save_teacher_feature(self, module, input, output):
        """
        Save the teacher's features for distillation.
        """
        self.teacher_features.append(output)

    def _save_student_feature(self, module, input, output):
        """
        Save the student's features for distillation.
        """
        self.student_features.append(output)

    def forward(self, image):
        return self.student_unet(image.unsqueeze(1)).squeeze(1), self.teacher_unet(image.unsqueeze(1)).squeeze(1) if self.teacher_unet is not None else None

    def _standardize_features(self, x):
        """
        Standardize the input tensor along the channel dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Standardized tensor with mean 0 and std 1 along channels.
        """
        mean = x.mean(dim=(2, 3), keepdim=True)  # Mean over H and W
        std = x.std(dim=(2, 3), keepdim=True)    # Std over H and W
        return (x - mean) / (std + 1e-8)         # Avoid division by zero
        
    def _distillation_loss(self):
        """
        Compute the distillation loss between the teacher and student features.
        """
        distil_loss = 0.0
        for teacher_feature, student_feature, projection in zip(self.teacher_features, self.student_features, self.projection_layers):
           # Standardize features
            teacher_feature = self._standardize_features(teacher_feature)
            student_feature = self._standardize_features(student_feature)
            
            # Project the student features to the teacher feature space
            student_feature = projection(student_feature)
            if torch.cuda.is_available():
                student_feature.to('cuda')
            
            distil_loss += F.mse_loss(teacher_feature, student_feature)
        return distil_loss
    
    def _reconstruction_loss(self, output, target):
        """
        Compute the reconstruction loss.
        """
        weight = batched_central_weight_mask(output.shape)
        if torch.cuda.is_available():
            weight.to('cuda')
        return self.recon_loss(output, target, weight)
    
    def loss_func(self, batch, s_output):
         # Compute the reconstruction loss
        reconstruction_loss = self._reconstruction_loss(s_output, batch.target)

        # Compute the distillation loss
        distillation_loss = self._distillation_loss()

        # Combine the losses
        return self.recon_weight * reconstruction_loss + distillation_loss * self.kdist_weight


     # Exclude teacher_unet from saving in checkpoints
    def on_save_checkpoint(self, checkpoint):
        if "teacher_unet" in checkpoint["state_dict"]:
            del checkpoint["state_dict"]["teacher_unet"]
    
    def training_step(self, batch, batch_idx):
        s_output, _ = self(batch.image)
         # Reset the feature lists to store new features
        self.student_features = []  # Clear stored student features
        self.teacher_features = []  # Clear stored teacher features

        loss = self.loss_func(batch, s_output)
        self.log("loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        output, _ = self(batch.image)
        

        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        # Reset the feature lists to store new features
        self.student_features = []  # Clear stored student features
        self.teacher_features = []  # Clear stored teacher features

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss":self.loss_func(batch, output),
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
