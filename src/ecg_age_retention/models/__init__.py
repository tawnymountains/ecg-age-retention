"""ECG encoder backbone, self-supervised objectives and the Cox batch loss.

Requires PyTorch.
"""
from .resnet1d import ResBlock1d, ResNet1dEncoder
from .ssl import (
    BYOLObjective,
    ContrastiveObjective,
    DenoisingObjective,
    MaskedReconstructionObjective,
    VICRegObjective,
    byol_loss,
    masked_reconstruction_loss,
    mlp_bn,
    nt_xent_loss,
    projection_head,
    reconstruction_decoder,
    reconstruction_mse_loss,
    vicreg_expander,
    vicreg_loss,
)
from .survival_loss import CoxObjective, cox_ph_loss, survival_head

__all__ = [
    "ResBlock1d",
    "ResNet1dEncoder",
    "ContrastiveObjective",
    "VICRegObjective",
    "BYOLObjective",
    "MaskedReconstructionObjective",
    "DenoisingObjective",
    "CoxObjective",
    "nt_xent_loss",
    "vicreg_loss",
    "byol_loss",
    "masked_reconstruction_loss",
    "reconstruction_mse_loss",
    "cox_ph_loss",
    "projection_head",
    "mlp_bn",
    "vicreg_expander",
    "reconstruction_decoder",
    "survival_head",
]
