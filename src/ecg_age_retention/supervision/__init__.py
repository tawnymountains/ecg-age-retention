"""Auxiliary age supervision: age head, Huber age loss and the joint objective.

Requires PyTorch.
"""
from .age_head import AgeHead, AgeSupervisedObjective, AgeSupervisedOutput, age_supervision_loss

__all__ = ["AgeHead", "AgeSupervisedObjective", "AgeSupervisedOutput", "age_supervision_loss"]
