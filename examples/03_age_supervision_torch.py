"""Attach the auxiliary age head to the ResNet encoder and run one CPU training step.

Random tensors only. The 1D ResNet encoder is wrapped in the contrastive
(NT-Xent) objective, an age head is attached with
:class:`ecg_age_retention.supervision.AgeSupervisedObjective`, and one forward
and backward pass of ``L = L_task + lambda_age * L_age`` (eq. 10) is run on
random 12-lead signals with random ages. The weight ``lambda_age`` has no
default and must be given on the command line.

Requires PyTorch. Run from the repository root after ``pip install ".[models]"``
(or with ``PYTHONPATH=src``)::

    python examples/03_age_supervision_torch.py --lambda-age 0.1
"""

from __future__ import annotations

import argparse

import torch

from ecg_age_retention.models import ContrastiveObjective, ResNet1dEncoder
from ecg_age_retention.supervision import AgeHead, AgeSupervisedObjective


def gradient_norm(module: torch.nn.Module) -> float:
    """Euclidean norm of all parameter gradients of ``module``."""
    squares = [p.grad.detach().pow(2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def n_parameters(module: torch.nn.Module) -> int:
    """Number of parameters of ``module``."""
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--lambda-age", type=float, required=True, help="weight of the auxiliary age loss (required, no default)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="number of random ECGs")
    parser.add_argument("--n-samples", type=int, default=4096, help="samples per lead")
    parser.add_argument("--seed", type=int, default=7, help="seed of the random tensors and initial weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    encoder = ResNet1dEncoder()
    model = AgeSupervisedObjective(
        ContrastiveObjective(encoder),
        AgeHead(hidden_dim=256),
        lambda_age=args.lambda_age,
    ).to(device)
    model.train()

    # Random signals and two placeholder views; any view-generation scheme can be used here.
    signals = torch.randn(args.batch_size, 12, args.n_samples, device=device)
    view1 = signals + 0.05 * torch.randn_like(signals)
    view2 = signals + 0.05 * torch.randn_like(signals)
    age = torch.empty(args.batch_size, device=device).uniform_(18.0, 95.0)

    output = model(view1, view2, age=age)
    output.total.backward()

    print("Synthetic demonstration: auxiliary age supervision, one CPU forward/backward pass")
    print("Random tensors; every value below is illustrative and does not come from the paper.")
    print()
    print(f"Input views:                    {tuple(view1.shape)} (batch, leads, samples)")
    print(f"Age predictions per view:       {tuple(output.age_predictions[0].shape)}")
    print(f"lambda_age (command line):      {args.lambda_age:g}")
    print(f"Task loss (NT-Xent):            {output.task_loss.item():.4f}")
    print(f"Age loss term:                  {output.age_loss.item():.4f}  (lambda_age * Huber age loss)")
    print(f"Total loss:                     {output.total.item():.4f}")
    print(f"Gradient norm, encoder:         {gradient_norm(model.encoder):.4f}")
    print(f"Gradient norm, projection head: {gradient_norm(model.objective.head):.4f}")
    print(f"Gradient norm, age head:        {gradient_norm(model.age_head):.4f}")

    # After training only the encoder is kept; computing a representation needs no age.
    kept = model.encoder.eval()
    with torch.no_grad():
        representation = kept(signals)
    print()
    print(f"Encoder kept after training:    {n_parameters(kept)} parameters")
    print(f"Age head discarded:             {n_parameters(model.age_head)} parameters")
    print(f"Representation from encoder:    {tuple(representation.shape)}")


if __name__ == "__main__":
    main()
