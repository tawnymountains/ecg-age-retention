"""Shapes and losses of the PyTorch modules (skipped when PyTorch is not installed)."""

import importlib.util
import math
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from ecg_age_retention.models import (
        BYOLObjective,
        ContrastiveObjective,
        CoxObjective,
        DenoisingObjective,
        MaskedReconstructionObjective,
        ResBlock1d,
        ResNet1dEncoder,
        VICRegObjective,
        cox_ph_loss,
        nt_xent_loss,
    )
    from ecg_age_retention.supervision import AgeHead, AgeSupervisedObjective, age_supervision_loss


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchEncoderShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_encoder_output(self):
        encoder = ResNet1dEncoder().eval()
        with torch.no_grad():
            self.assertEqual(tuple(encoder(torch.randn(2, 12, 4096)).shape), (2, 256))
            self.assertEqual(tuple(encoder(torch.randn(3, 12, 1000)).shape), (3, 256))
        self.assertEqual(encoder.out_dim, 256)

    def test_residual_block(self):
        block = ResBlock1d(64, 128).eval()
        with torch.no_grad():
            self.assertEqual(tuple(block(torch.randn(2, 64, 1024)).shape), (2, 128, 256))

    def test_age_head(self):
        head = AgeHead(hidden_dim=128)
        self.assertEqual(tuple(head(torch.randn(5, 256)).shape), (5, 1))
        with self.assertRaises(TypeError):
            AgeHead()


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchObjectiveShapeTests(unittest.TestCase):
    batch, length = 4, 1024

    def setUp(self):
        torch.manual_seed(2)
        self.x = torch.randn(self.batch, 12, self.length)

    def check(self, loss, representations, n_views):
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(representations), n_views)
        for z in representations:
            self.assertEqual(tuple(z.shape), (self.batch, 256))

    def test_two_view_objectives(self):
        view2 = self.x + 0.1 * torch.randn_like(self.x)
        for objective in (
            ContrastiveObjective(ResNet1dEncoder()),
            VICRegObjective(ResNet1dEncoder()),
            BYOLObjective(ResNet1dEncoder()),
        ):
            loss, representations = objective(self.x, view2)
            self.check(loss, representations, 2)

    def test_reconstruction_objectives(self):
        mask = torch.zeros(self.batch, 1, self.length)
        mask[..., 100:300] = 1.0
        loss, representations = MaskedReconstructionObjective(ResNet1dEncoder())(self.x * (1 - mask), self.x, mask)
        self.check(loss, representations, 1)
        loss, representations = DenoisingObjective(ResNet1dEncoder())(self.x + 0.2 * torch.randn_like(self.x), self.x)
        self.check(loss, representations, 1)

    def test_survival_objective(self):
        time = torch.tensor([5.0, 3.0, 4.0, 1.0])
        event = torch.tensor([1.0, 0.0, 1.0, 1.0])
        loss, representations = CoxObjective(ResNet1dEncoder())(self.x, time, event)
        self.check(loss, representations, 1)

    def test_byol_target_update(self):
        objective = BYOLObjective(ResNet1dEncoder(), momentum=0.9)
        online = next(objective.enc.parameters())
        target = next(objective.target_enc.parameters())
        with torch.no_grad():
            before = target.clone()
            online.add_(1.0)
        objective.update_target()
        torch.testing.assert_close(target, 0.9 * before + 0.1 * online)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchLossTests(unittest.TestCase):
    def test_age_supervision_loss(self):
        # Huber_5: 0 -> 0, 10 -> 5 * (10 - 2.5) = 37.5, 3 -> 0.5 * 9 = 4.5; mean 14, times lambda 2.
        prediction = torch.tensor([0.0, 10.0, 3.0])
        target = torch.zeros(3)
        self.assertAlmostEqual(age_supervision_loss([prediction], [target], lambda_age=2.0).item(), 28.0, places=5)
        # Two views: the second view has zero loss, so the average over views is 14.
        value = age_supervision_loss([prediction, torch.zeros(3)], [target, target], lambda_age=2.0)
        self.assertAlmostEqual(value.item(), 14.0, places=5)

    def test_cox_partial_likelihood(self):
        # Times 3 > 2 > 1, deaths at 3 and 2, zero risk scores: the death at 2 has risk set {3, 2}.
        risk = torch.zeros(3)
        loss = cox_ph_loss(risk, torch.tensor([3.0, 2.0, 1.0]), torch.tensor([1.0, 1.0, 0.0]))
        self.assertAlmostEqual(loss.item(), math.log(2.0) / 2.0, places=5)
        no_event = cox_ph_loss(risk, torch.tensor([3.0, 2.0, 1.0]), torch.zeros(3))
        self.assertEqual(no_event.item(), 0.0)

    def test_nt_xent_identical_views(self):
        # Two samples with orthogonal projections and identical views: each positive has
        # cosine 1 (logit 1/T), the two negatives have cosine 0 (logit 0).
        p = torch.eye(2)
        expected = -math.log(math.exp(1 / 0.2) / (math.exp(1 / 0.2) + 2.0))
        self.assertAlmostEqual(nt_xent_loss(p, p.clone(), temperature=0.2).item(), expected, places=5)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchAgeSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_joint_objective(self):
        model = AgeSupervisedObjective(ContrastiveObjective(ResNet1dEncoder()), AgeHead(hidden_dim=256), lambda_age=0.5)
        x = torch.randn(4, 12, 1024)
        age = torch.tensor([30.0, 50.0, 70.0, 90.0])
        output = model(x, x + 0.1 * torch.randn_like(x), age=age)
        self.assertEqual(len(output.age_predictions), 2)
        self.assertEqual(tuple(output.age_predictions[0].shape), (4,))
        torch.testing.assert_close(output.total.detach(), output.task_loss + output.age_loss)
        output.total.backward()
        for module in (model.encoder, model.age_head, model.objective.head):
            self.assertTrue(all(p.grad is not None for p in module.parameters()))
        self.assertIs(model.encoder, model.objective.enc)

    def test_lambda_is_required(self):
        with self.assertRaises(TypeError):
            AgeSupervisedObjective(ContrastiveObjective(ResNet1dEncoder()), AgeHead(hidden_dim=256))


if __name__ == "__main__":
    unittest.main()
