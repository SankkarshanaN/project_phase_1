"""Evidential Deep Learning classification head.

Replaces a standard softmax classification layer with one that outputs a
Dirichlet distribution over classes instead of a point-estimate probability
vector. The Dirichlet's concentration parameters (`alpha`) encode both the
predicted class distribution AND how much "evidence" supports it -- so a
detection with low total evidence (e.g. a partially-occluded pedestrian
glimpsed for one frame) comes out with high uncertainty even if the raw
class scores look confident, instead of a softmax head's forced-confident
wrong answer.

Reference: Sensoy et al., "Evidential Deep Learning to Quantify
Classification Uncertainty" (NeurIPS 2018).
"""
import torch
import torch.nn as nn


class EvidentialHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
        self.softplus = nn.Softplus()  # ensures positive evidence

    def forward(self, x: torch.Tensor):
        evidence = self.softplus(self.fc(x))
        alpha = evidence + 1.0  # Dirichlet concentration parameters
        uncertainty = self.num_classes / alpha.sum(dim=-1, keepdim=True)
        return alpha, uncertainty

    @staticmethod
    def expected_probability(alpha: torch.Tensor) -> torch.Tensor:
        """Mean of the Dirichlet(alpha) distribution -- the calibrated class
        probability vector to report as "confidence" alongside `uncertainty`."""
        return alpha / alpha.sum(dim=-1, keepdim=True)


DEFAULT_UNCERTAINTY_THRESHOLD = 0.5


def uncertainty_flag(uncertainty, threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD) -> bool:
    """Per spec: "Uncertainty flag when below threshold" -- flags a
    detection as low-confidence/needs-caution once its uncertainty reaches
    `threshold`, e.g. to trigger a "trust radar over camera" fallback in a
    downstream sensor-fusion decision. Accepts a python float or a
    single-element tensor."""
    if torch.is_tensor(uncertainty):
        uncertainty = uncertainty.item()
    return bool(uncertainty >= threshold)


def evidential_mse_loss(alpha: torch.Tensor, target_onehot: torch.Tensor, epoch: int,
                         annealing_epochs: int = 10) -> torch.Tensor:
    """Sensoy et al.'s type-II maximum likelihood loss with a KL-divergence
    regularizer (annealed in over `annealing_epochs`) that shrinks evidence
    for the *wrong* classes toward zero -- this is what teaches the head to
    output high uncertainty instead of a confident wrong answer, rather than
    just fitting the Dirichlet mean to the target like a relabeled MSE would.
    """
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S

    # Expected sum-of-squares error between the Dirichlet mean and the target,
    # plus the Dirichlet's own variance term (forces evidence down when wrong).
    err = (target_onehot - p).pow(2).sum(dim=-1, keepdim=True)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=-1, keepdim=True)
    mse = (err + var).squeeze(-1)

    # KL(Dir(alpha_tilde) || Dir(1,...,1)) regularizer: alpha_tilde removes
    # the evidence that already supports the correct class, so only
    # *incorrect* evidence is penalized.
    alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
    K = alpha.shape[-1]
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde).squeeze(-1)
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
    )

    annealing_coef = min(1.0, epoch / max(annealing_epochs, 1))
    return (mse + annealing_coef * kl).mean()
