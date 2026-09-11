"""Langevin refinement in mask-logit space. Safe under torch.no_grad() callers:
gradients w.r.t. logits are taken inside an enable_grad() block."""
import torch


def langevin_refine(energy, feat, logits, steps, step_size=5.0, noise=0.02, clamp=6.0):
    logits = logits.detach()
    for _ in range(int(steps)):
        with torch.enable_grad():
            logits = logits.requires_grad_(True)
            e = energy(feat, torch.sigmoid(logits)).sum()
            g = torch.autograd.grad(e, logits)[0]
        logits = (logits.detach() - step_size * g
                  + noise * torch.randn_like(logits)).clamp(-clamp, clamp)
    return logits
