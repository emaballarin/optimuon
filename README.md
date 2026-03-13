# optimuon

A performance-optimized [Muon](https://kellerjordan.github.io/posts/muon/) optimizer for PyTorch.

**Features:**

- **Foreach-native**: uses `torch._foreach_*` ops for momentum, weight decay, and parameter updates.
- **Batched Newton-Schulz**: groups matrices by shape for parallel orthogonalization.
- **Auto-parameter routing**: automatically partitions model parameters into Muon-eligible (≥2D hidden weights) and auxiliary (embeddings, heads, norms, biases).
- **Composite optimizer**: `CompositeMuon` combines Muon with any arbitrary auxiliary optimizer (not just AdamW).
- **Three LR modes**: Keller Jordan's `"original"` (with aspect-ratio scaling), Moonshot AI's `"match_rms_adamw"`, and `"none"` (no scaling).
- **Momentum conventions**: `"ema"` (`m = beta*m + (1-beta)*g`, default) and `"classical"` (`m = beta*m + g`).
- **Corrections**: MARS, cautious updates, gradient/update clipping (all toggleable).
- **Distributed**: `torch.distributed` gradient sharding via `all_gather`.

## Installation

```bash
uv pip install git+https://github.com/emaballarin/optimuon
```

## Quick start

### Standalone Muon (manual parameter selection)

```python
from optimuon import Muon

# Muon for ≥2D hidden weight matrices only
muon = Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.01)

# Separate AdamW for everything else
import torch
adamw = torch.optim.AdamW(other_params, lr=3e-4)

# Training loop
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    muon.step()
    adamw.step()
    muon.zero_grad()
    adamw.zero_grad()
```

### CompositeMuon with auto-routing (recommended)

```python
from optimuon import CompositeMuon

optimizer = CompositeMuon(
    model,
    muon_lr=0.02,
    muon_kwargs={"weight_decay": 0.01, "foreach": True},
    aux_optimizer_class=torch.optim.AdamW,
    aux_optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.01},
    verbose=True,
)

for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### With corrections

```python
from optimuon import CompositeMuon

optimizer = CompositeMuon(
    model,
    muon_lr=0.02,
    muon_kwargs={
        "weight_decay": 0.01,
        "mars": True,           # MARS gradient correction
        "cautious": True,       # cautious update masking
        "grad_clip": 1.0,       # gradient norm clipping
    },
    aux_optimizer_class=torch.optim.AdamW,
    aux_optimizer_kwargs={"lr": 3e-4},
)
```

### With a custom auxiliary optimizer

```python
from optimuon import CompositeMuon

optimizer = CompositeMuon(
    model,
    muon_lr=0.02,
    aux_optimizer_factory=lambda param_groups: SomeExoticOptimizer(param_groups, lr=1e-3),
)
```

### Manual routing utilities

```python
from optimuon import partition_params

result = partition_params(model)
print(f"Muon: {result.muon_names}")
print(f"Aux:  {result.aux_names}")
```

## References

- Keller Jordan et al., [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (2024)
- Moonshot AI, [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (2025)
- Essential AI, [Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222) (2025)

## License

MIT
