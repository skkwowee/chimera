"""Hardware-compat shim for the MoE grouped-matmul kernel path.

torch 2.8's `torch._grouped_mm` is Hopper-only — it raises unless the GPU has
compute capability *exactly* 9.0:

    RuntimeError: torch._grouped_mm is only supported on CUDA devices with
    compute capability = 9.0

transformers' `_can_use_grouped_mm` (transformers/integrations/moe.py) gates
the call on `get_device_capability() >= (9, 0)`. On Blackwell (cc 12.0) that
check passes, transformers calls `torch._grouped_mm`, and the kernel raises.

torch >= 2.9 widened `_grouped_mm` to SM80+, and transformers' `>= (9, 0)`
check is then correct. So the *only* broken case is torch < 2.9 on a
non-Hopper GPU. There, we force transformers' pure-torch
`grouped_mm_fallback` — a `torch.library` custom op with registered autograd,
so it works for forward and backward on any device. Slower than the native
grouped-mm kernel, but correct.

Verified on RTX PRO 6000 Blackwell (cc 12.0): Qwen3.6-35B-A3B loads + LoRA
forward/backward at ~72 GB peak VRAM with this shim. Without it, the forward
pass dies in `grouped_mm_experts_forward`.
"""

from __future__ import annotations


def patch_moe_for_blackwell() -> None:
    """Force transformers' grouped_mm fallback on torch<2.9 + non-Hopper GPUs.

    Idempotent and safe to call before every model load. No-op on CPU, on
    Hopper (cc 9.0), and on torch >= 2.9 where the native check is correct.
    """
    import torch

    if not torch.cuda.is_available():
        return

    # torch >= 2.9: _grouped_mm supports SM80+, transformers' check is correct.
    try:
        from transformers.utils import is_torch_greater_or_equal

        if is_torch_greater_or_equal("2.9", accept_dev=True):
            return
    except Exception:
        # If the version helper isn't importable, fall through to the
        # capability check — the worst case is we use the (correct, slower)
        # fallback on a GPU that didn't strictly need it.
        pass

    # Hopper (cc 9.0): torch 2.8's _grouped_mm works here — keep the fast path.
    if torch.cuda.get_device_capability() == (9, 0):
        return

    # torch < 2.9 on a non-Hopper GPU: transformers checks >= (9, 0) but the
    # kernel enforces == 9.0. Force the pure-torch fallback.
    import transformers.integrations.moe as _moe

    _moe._can_use_grouped_mm = lambda *args, **kwargs: False  # noqa: ARG005
    cap = torch.cuda.get_device_capability()
    print(
        f"  [moe_compat] GPU cc={cap[0]}.{cap[1]}, torch<2.9 — "
        "forcing transformers grouped_mm fallback (Blackwell-safe)"
    )
