"""The handful of array operations the physics and controller use, for NumPy or PyTorch.

physics.py and controller.py take an `xp` argument: the numpy module itself,
or torch_ops() below. NumPy already has every function they call; this shim
gives PyTorch the same names and argument conventions, so there is one copy
of the equations for both backends.

The torch versions avoid building tensors from Python numbers while a CUDA
graph is being captured, because that would copy from the CPU and break the
capture. Python-number arguments mostly go straight to torch as kernel
scalars; divide() keeps a cache of the few constants it needs as tensors.

Division is the one operation spelled differently from plain `/`: code that
divides a tensor by a number (or the reverse) calls xp.divide. NumPy's own
np.divide is the same as `/`.
"""


class _TorchOps:
    def __init__(self):
        import torch
        self._torch = torch
        self.sin, self.cos, self.abs = torch.sin, torch.cos, torch.abs
        self._constants = {}

    def _constant(self, value, like):
        """value as a 0-d tensor on like's device, cached so it's only ever copied over once.

        The first use has to happen outside CUDA graph capture (TorchSimulator
        warms up before capturing, so it does).
        """
        key = (float(value), like.dtype, like.device)
        if key not in self._constants:
            self._constants[key] = self._torch.tensor(float(value), dtype=like.dtype, device=like.device)
        return self._constants[key]

    def divide(self, a, b):
        """Correctly rounded a / b, like NumPy.

        PyTorch computes `number / tensor` (and, on CUDA, `tensor / number`) as
        a reciprocal times a multiply, which rounds twice and can be one bit off.
        Dividing by a 0-d tensor on the same device avoids that.
        """
        a_is, b_is = isinstance(a, self._torch.Tensor), isinstance(b, self._torch.Tensor)
        if a_is and not b_is:
            b = self._constant(b, a)
        elif b_is and not a_is:
            a = self._constant(a, b)
        return a / b

    def where(self, condition, a, b):
        return self._torch.where(condition, a, b)

    def clip(self, x, low, high):
        return self._torch.clamp(x, low, high)

    def minimum(self, a, b):
        return self._torch.minimum(a, b) if isinstance(b, self._torch.Tensor) else self._torch.clamp(a, max=b)

    def maximum(self, a, b):
        return self._torch.maximum(a, b) if isinstance(b, self._torch.Tensor) else self._torch.clamp(a, min=b)

    def copysign(self, magnitude, sign):
        if not isinstance(magnitude, self._torch.Tensor):
            # full_like keeps sign's dtype (torch.where on two Python numbers
            # would give float32) and is a fill kernel, so capture-safe.
            magnitude = self._torch.full_like(sign, magnitude)
        return self._torch.copysign(magnitude, sign)


_torch_ops = None


def torch_ops() -> _TorchOps:
    """The PyTorch shim, created on first use so NumPy-only code never imports torch."""
    global _torch_ops
    if _torch_ops is None:
        _torch_ops = _TorchOps()
    return _torch_ops
