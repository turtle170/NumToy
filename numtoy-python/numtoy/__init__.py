from .numtoy_core import PyEngine, PyExpr, PyTensor
import random as _random

class Engine:
    """
    NumToy Hardware Engine.
    Manages Zig ArenaAllocator, SIMD bit-packing, and buffer tiling.
    """
    def __init__(self):
        self._engine = PyEngine()

    @property
    def inner(self):
        return self._engine


class Expr:
    """
    NumToy Lazy Expression Graph Node.
    Supports +, -, *, / operators, grad(), and device-targeted execution.
    """
    def __init__(self, inner_expr: PyExpr):
        self._expr = inner_expr

    @staticmethod
    def var(engine: Engine, name: str, values: list,
            dtype: str = "float", bits: int = 32, scale: int = 1) -> 'Expr':
        """Create a new packed variable expression node.

        Args:
            engine: The NumToy Engine.
            name: Identifier for this variable (used in grad).
            values: List of float/int values.
            dtype: 'float' | 'int' | 'dynamic_float' | 'floating_int'
            bits: Bit-width (e.g. 32 for float32, 16 for int16).
            scale: Scale factor for 'floating_int' type.

        Returns:
            Expr whose id is stable and can be passed to .grad().
        """
        id_ = _random.randint(1, 1_000_000)
        inner = PyExpr.new_var(engine.inner, id_, name, dtype, bits,
                               [float(v) for v in values], scale)
        expr = Expr(inner)
        expr._id = id_  # expose for grad()
        return expr

    @staticmethod
    def const(value: float) -> 'Expr':
        """Create a scalar constant expression node."""
        return Expr(PyExpr.new_const(float(value)))

    # ── Arithmetic ────────────────────────────────────────
    def __add__(self, other) -> 'Expr':
        other = _coerce(other)
        return Expr(self._expr.add(other._expr))

    def __radd__(self, other) -> 'Expr':
        return Expr.const(other) + self

    def __sub__(self, other) -> 'Expr':
        other = _coerce(other)
        return Expr(self._expr.sub(other._expr))

    def __rsub__(self, other) -> 'Expr':
        return Expr.const(other) - self

    def __mul__(self, other) -> 'Expr':
        other = _coerce(other)
        return Expr(self._expr.mul(other._expr))

    def __rmul__(self, other) -> 'Expr':
        return Expr.const(other) * self

    def __truediv__(self, other) -> 'Expr':
        other = _coerce(other)
        return Expr(self._expr.div(other._expr))

    def __rtruediv__(self, other) -> 'Expr':
        return Expr.const(other) / self

    # ── Auto-diff ─────────────────────────────────────────
    def grad(self, wrt) -> 'Expr':
        """Return symbolic gradient d(self)/d(wrt).

        Args:
            wrt: Either an Expr created by Expr.var() (uses its id),
                 or an integer variable id.
        """
        if isinstance(wrt, Expr):
            wrt_id = getattr(wrt, '_id', None)
            if wrt_id is None:
                raise ValueError("grad() requires a variable Expr created by Expr.var()")
        else:
            wrt_id = int(wrt)
        return Expr(self._expr.grad(wrt_id))

    # ── Execution & Unpack ────────────────────────────────
    def execute(self, engine: Engine, device: str = "cpu") -> 'Expr':
        """Compile and execute via Cranelift JIT (cpu) or WebGPU (gpu)."""
        return Expr(self._expr.execute(engine.inner, device))

    def unpack(self, engine: Engine) -> list:
        """Unpack results from the packed hardware buffer as a list of floats."""
        return self._expr.unpack(engine.inner)


class Tensor:
    """
    NumToy Unified Memory Tensor.
    Multi-dimensional array with shape/stride, broadcasting, auto-diff,
    and device-targeted execution.
    """
    def __init__(self, inner: PyTensor):
        self._t = inner

    @staticmethod
    def from_values(engine: Engine, values: list, shape: list,
                    dtype: str = "float", bits: int = 32,
                    name: str = "t", scale: int = 0) -> 'Tensor':
        """Create a Tensor from a flat list of values and a shape.

        Example:
            t = nt.Tensor.from_values(engine, [1.0,2.0,3.0,4.0,5.0,6.0], [2,3])
        """
        numel = 1
        for d in shape: numel *= d
        assert len(values) == numel, \
            f"Tensor.from_values: len(values)={len(values)} != numel={numel}"
        id_ = _random.randint(1, 1_000_000)
        inner = PyTensor.from_values(
            engine.inner, id_, name, [float(v) for v in values],
            list(shape), dtype, bits, scale if scale else None,
        )
        t = Tensor(inner)
        t._id = id_
        return t

    # ── Properties ────────────────────────────────────────
    @property
    def shape(self) -> list:
        return list(self._t.shape())

    def reshape(self, new_shape: list) -> 'Tensor':
        return Tensor(self._t.reshape(list(new_shape)))

    def broadcast_to(self, engine: Engine, target_shape: list) -> 'Tensor':
        return Tensor(self._t.broadcast_to(engine.inner, list(target_shape)))

    # ── Arithmetic ────────────────────────────────────────
    def __add__(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            raise TypeError("Tensor arithmetic requires Tensor operands")
        return _TensorPendingBinaryOp(self, other, 'add')

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            raise TypeError("Tensor arithmetic requires Tensor operands")
        return _TensorPendingBinaryOp(self, other, 'sub')

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            raise TypeError("Tensor arithmetic requires Tensor operands")
        return _TensorPendingBinaryOp(self, other, 'mul')

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            raise TypeError("Tensor arithmetic requires Tensor operands")
        return _TensorPendingBinaryOp(self, other, 'div')

    # ── Auto-diff ─────────────────────────────────────────
    def grad(self, wrt) -> 'Tensor':
        """Symbolic gradient d(self)/d(wrt)."""
        if isinstance(wrt, Tensor):
            wrt_id = getattr(wrt, '_id', None)
            if wrt_id is None:
                raise ValueError("grad() requires a Tensor created by Tensor.from_values()")
        else:
            wrt_id = int(wrt)
        return Tensor(self._t.grad(wrt_id))

    # ── Execution & Unpack ────────────────────────────────
    def execute(self, engine: Engine, device: str = "cpu") -> 'Tensor':
        """JIT-compile and execute the backing expression graph."""
        return Tensor(self._t.execute(engine.inner, device))

    def numpy(self, engine: Engine):
        """Unpack to a nested Python list matching the tensor shape."""
        flat = self._t.to_flat_f64(engine.inner)
        return _reshape_flat(flat, self.shape)


class _TensorPendingBinaryOp(Tensor):
    """Lazy binary op on two tensors — executed on .execute() or .numpy()."""
    def __init__(self, left: Tensor, right: Tensor, op: str):
        self._left = left
        self._right = right
        self._op = op
        self._t = None  # materialised on demand

    def _materialise(self, engine: Engine):
        op_fn = getattr(self._left._t, f'{self._op}_tensor')
        self._t = op_fn(self._right._t, engine.inner)

    def execute(self, engine: Engine, device: str = "cpu") -> 'Tensor':
        self._materialise(engine)
        return Tensor(self._t.execute(engine.inner, device))

    def numpy(self, engine: Engine):
        self._materialise(engine)
        result = Tensor(self._t.execute(engine.inner, "cpu"))
        return result.numpy(engine)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce(v):
    """Convert a scalar number to a constant Expr if needed."""
    if isinstance(v, (int, float)):
        return Expr.const(v)
    if isinstance(v, Expr):
        return v
    raise TypeError(f"Cannot coerce {type(v)} to Expr")


def _reshape_flat(flat: list, shape: list):
    """Recursively build a nested list from a flat list + shape."""
    if len(shape) == 1:
        return flat[:shape[0]]
    stride = len(flat) // shape[0]
    return [_reshape_flat(flat[i*stride:(i+1)*stride], shape[1:])
            for i in range(shape[0])]


def compile(func):
    """Decorator to compile a Python function into a fused NumToy expression.

    Usage:
        @nt.compile
        def f(x, y):
            return (x + y) * 2.0

        result = f(engine, [1.0, 2.0], [3.0, 4.0])           # CPU
        result = f(engine, [1.0, 2.0], [3.0, 4.0], device="gpu")  # GPU
    """
    def wrapper(*args, **kwargs):
        device = kwargs.pop("device", "cpu")
        if len(args) > 0 and isinstance(args[0], Engine):
            engine = args[0]
            func_args = []
            for i, arg in enumerate(args[1:]):
                if isinstance(arg, Expr):
                    func_args.append(arg)
                elif isinstance(arg, (list, tuple)):
                    func_args.append(Expr.var(engine, f"arg_{i}", arg))
                else:
                    func_args.append(Expr.const(arg))
            expr = func(*func_args, **kwargs)
            return expr.execute(engine, device=device)
        else:
            return func(*args, **kwargs)
    return wrapper
