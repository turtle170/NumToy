from .numtoy_core import PyEngine, PyExpr

class Engine:
    """
    NumToy Hardware Engine.
    Manages allocations and bit-packing/unpacking.
    """
    def __init__(self):
        self._engine = PyEngine()

    @property
    def inner(self):
        return self._engine


class Expr:
    """
    NumToy Lazy Expression Graph Node.
    """
    def __init__(self, inner_expr: PyExpr):
        self._expr = inner_expr

    @staticmethod
    def var(engine: Engine, name: str, values: list, dtype: str = "float", bits: int = 32, scale: int = 1) -> 'Expr':
        """
        Create a new variable expression node.
        
        Args:
            engine: The NumToy Engine.
            name: Name of the variable.
            values: A list of float/int numbers.
            dtype: The datatype to pack values into.
                   Supported: 'float' (bits: 32 or 64), 'int' (bits: 8, 16, 32, 64), 'dynamic_float', 'floating_int'
            bits: The bit-width for the datatype.
            scale: Scaling factor (specifically for 'floating_int' type).
        """
        import random
        id_ = random.randint(1, 1000000)
        inner = PyExpr.new_var(engine.inner, id_, name, dtype, bits, [float(v) for v in values], scale)
        return Expr(inner)

    @staticmethod
    def const(value: float) -> 'Expr':
        """Create a constant expression node."""
        inner = PyExpr.new_const(float(value))
        return Expr(inner)

    def __add__(self, other) -> 'Expr':
        if isinstance(other, (int, float)):
            other = Expr.const(other)
        elif not isinstance(other, Expr):
            raise TypeError("Addition operand must be Expr or number")
        return Expr(self._expr.add(other._expr))

    def __radd__(self, other) -> 'Expr':
        return Expr.const(other) + self

    def __mul__(self, other) -> 'Expr':
        if isinstance(other, (int, float)):
            other = Expr.const(other)
        elif not isinstance(other, Expr):
            raise TypeError("Multiplication operand must be Expr or number")
        return Expr(self._expr.mul(other._expr))

    def __rmul__(self, other) -> 'Expr':
        return Expr.const(other) * self

    def execute(self, engine: Engine, device: str = "cpu") -> 'Expr':
        """Compile and execute the expression tree using Cranelift JIT fuser (CPU) or wgpu (GPU).
        
        Args:
            engine: The NumToy Engine.
            device: Execution target - 'cpu' (default) for Cranelift JIT or 'gpu' for WebGPU.
        """
        return Expr(self._expr.execute(engine.inner, device))

    def unpack(self, engine: Engine) -> list:
        """Unpack the values from the underlying bit-packed hardware buffer."""
        return self._expr.unpack(engine.inner)


def compile(func):
    """
    Decorator to compile a Python function into a fused NumToy expression.
    When invoked, it executes the function using lazy evaluation to build the IR graph,
    compiles it using Cranelift JIT (CPU) or wgpu (GPU), and executes it.
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
