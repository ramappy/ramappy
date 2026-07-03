import sys
from unittest.mock import MagicMock

# Mock numba if not installed or incompatible
try:
    import numba
except (ImportError, RuntimeError):
    numba_mock = MagicMock()

    def jit_mock(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) > 0 and callable(args[0]):
            return args[0]
        return decorator

    numba_mock.jit = jit_mock
    sys.modules["numba"] = numba_mock
