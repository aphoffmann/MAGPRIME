# magprime/utility/interpolation/__init__.py
#
# Interpolation backends depend on optional third-party packages (tqdm for the
# linear / zero-fill helpers, pymssa for the MSSA backend).  Import them
# defensively so that importing magprime.utility never hard-fails when an
# optional backend is missing; unavailable backends are left as ``None``.

linear = None
zero_fill = None
mssa = None

try:
    from . import linear
except ImportError:
    pass

try:
    from . import zero_fill
except ImportError:
    pass

try:
    from . import mssa
    from .mssa import *
except ImportError:
    pass
