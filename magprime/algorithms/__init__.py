# magprime/algorithms/__init__.py

# ---- Interference removers ----
from .interference    import ICA
from .interference   import MSSA
from .interference   import NESS
from .interference import NEUBAUER
from .interference  import PiCoG
from .interference  import RAMEN
from .interference   import REAM
from .interference import SHEINKER
from .interference   import UBSS
from .interference import WAICUP
from .interference import DAFGrad


# ---- Anomaly detectors ----
from .anomaly import RUDE
from .anomaly import RUDER

# ---- Geological Survey Methods ----
from .survey import PFSS

__all__ = [
    "ICA", "MSSA", "NESS", "NEUBAUER", "PiCoG",
    "RAMEN", "REAM", "SHEINKER", "UBSS", "WAICUP", "DAFGrad",
    "RUDE", "RUDER", "PFSS"
]
