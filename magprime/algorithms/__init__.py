# magprime/algorithms/__init__.py

# ---- Interference removers ----
from .interference    import ICA
from .interference   import MSSA
from .interference   import NESS
from .interference  import NESSA
from .interference import NEUBAUER
from .interference  import PiCoG
from .interference  import RAMEN
from .interference   import REAM
from .interference import SHEINKER
from .interference   import UBSS
from .interference import WAICUP
from .interference import WNEUBAUER


# ---- Anomaly detectors ----
from .anomaly import EventDetector

# ---- Geological Survey Methods ----
from .survey import PFSS

__all__ = [
    "ICA", "MSSA", "NESS", "NESSA", "NEUBAUER", "PiCoG",
    "RAMEN", "REAM", "SHEINKER", "UBSS", "WAICUP", "WNEUBAUER",
    "EventDetection", "PFSS"
]
