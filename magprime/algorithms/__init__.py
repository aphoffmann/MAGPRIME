# magprime/algorithms/__init__.py

# ---- Interference removers ----
from .interference.ICA    import ICA
from .interference.MSSA   import MSSA
from .interference.NESS   import NESS
from .interference.NESSA  import NESSA
from .interference.NEUBAUER import NEUBAUER
from .interference.PiCoG  import PiCoG
from .interference.RAMEN  import RAMEN
from .interference.REAM   import REAM
from .interference.SHEINKER import SHEINKER
from .interference.UBSS   import UBSS
from .interference.WAICUP import WAICUP
from .interference.WNEUBAUER import WNEUBAUER


# ---- Anomaly detectors ----
from .anomaly.EventDetection import EventDetection


__all__ = [
    "ICA", "MSSA", "NESS", "NESSA", "NEUBAUER", "PiCoG",
    "RAMEN", "REAM", "SHEINKER", "UBSS", "WAICUP", "WNEUBAUER",
    "EventDetection",
]
