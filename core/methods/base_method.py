from .pgdat import PGDAT
from .atbsl import ATBSL
from .awp import AdvWeightPerturb
from .reat import REAT
from .robal import RoBal
from .taet import TAET

BASE_METHOD_DICT = {
    "pgdat": PGDAT,
    "awp": AdvWeightPerturb,
    "robal": RoBal,
    "reat": REAT,
    "atbsl": ATBSL,
    "taet": TAET,
}





