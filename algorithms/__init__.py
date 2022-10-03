from algorithms.afn.safn import Safn
from algorithms.ar.ar import Ar
from algorithms.ba3us.ba3us import BA3US
from algorithms.deepjdot.deepjdot import Deepjdot
from algorithms.etn.etn import Etn
from algorithms.jumbot.jumbot import Jumbot
from algorithms.mixunbot.mixunbot import Mixunbot
from algorithms.mpot.mpot import Mpot
from algorithms.pada.pada import Pada
from algorithms.source_only.source_only import SourceOnly
from algorithms.source_only.source_only_plus import SourceOnlyPlus
from algorithms.source_only.source_only_augmented import SourceOnlyAugmented

algorithms_dict = {
    'ar': Ar,
    'ba3us': BA3US,
    'deepjdot': Deepjdot,
    'etn': Etn,
    'jumbot': Jumbot,
    'mixunbot': Mixunbot,
    'mpot': Mpot,
    'pada': Pada,
    'safn': Safn,
    'source_only': SourceOnly,
    'source_only_plus': SourceOnlyPlus,
    'source_only_augmented': SourceOnlyAugmented
}
