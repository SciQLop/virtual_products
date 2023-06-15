from datetime import datetime
import speasy as spz
import numpy as np

# https://github.com/nicolasaunai/pyspace
from pyspace.analysis import mva

from SciQLop.backend.pipelines_model.easy_provider import EasyVector, EasyScalar


def mms1_Blmn_srvy(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    b = spz.get_data(
        spz.inventories.tree.cda.MMS.MMS1.FGM.MMS1_FGM_SRVY_L2.mms1_fgm_b_gsm_srvy_l2,
        start,
        stop,
    )
    tb = b["Bt"].time.astype(np.timedelta64) / np.timedelta64(1, "s")

    lmn = mva.MVA(b.values[:, 3])
    blmn = lmn.vec2lmn(b.values[:, 3])

    return tb, blmn


provider = EasyVector(
    path="mms_vprod/mms1/b_lmn_srvy",
    get_data_callback=mms1_Blmn_srvy,
    components_names=["N", "M", "L"],
    metadata={},
)
