from datetime import datetime
import speasy as spz
import numpy as np
import scipy
from pyspace.analysis import mva

from SciQLop.backend.pipelines_model.easy_provider import EasyVector, EasyScalar


def rififi(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    b = spz.get_data(
        spz.inventories.tree.cda.MMS.MMS1.FGM.MMS1_FGM_SRVY_L2.mms1_fgm_b_gsm_srvy_l2,
        start,
        stop,
    )
    tb = b["Bt"].time.astype(np.timedelta64) / np.timedelta64(1, "s")

    lmn = mva.MVA(b.values[:, :3])

    Bx = b["Bx GSM"].values[:, 0].mean()
    By = b["By GSM"].values[:, 0].mean()
    Bz = b["Bz GSM"].values[:, 0].mean()

    B2 = np.sqrt(Bx**2 + By**2 + Bz**2)
    Lx = lmn._LMN[2, 0]
    Ly = lmn._LMN[2, 1]
    Lz = lmn._LMN[2, 2]

    phi = (
        np.zeros_like(tb)
        + np.arccos((Lx * Bx + Ly * By + Lz * Bz) / (B2)) * 180 / np.pi
    )

    return tb, phi


phi_provider = EasyScalar(
    path="riphiphi",
    get_data_callback=rififi,
    component_name="x",
    metadata={},
)
