from datetime import datetime
import speasy as spz
import numpy as np
import scipy

from SciQLop.backend.pipelines_model.easy_provider import EasyVector, EasyScalar


def mms1_beta_srvy(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    b = spz.get_data(
        spz.inventories.tree.cda.MMS.MMS1.FGM.MMS1_FGM_SRVY_L2.mms1_fgm_b_gsm_srvy_l2,
        start,
        stop,
    )
    P = spz.get_data(
        spz.inventories.tree.cda.MMS.MMS1.DIS.MMS1_FPI_FAST_L2_DIS_MOMS.mms1_dis_prestensor_gse_fast,
        start,
        stop,
    )
    bt = b["Bt"].values
    mu_0 = 1.256637062e-6  # H/m
    Ptot = (P.values[:, 0, 0] + P.values[:, 1, 1] + P.values[:, 2, 2]) / 3.0
    tb = b["Bt"].time.astype(np.timedelta64) / np.timedelta64(1, "s")
    tp = P.time.astype(np.timedelta64) / np.timedelta64(1, "s")
    btfunc = scipy.interpolate.interp1d(tb, bt[:, 0])
    btfinal = btfunc(tp)

    beta = Ptot * 1e-9 * 2 * mu_0 / (btfinal * 1e-9) ** 2
    return tp, beta


beta_provider = EasyScalar(
    path="mms_vprods/beta_srvy",
    get_data_callback=mms1_beta_srvy,
    component_name="x",
    metadata={},
)
