from datetime import datetime
import speasy as spz
import numpy as np
import scipy
import scipy.constants as cst

from SciQLop.backend.pipelines_model.easy_provider import EasyVector, EasyScalar


def mms1_mirror_cm(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    b = spz.get_data(
        spz.inventories.tree.cda.MMS.MMS1.FGM.MMS1_FGM_SRVY_L2.mms1_fgm_b_gsm_srvy_l2,
        start,
        stop,
    )
    Tperp = spz.get_data(
        spz.inventories.data_tree.cda.MMS.MMS1.DIS.MMS1_FPI_FAST_L2_DIS_MOMS.mms1_dis_tempperp_fast,
        start,
        stop,
    )
    Tpara = spz.get_data(
        spz.inventories.data_tree.cda.MMS.MMS1.DIS.MMS1_FPI_FAST_L2_DIS_MOMS.mms1_dis_temppara_fast,
        start,
        stop,
    )
    N = spz.get_data(
        spz.inventories.data_tree.cda.MMS.MMS1.DIS.MMS1_FPI_FAST_L2_DIS_MOMS.mms1_dis_numberdensity_fast,
        start,
        stop,
    )
    anisotropy = Tperp["eT_perp"].values / Tpara["eT_para"].values
    bt = b["Bt"].values
    Pperp = N["N"].values * Tperp["eT_perp"].values
    tb = b["Bt"].time.astype(np.timedelta64) / np.timedelta64(1, "s")
    tp = N.time.astype(np.timedelta64) / np.timedelta64(1, "s")
    pfunc = scipy.interpolate.interp1d(tp, Pperp[:, 0], bounds_error=False)
    Afunc = scipy.interpolate.interp1d(tp, anisotropy[:, 0], bounds_error=False)
    betaperp = pfunc(tb) * 1e6 * cst.e * 2 * cst.mu_0 / (bt[:, 0] * 1e-9) ** 2
    cm = betaperp * (Afunc(tb) - 1)
    return tb, cm


cm_provider = EasyScalar(
    path="mms_vprods/mirror_cm",
    get_data_callback=mms1_mirror_cm,
    component_name="C_M_b",
    metadata={},
)
