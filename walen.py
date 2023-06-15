from datetime import datetime
import speasy as spz
import numpy as np

# https://github.com/nicolasaunai/pyspace
from pyspace.analysis import mva
import scipy.constants as cst
import scipy

from SciQLop.backend.pipelines_model.easy_provider import EasyVector, EasyScalar


def mms1_walen(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):

    v = spz.get_data(
        spz.inventories.data_tree.cda.MMS.MMS1.DIS.MMS1_FPI_FAST_L2_DIS_MOMS.mms1_dis_bulkv_gse_fast,
        start,
        stop,
    )
    b = spz.get_data(
        spz.inventories.tree.cda.MMS.MMS1.FGM.MMS1_FGM_SRVY_L2.mms1_fgm_b_gsm_srvy_l2,
        start,
        stop,
    )
    N = spz.get_data(
        spz.inventories.data_tree.cda.MMS.MMS1.DIS.MMS1_FPI_FAST_L2_DIS_MOMS.mms1_dis_numberdensity_fast,
        start,
        stop,
    )
    vx = v["Vx_GSE"].values[:, 0]
    vy = v["Vy_GSE"].values[:, 0]
    vz = v["Vz_GSE"].values[:, 0]

    t = v["Vx_GSE"].time.astype(np.timedelta64) / np.timedelta64(1, "s")
    tb = b["Bt"].time.astype(np.timedelta64) / np.timedelta64(1, "s")
    Nfunc = scipy.interpolate.interp1d(
        t, N.values[:, 0], bounds_error=False, fill_value="extrapolate"
    )
    NonB = Nfunc(tb)

    Vax = b["Bx GSM"].values[:, 0] * 1e-9 / np.sqrt(cst.mu_0 * cst.m_p * NonB * 1e6)
    Vay = b["By GSM"].values[:, 0] * 1e-9 / np.sqrt(cst.mu_0 * cst.m_p * NonB * 1e6)
    Vaz = b["Bz GSM"].values[:, 0] * 1e-9 / np.sqrt(cst.mu_0 * cst.m_p * NonB * 1e6)

    imid = int(Vax.size / 2.0)
    vxl = vx[0] * 1e3 + Vax[:imid] - Vax[0]
    vyl = vy[0] * 1e3 + Vay[:imid] - Vay[0]
    vzl = vz[0] * 1e3 + Vaz[:imid] - Vaz[0]
    vxr = vx[-1] * 1e3 + Vax[imid:] - Vax[-1]
    vyr = vy[-1] * 1e3 + Vay[imid:] - Vay[-1]
    vzr = vz[-1] * 1e3 + Vaz[imid:] - Vaz[-1]

    print(vx[0], vy[0], vz[0])

    vw = np.zeros((Vax.size, 3))
    vw[:imid, 0] = vxl / 1e3
    vw[imid:, 0] = vxr / 1e3

    vw[:imid, 1] = vyl / 1e3
    vw[imid:, 1] = vyr / 1e3

    vw[:imid, 2] = vzl / 1e3
    vw[imid:, 2] = vzr / 1e3

    return tb, vw


provider = EasyVector(
    path="mms_vprod/mms1/vwalen",
    get_data_callback=mms1_walen,
    components_names=["x", "y", "z"],
    metadata={},
)
