from datetime import datetime
import speasy as spz
import numpy as np
import scipy

from SciQLop.backend.pipelines_model.easy_provider import EasyVector, EasyScalar

src = {1: spz.inventories.tree.cda.MMS.MMS1.MEC.MMS1_MEC_SRVY_L2_EPHT89D.mms1_mec_r_gsm,
2: spz.inventories.tree.cda.MMS.MMS2.MEC.MMS2_MEC_SRVY_L2_EPHT89D.mms2_mec_r_gsm,
3 : spz.inventories.tree.cda.MMS.MMS3.MEC.MMS3_MEC_SRVY_L2_EPHT89D.mms3_mec_r_gsm,
4 : spz.inventories.tree.cda.MMS.MMS4.MEC.MMS4_MEC_SRVY_L2_EPHT89D.mms4_mec_r_gsm}

def mms_position(start: datetime, stop: datetime, source) -> (np.ndarray, np.ndarray):
    pos = spz.get_data(
        source,
        start,
        stop,
    )
    tp = pos.time.astype(np.timedelta64) / np.timedelta64(1, "s")

    Re_km = 6371
    print(pos.axes[0].shape,pos.values.shape)
    #pos = spz.products.variable.SpeasyVariable(pos.axes, spz.core.data_containers.DataContainer(pos.values / Re_km), pos.columns)
    pos = pos.values/Re_km
    print(pos)
    return tp, pos[:]


def mms1_position(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    return mms_position(start, stop, src[1])

def mms2_position(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    return mms_position(start, stop, src[2])

def mms3_position(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    return mms_position(start, stop, src[3])

def mms4_position(start: datetime, stop: datetime) -> (np.ndarray, np.ndarray):
    return mms_position(start, stop, src[4])

funcs = [mms1_position, mms2_position, mms3_position, mms4_position]

for i,func in enumerate(funcs):
    EasyScalar(path=f"mms_vprods/position_earth_radii_{i+1}",
        get_data_callback=func,
        component_name=["x","y","z"],
        metadata={},
)
