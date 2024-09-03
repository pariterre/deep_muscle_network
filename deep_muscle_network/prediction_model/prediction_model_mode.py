from enum import Enum, auto


class PredictionModelMode(Enum):
    """
    Enumeration for different modes of operation.

    Attributes
    ----------
    MUSCLE
        Mode for muscle (lmt) only.
    DLMT_DQ
        Mode for muscle length jacobian (dlmt_dq) only.
    MUSCLE_DLMT_DQ
        Mode for both muscle (lmt) and muscle length jacobian (dlmt_dq).
    TORQUE
        Mode for torque only.
    TORQUE_MUS_DLMT_DQ
        Mode for torque, muscle (lmt), and muscle length jacobian (dlmt_dq).
    DLMT_DQ_FM
        Mode for muscle length jacobian (dlmt_dq) and muscular force (fm).
    FORCE
        Mode for muscular force (fm) only.
    DLMT_DQ_F_TORQUE
        Mode for muscle length jacobian (dlmt_dq), muscular force (fm) and torque.
    """

    MUSCLE = auto()
    DLMT_DQ = auto()
    MUSCLE_DLMT_DQ = auto()
    TORQUE = auto()
    TORQUE_MUS_DLMT_DQ = auto()
    DLMT_DQ_FM = auto()
    FORCE = auto()
    DLMT_DQ_F_TORQUE = auto()
