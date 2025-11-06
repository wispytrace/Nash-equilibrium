from .fixed import *
from .fixed_2 import *
from .euler import *
from .communication import *
from .high_order import *
from .euler_constraint import *
from .euler_mq import *
from .fixed_switch import *
from .fixed_high_order import *

CONFIG_MAP = {
    "fixed": fixed.config,
    "fixed2": fixed_2.config,
    "euler": euler.config,
    "communication": communication.config,
    "high_order": high_order.config,
    "euler_constraint": euler_constraint.config,
    "euler_mq": euler_mq.config,
    "fixed_switch": fixed_switch.config,
    "fixed_high_order": fixed_high_order.config
}