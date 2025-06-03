from .fixed import Model
from .fixed2 import Model
from .fixed3 import Model
from .fixed4 import Model
from .asymptotic import Model
from .asymptotic2 import Model
from .euler import Model
from .euler_asym import Model
from .euler2 import Model
from .trigger import Model
from .switching import Model
from .high_order import Model
from .euler_constraint import Model
from .euler_constraint_uc import Model
from .euler_constraint2 import Model
from .euler_mq import Model
from .fixed_switch import Model

Res = {
    "fixed" : fixed.Model,
    "fixed2" : fixed2.Model,
    "fixed3" : fixed3.Model,
    "fixed4" : fixed4.Model,
    "asym": asymptotic.Model,
    "asym2": asymptotic2.Model,
    "euler": euler.Model,
    "euler_asym": euler_asym.Model,
    "euler2": euler2.Model,
    "trigger": trigger.Model,
    "switching": switching.Model,
    "high_order": high_order.Model,
    "euler_constraint": euler_constraint.Model,
    "euler_constraint2": euler_constraint2.Model,
    "euler_mq": euler_mq.Model,
    "fixed_switch": fixed_switch.Model,
    "euler_constraint_uc": euler_constraint_uc.Model,
}