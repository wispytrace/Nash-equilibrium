from .fixed import DumpRecords
from .euler import DumpRecords
from .event_trigger import DumpRecords
from .switching import DumpRecords
from .high_order import DumpRecords
from .euler_constraint import DumpRecords
from .fixed_switch import DumpRecords
 

Draw = {
    "fixed" : fixed.DumpRecords,
    "euler": euler.DumpRecords,
    "event_trigger": event_trigger.DumpRecords,
    "switching": switching.DumpRecords,
    "high_order": high_order.DumpRecords,
    "euler_constraint" : euler_constraint.DumpRecords,
    "fixed_switch": fixed_switch.DumpRecords,
}