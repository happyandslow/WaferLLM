"""
Hal-device python binding shared object

# simfabric
binding = hal.HalSimfab("plan.meta")
device = binding.configure()

# CM
binding = hal.HalCm("127.0.0.1:1234")
device = binding.configure()

# elf editor
editor = das.CoreEditor("plan.meta")
core = editor.load("myelf.elf")
core.read(hal.Rectangle(0,0), 0, 1)
core.write(hal.Rectangle(0,0), 0, b'\x01\x01')
editor.save(core, "mynewelf.elf")
"""

from . import halpybind as hal
from cerebras.das_common import Dimensions
from cerebras.plangen.lib.planpybind import StreamingConnection, StreamingDirection, PatioPosition

Rectangle = hal.Rectangle
