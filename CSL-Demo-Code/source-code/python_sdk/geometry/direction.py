"""
Python helper functions for the C++-implemented cerebras.geometry.geometry
module.

to_cardinal and to_direction are copy-pasta'd from
src/pb/common/place.cc, which is difficult to wrap via pybind in the
cerebras.geometry.geometry module because of circular dependency
problems mentioned here: #15976
"""
from cerebras.pb.common.direction_pb2 import Direction, XP, XM, YP, YM, OF, ON
from cerebras.geometry.geometry import NORTH, SOUTH, WEST, EAST, NUM_DIRECTIONS

def to_cardinal(direction):
    """
    Convert Direction to cardinal_direction
    """
    if direction == XP:
        return EAST
    elif direction == XM:
        return WEST
    elif direction == YP:
        return SOUTH
    elif direction == YM:
        return NORTH
    elif direction == OF:
        return NUM_DIRECTIONS
    elif direction == ON:
        return NUM_DIRECTIONS
    else:
        raise ValueError(f"Inconvertible direction: {Direction.Value(direction)}")

def to_direction(cardinal_direction):
    """
    Convert cardinal_direction to Direction
    """
    if cardinal_direction == EAST:
        return XP
    elif cardinal_direction == WEST:
        return XM
    elif cardinal_direction == SOUTH:
        return YP
    elif cardinal_direction == NORTH:
        return YM
    else:
        raise ValueError(f"Inconvertible cardinal_direction {cardinal_direction}")

def to_string(cardinal_direction):
    """
    Convert cardinal_direction to string
    """
    if cardinal_direction == EAST:
        return "xp"
    elif cardinal_direction == WEST:
        return "xm"
    elif cardinal_direction == SOUTH:
        return "yp"
    elif cardinal_direction == NORTH:
        return "ym"
    else:
        raise ValueError(f"Inconvertible cardinal_direction {cardinal_direction}")

def is_valid(cardinal_direction):
    return (int(cardinal_direction) >= 0 and 
        int(cardinal_direction) < int(NUM_DIRECTIONS))

def is_vertical(cardinal_direction):
    return int(cardinal_direction) < 2

def is_horizontal(cardinal_direction):
	return (int(cardinal_direction) >> 1) & 1