"""
Convert between different names of data types.

See also:
  src/pb/common/cerebras_type.proto
"""
import numpy as np
from cerebras.pb.common.cerebras_type_pb2 import CerebrasType as CerebrasTypePB
from cerebras.das.context.daspybind import CerebrasType as CerebrasTypeEnum


# Cerebras name -> numpy name
_CS_TO_NP_TYPENAMES_MAP = {
    'T_BOOL': 'bool',
    'T_F8E5': 'float32',
    'T_F8E4': 'float32',
    'T_BF16': 'float32',
    'T_CB16': 'float32',
    'T_F16': 'float16',
    'T_F32': 'float32',
    'T_F64': 'float64',
    'T_I8': 'int8',
    'T_I16': 'int16',
    'T_I32': 'int32',
    'T_I64': 'int64',
    'T_U8': 'uint8',
    'T_U16': 'uint16',
    'T_U32': 'uint32',
    'T_U64': 'uint64',
}
# numpy name -> Cerebras name
_NP_TO_CS_TYPENAMES_MAP = {
    'bool': 'T_BOOL',
    'float16': 'T_F16',
    'float32': 'T_F32',
    'float64': 'T_F64',
    'int8': 'T_I8',
    'int16': 'T_I16',
    'int32': 'T_I32',
    'int64': 'T_I64',
    'uint8': 'T_U8',
    'uint16': 'T_U16',
    'uint32': 'T_U32',
    'uint64': 'T_U64',
}
# numpy short-hand synonyms
_NP_ALIAS_TO_NP_TYPENAMES_MAP = {
    'f': 'float32',
    'f2': 'float16',
    'f4': 'float32',
    'f8': 'float64',
    'f16': 'float128',
    'double': 'float64',
    'longdouble': 'float128',
    'i1': 'int8',
    'i2': 'int16',
    'i4': 'int32',
    'i8': 'int64',
    'long': 'int64',
    'u1': 'uint8',
    'u2': 'uint16',
    'u4': 'uint32',
    'u8': 'uint64',
}


def to_numpy(value, typename):
    """
    Convenience function to convert a native Python number to a numpy
    number of the given type.

    :param value: int or float
    :param str typename: numpy type name

    Example:
      >>> val = to_numpy(3.5, 'float16')
      >>> str(val)
      '3.5'
      >>> type(val)
      <class 'numpy.float16'>
    """
    return np.dtype(typename).type(value)


def typename_from_np_to_cs(typename):
    """
    Convert data type names from numpy names to plan.meta names
    Returns a string

    Examples:
      >>> typename_from_np_to_cs('uint16')
      'T_U16'
      >>> typename_from_np_to_cs('u2')
      'T_U16'
    """
    # If passed in a native python type, convert it to numpy:
    if ((typename == 'float' or
         (isinstance(typename, type) and issubclass(typename, float)))):
        typename = 'float64'  # numpy interprets native float as 64-bit
    elif ((typename == 'int' or
           (isinstance(typename, type) and issubclass(typename, int)))):
        typename = 'int64'  # numpy interprets native int as 64-bit
    # numpy class:
    if isinstance(typename, type) and issubclass(typename, np.number):
        typename = str(np.dtype(typename))
    # nympy type name:
    typename = str(typename)
    if typename in _NP_ALIAS_TO_NP_TYPENAMES_MAP:
        typename = _NP_ALIAS_TO_NP_TYPENAMES_MAP[typename]
    try:
        return _NP_TO_CS_TYPENAMES_MAP[typename]
    except KeyError:
        raise ValueError("Do not understand type name " + repr(typename))


def typename_from_cs_to_np(typename):
    """
    Convert data type names from plan.meta names to numpy names
    Returns a string

    Examples:
      >>> # Here we use CerebrasTypeEnum for consistency, but if
      >>> # imported alone you might be doing CerebrasType.T_...
      >>> typename_from_cs_to_np('CerebrasTypeEnum.T_F32')
      'float32'
      >>> typename_from_cs_to_np('CerebrasTypeEnum.T_U8')
      'uint8'
      >>> typename_from_cs_to_np('T_F32')
      'float32'
      >>> typename_from_cs_to_np('T_U8')
      'uint8'
    """
    if isinstance(typename, CerebrasTypeEnum):
        typename = typename.name
    elif isinstance(typename, int):
        typename = CerebrasTypePB.Name(typename)
    elif not isinstance(typename, str):
        raise ValueError(f"typename_from_cs_to_np expected string or int, got {type(typename)}")
    try:
        return _CS_TO_NP_TYPENAMES_MAP[typename]
    except KeyError:
        raise ValueError("Do not understand type name " + repr(typename))


def np_dtype_from_cb_typename(cb_typename: str) -> np.dtype:
    """Converts to a numpy dtype from cb typename
    """
    try:
        return np.dtype(typename_from_cs_to_np(cb_typename))
    except KeyError:
        raise ValueError("Do not understand type name " + repr(cb_typename))
