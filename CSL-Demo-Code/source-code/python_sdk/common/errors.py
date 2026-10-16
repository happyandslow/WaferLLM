"""
ErrorTypes for external use which are specific to Cerebras software.
"""

################################################################################
## Base exceptions
################################################################################


class CerebrasErrorMetaclass(type):
    """
    Metaclass for providing proper pickling/unpickling of exceptions.

    In the default mechanism, postional arguments passed to Exception.__init__
    are stored in an instance 'args' attribute. Later, when unpickling the
    exception, these pickled arguments are passed to the original exception to
    reconstruct the exception. In other words, CustomException.__init__(*args)
    is called.

    If all custom exceptions call super().__init__(*args), then the default
    pickling mechanism works. However, when a custom exception consumes some
    arguments and doesn't pass them along to super(), then unpickling fails
    because the required number of arguments don't exist in Exception.args.
    For more background, see: https://bugs.python.org/issue1692335

    The aim of this metaclass is to wrap around CustomException.__init__ and
    store the passed args/kwargs, and use them when restoring the exception
    through unpickling.
    """
    def __new__(mcs, name, bases, attrs):
        # If a custom init is not provided, the default mechanism works, so
        # nothing needs to be done.
        original_init = attrs.get('__init__', None)
        if original_init is not None:
            # args/kwargs passed to the exception are stored in these variables.
            init_args = tuple()
            init_kwargs = dict()

            def _init_wrapper(self, *args, **kwargs):
                # Store args/kwargs
                nonlocal init_args
                nonlocal init_kwargs
                init_args = args
                init_kwargs = kwargs

                # Call the actual __init__ method
                original_init(self, *args, **kwargs)

            # Pickle calls this function during unpickling
            # https://docs.python.org/3/library/pickle.html#object.__reduce__
            def _reduce_ex_wrapper(self, protocol):
                nonlocal init_args
                nonlocal init_kwargs
                return (
                    mcs._obj_reconstructor,
                    (self.__class__, init_args, init_kwargs)
                )

            # Override some magic methods of the class
            attrs['__init__'] = _init_wrapper
            attrs['__reduce_ex__'] = _reduce_ex_wrapper

        return super().__new__(mcs, name, bases, attrs)

    @classmethod
    def _obj_reconstructor(mcs, cls, args, kwargs):
        """
        The callable called to create the initial version of the object.
        """
        return cls(*args, **kwargs)


class CerebrasBaseError(Exception, metaclass=CerebrasErrorMetaclass):
    """
    Base class for Cerebras' specific exceptions.
    """
    pass


class CerebrasUserError(CerebrasBaseError):
    """
    Base class for Cerebras' specific exceptions which are user errors and
    intentionally thrown in our stack as feedback to the users.
    """
    pass


class CerebrasHeaderError(CerebrasBaseError):
    """
    Base class for cerebras errors which will have a header attached to them at
    the beginning of the error.
    """

    def __init__(self, message, header):
        """
        Initializer for this exception. Makes sure that a string header is
        added to the beginning of the error message. Does not take in account
        casing for the header start.
        """
        self._header = header
        header = header.lower() if header else header
        if header and (not message.lower().startswith(header)):
            # Then let's add it to the message. Note that case is preserved
            # when this gets added to the error message.
            message = self._header + message
        super().__init__(message)


################################################################################
## User errors which will be "intentionally" thrown as feedback
################################################################################


class TFUnSupportedError(CerebrasHeaderError, CerebrasUserError):
    """
    TFUnSupportedError class for features we do not support yet in our
    framework import.
    """

    def __init__(self, message):
        """
        Initializer for unsupported TF operation.
        """
        super().__init__(
            message=message,
            header="[Unsupported TensorFlow Operation] ",
        )


class IRLoweringError(CerebrasHeaderError, CerebrasUserError):
    """
    Class for IR lowering error. These are errors coming from the MLIR
    translation utilities.
    """

    def __init__(self, message):
        """
        Initializer for MLIR Lowering error
        """
        super().__init__(
            message=message,
            header="[MLIR Translation Error] ",
        )


class LairTranslationError(CerebrasHeaderError, CerebrasUserError):
    """
    Class for lair translation error. This are error coming from possibly
    corrupted or unsupported XLA graphs into our LAIR compilation toolchain.
    """

    def __init__(self, message):
        """
        Initializer for failed LAIR translation.
        """
        super().__init__(
            message=message,
            header="[LAIR Translation Error] ",
        )


################################################################################
## Errors that may or may not come as issues from the user but it is unclear
## due to the callee not being clear about the exception.
################################################################################


class CerebrasCompileError(CerebrasHeaderError):
    """
    Error used for issues in our compilation stack when a specific stage
    failed.
    """

    def __init__(self, message, subtype=None, source=None):
        """
        @param      self     The object
        @param      message  The message to be used for this error.
        @param      stage    The stage in our compilation stack where this
                             failure happened.
        """
        modifiers = []
        if subtype:
            modifiers.append('subtype "{}"'.format(subtype))
        if source:
            modifiers.append('source "{}"'.format(source))
        inner = " ({})".format(", ".join(modifiers)) if modifiers else ""

        super().__init__(
            message=message,
            header='[Cerebras Compile Error{}] '.format(inner),
        )


class CerebrasInternalError(CerebrasHeaderError):
    """
    Error used for issues in our compilation stack when a specific stage
    failed and we do no have a specific handler for it. It will ask users to
    contact Cerebras for help in that case.
    """

    def __init__(self, message, subtype=None, source=None):
        """
        @param      self     The object
        @param      message  The message to be used for this error.
        @param      stage    The stage in our compilation stack where this
                             failure happened.
        """
        self.original_message = message
        self.subtype = subtype
        self.source = source

        if not message.endswith("\n"):
            # Then let's add it
            message += "\n"
        # And add the contact string at the end of this message
        end = "Please contact Cerebras support team."
        if not message.strip().lower().endswith(end.lower()):
            message += end

        modifiers = []
        if subtype:
            modifiers.append('subtype "{}"'.format(subtype))
        if source:
            modifiers.append('source "{}"'.format(source))
        inner = " ({})".format(", ".join(modifiers)) if modifiers else ""

        super().__init__(
            message=message,
            header='[Cerebras Internal Error{}] '.format(inner),
        )


################################################################################
# Typed exceptions
################################################################################

class CerebrasMessageException(CerebrasInternalError):
    """
    A generic error used when the underlying binary (e.g., stack_main) exits
    with a fatal-level message.
    """

class CerebrasConverterMissingError(CerebrasCompileError):
    """
    Error used when a converter is missing for some operations in our kernel
    matching stage.
    """

class CerebrasInvalidKernelParameter(CerebrasInternalError):
    """
    Error used when an invalid kernel parameter is specified.
    """

class CerebrasConditionViolation(CerebrasInternalError):
    """
    A generic error used when the a condition on a stage port is violated.
    """

class CerebrasKernelMissingError(CerebrasCompileError):
    """
    Error used when a kernel is missing for some operations in our graph.
    """

class CerebrasKernelMatchingError(CerebrasCompileError):
    """
    Error used when a kernel is missing for some operations in our kernel
    matching stage.
    """

_TYPED_EXCEPTIONS = {
    "cerebras.common.errors.CerebrasConverterMissingError": CerebrasConverterMissingError,
    "cerebras.common.errors.CerebrasInvalidKernelParameter": CerebrasInvalidKernelParameter,
    "cerebras.common.errors.CerebrasKernelMissingError": CerebrasKernelMissingError,
    "cerebras.common.errors.CerebrasKernelMatchingError": CerebrasKernelMatchingError,
}

_MESSAGE_EXCEPTIONS = {
    "SIML003": CerebrasConverterMissingError,
    "SMAK001": CerebrasConverterMissingError,
    "SREN001": CerebrasConverterMissingError,
    "KAPI001": CerebrasInvalidKernelParameter,
    "KAPI002": CerebrasInvalidKernelParameter,
    "KAPI003": CerebrasInvalidKernelParameter,
    "KAPI004": CerebrasInvalidKernelParameter,
    "KAPI030": CerebrasKernelMissingError,
    "SRMK001": CerebrasKernelMissingError,
    "SIMM005": CerebrasKernelMatchingError,
}

_CONDITION_VIOLATIONS = {
    "NoLeftoverNodes": CerebrasKernelMatchingError,
    "WhitelistedLeftoverNodes": CerebrasKernelMatchingError,
}

def typed_exception(typename, subtype):
    """Returns a stack error for the given string tag."""
    if typename == "MessageException":
        exc_cls = _MESSAGE_EXCEPTIONS.get(subtype)
        if not exc_cls:
            # Then create a dynamic type for it, subclassing from
            # CerebrasMessageException
            exc_cls = type(
                "Cerebras" + subtype + "Exception",
                (CerebrasMessageException,),
                {"subtype": subtype}
            )
            # memoize this exact anonymous type, to allow use in python xfail
            _MESSAGE_EXCEPTIONS[subtype] = exc_cls
        return exc_cls
    elif typename == "ConditionViolation":
        return _CONDITION_VIOLATIONS.get(subtype, CerebrasConditionViolation)
    else:
        return _TYPED_EXCEPTIONS.get(subtype, CerebrasInternalError)

def serialize_exc_to_proto(exc_info=None):
    """Serializes the exception to a proto exception.

    Args:
        exc_info: The exception to save. If None, the current active exception
            is used.
    Returns:
        The exception serialized to a `StandardExceptionProto` protobuf message.
    """
    if not exc_info:
        import sys
        exc_info = sys.exc_info()
    exc_type, exc_value, exc_traceback = exc_info
    from cerebras.pb.common.standard_exception_pb2 import StandardExceptionProto
    exc_module = exc_type.__module__
    if exc_module == "builtins":
        exc_type_str = exc_type.__qualname__
    else:
        exc_type_str = exc_module + "." + exc_type.__qualname__

    proto = StandardExceptionProto(
        what=str(exc_value), # string representation
        type="python",
        subtype=exc_type_str
    )
    # May be a dynamic type created from typed_exception, which can't pickle
    if issubclass(exc_type, CerebrasMessageException):
        proto.type = "MessageException"
        proto.subtype = exc_value.subtype
        proto.what = exc_value.original_message
        proto.source = exc_value.source
    else:
        try:
            import pickle, base64
            from tblib import pickling_support
            pickling_support.install()
            proto.source = base64.standard_b64encode(
                pickle.dumps(exc_info, protocol=pickle.HIGHEST_PROTOCOL)
            )
            proto.type = "pickled_python"
        except:
            pass

    return proto


def save_exc_to_jsonfile(filename, exc_info=None):
    """Serializes the exception to the given json file.

    Args:
        filename: The json filepath to save the exception to.
        exc_info: The exception to save. If None, the current active exception
            is used.
    """
    from cerebras.common.protobuf import proto_msg_to_jsonfile
    proto = serialize_exc_to_proto(exc_info)
    proto_msg_to_jsonfile(proto, filename)


def deserialize_exc_from_proto(proto):
    """Deserializes exception contained in the proto message.

    Args:
        proto: The exception serialized as a `StandardExceptionProto` message.
    Returns:
        A tuple of (exc_type, exc_value, exc_traceback).
    """
    if proto.type == "pickled_python":
        import pickle, base64
        exc_info = pickle.loads(base64.standard_b64decode(proto.source))
    else:
        exc_type = typed_exception(proto.type, proto.subtype)
        exc_value = exc_type(
            proto.what,
            subtype=proto.subtype,
            source=proto.source,
        )
        exc_info = (exc_type, exc_value, None)
    return exc_info


def raise_exc_from_jsonfile(filename):
    """Raises an exception from file containing a json-serialized message.

    The json-serialized message must be an instance of `StandardExceptionProto`.

    Args:
        filename: The json file containing the exception to raise.
    """
    import six
    from cerebras.common.protobuf import proto_msg_from_jsonfile
    from cerebras.pb.common.standard_exception_pb2 import StandardExceptionProto
    proto = proto_msg_from_jsonfile(StandardExceptionProto, filename)
    exc_info = deserialize_exc_from_proto(proto)
    six.reraise(*exc_info)
