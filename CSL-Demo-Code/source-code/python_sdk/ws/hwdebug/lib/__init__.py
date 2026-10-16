# pylint: disable=missing-module-docstring
try:
    # These imports are required to avoid conflicts between python
    # and pybind protobud messages.
    # See https://github.com/Cerebras/monolith/pull/85914 for more details.
    import cerebras.pb.workflow.appliance.common.common_config_pb2
except ImportError:
    # This try/except is required to avoid import errors in tests where
    # cerebras.pb is not available. For example: test_cbcore_file_include.py
    pass
