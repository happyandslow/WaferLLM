import os
from cerebras.util.py.utilpy import get_environ

# os.environ contains a copy of the environment.
# Therefore changes to the environment by C/C++
# code are not reflected in os.environ. This
# function will bring them back in sync.
def sync_environ():
    # This implementation is a bit silly in that
    # when os.environ is updated it will push the
    # updates to the os automatically. Of course this
    # is needlessly setting the values in the os to
    # themselves when all we want is to update the
    # cached values. The alternative is to modify
    # the internal members but this is not portable.

    # First get the environment directly from the os
    env = get_environ()

    # Second remove items that are not in the environment
    pop_list = []
    for name in os.environ.keys():
        if name not in env:
            pop_list.append(name)
    
    for name in pop_list:
        os.environ.pop(name)

    # Third update names that are in the environment
    for name, value in env.items():
        os.environ[name] = value