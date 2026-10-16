---
title: "SdkLayout 1: Introduction"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-01-introduction"
crawled_at: "2026-02-14 23:55:34"
---

# SdkLayout 1: Introduction

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-01-introduction](https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-01-introduction)

This tutorial introduces the `SdkLayout` API. `SdkLayout`
allows us to define and compile multi-PE WSE programs. Specifically,
it consists of the following main features:

- Creation of CSL code regions: rectangular CSL code regions can be
  instantiated given a CSL source code file path, a name, and the
  width and height dimensions.
- Routing and switching: for a given CSL code region we can specify
  routing and switching information on a single PE within the code
  region, on a rectangular sub-region, or on the entire code region.
  See tutorial [SdkLayout 2: Basic routing](csl/code-examples/tutorial-sdklayout-02-routing.md#sdkruntime-sdklayout-02-routing).
- Automatic color allocation: routing can be done based on symbolic
  colors. The `SdkLayout` engine will then allocate physical
  values automatically. See tutorials [SdkLayout 2: Basic routing](csl/code-examples/tutorial-sdklayout-02-routing.md#sdkruntime-sdklayout-02-routing)
  and [SdkLayout 3: Ports and connections](csl/code-examples/tutorial-sdklayout-03-ports-and-connections.md#sdkruntime-sdklayout-03-ports-and-connections).
- Automatic routing between code regions: users can create input
  and output ports on code regions and connect them. The `SdkLayout`
  engine will automatically find optimal routes between them.
  See tutorial [SdkLayout 3: Ports and connections](csl/code-examples/tutorial-sdklayout-03-ports-and-connections.md#sdkruntime-sdklayout-03-ports-and-connections).
- Host-to-device and device-to-host connections: an input or
  output port can be connected to the host to create an input
  or output stream respectively. See tutorial
  [SdkLayout 4: Host-to-device and device-to-host data streaming](csl/code-examples/tutorial-sdklayout-04-h2d-d2h.md#sdkruntime-sdklayout-04-h2d-d2h).

This tutorial demonstrates the most basic compilation flow,
where a single-PE program with no colors and no routing sets the value
of a global variable in device memory based on the value of
a parameter.

## gv.csl

```csl
// Code parameter specified by the host using 'set_param_all'.
param value: i16;
export var gv: i16;

const main_id = @get_local_task_id(8);
task main() void {
  gv = value;
}

comptime {
  @bind_local_task(main, main_id);
  @activate(main_id);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    SdkTarget,
    SdkLayout,
    SimfabConfig,
    get_platform,
)

parser = argparse.ArgumentParser()
parser.add_argument('--cmaddr', help='IP:port for CS system')
parser.add_argument(
    '--arch',
    choices=['wse2', 'wse3'],
    default='wse3',
    help='Target WSE architecture (default: wse3)'
)
args = parser.parse_args()

########################################
### Layout, code region and compilation
########################################
value = 550
# If 'cmaddr' is empty then we create a default simulation layout.
# If 'cmaddr' is not empty then 'config' and 'target' are ignored.
config = SimfabConfig(dump_core=True)
target = SdkTarget.WSE3 if (args.arch == 'wse3') else SdkTarget.WSE2
platform = get_platform(args.cmaddr, config, target)
layout = SdkLayout(platform)
# Create a 1x1 code region using 'gv.csl' as the source code.
# The code region is called 'gv'.
code = layout.create_code_region('./gv.csl', 'gv', 1, 1)
# Set the 'value' param on all PEs in the region. In this
# example 'code' has only one PE.
code.set_param_all('value', value)
# Compile the layout and use 'out' as the prefix for all
# produced artifacts.
compile_artifacts = layout.compile(out_prefix='out')

############
### Runtime
############
# Create the runtime using the compilation artifacts and the execution platform.
runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()
runtime.stop()

#################
### Verification
#################
# Finally, once execution has stopped, read 'value' from device memory
# and compare it against the expected value.
result = runtime.read_symbol(0, 0, 'gv', dtype='uint16')
assert result == [value]
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cs_python run.py --arch=wse3

```
[previous

Pipeline 3: Add an artificial halo](csl/code-examples/tutorial-pipeline-03-multiple.md "previous page")
[next

SdkLayout 2: Basic routing](csl/code-examples/tutorial-sdklayout-02-routing.md "next page")
