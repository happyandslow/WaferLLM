---
title: "SDK Appliance API Reference"
url: "https://sdk.cerebras.net/api-docs/appliance-api"
crawled_at: "2026-02-14 23:55:44"
---

# SDK Appliance API Reference

> Source: [https://sdk.cerebras.net/api-docs/appliance-api](https://sdk.cerebras.net/api-docs/appliance-api)

This section presents the SDK Appliance API reference for running SDK
programs on a Cerebras Wafer-Scale Cluster (WSC).

See [Running SDK on a Wafer-Scale Cluster](appliance-mode.md#appliance-mode) for an introduction to running in appliance mode
on a Wafer-Scale Cluster.

## SdkCompiler

Python API for compiling SDK programs on a Cerebras Wafer-Scale Cluster.

*class* `cerebras.sdk.client.``SdkCompiler`(*\*\*kwargs*)
:   Bases: `object`

    Manages the generation of compile artifacts on a Cerebras Wafer-Scale
    Cluster using the CSL compiler.

    [`SdkCompiler`](#cerebras.sdk.client.SdkCompiler "cerebras.sdk.client.SdkCompiler") must be used via a context manager.

    Keyword Arguments
    :   - **mgmt\_namespace** (`str`) –
          Appliance cluster namespace to which the job is submitted. Default is
          the default namespace.
        - **resource\_cpu** (`int`) –
          CPU cores on the WSC management node used by the compile job in units
          of 1/1000 CPU (default: 24000, or 24 cores)
        - **resource\_mem** (`int`) –
          Memory in bytes requested from the management node for the compile job
          (default: `64 << 30`, or 64 GiB)
        - **disable\_version\_check** (`bool`) – If `True`, ignore version
          differences between appliance client and server

    **Example**:

    > In the following example, an [`SdkCompiler`](#cerebras.sdk.client.SdkCompiler "cerebras.sdk.client.SdkCompiler") object is instantiated
    > via a context manager.
    > The [`SdkCompiler.compile()`](#cerebras.sdk.client.SdkCompiler.compile "cerebras.sdk.client.SdkCompiler.compile") function takes four arguments:
    >
    > - the directory containing the CSL code files,
    > - the name of the top level CSL code file that contains the layout block,
    > - the compiler arguments,
    > - and the output directory or output file for the compile artifacts.
    >
    > ```python
    > import json
    > from cerebras.sdk.client import SdkCompiler
    >
    > # Instantiate compiler using a context manager
    > with SdkCompiler(disable_version_check=True) as compiler:
    >
    >     # Launch compile job
    >     artifact_path = compiler.compile(
    >         ".",
    >         "layout.csl",
    >         "--fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out",
    >         "."
    >     )
    >
    >     # Write the artifact_path to a JSON file
    >     with open("artifact_path.json", "w", encoding="utf8") as f:
    >         json.dump({"artifact_path": artifact_path,}, f)
    >
    > ```

    `compile`(*codedir: str*, *layout: str*, *args: str*, *outdir: str*) → str
    :   Generates compile artifacts using the CSL compiler.

        Parameters
        :   - **codedir** (`str`) – Directory containing CSL code files.
            - **layout** (`str`) – Top-level CSL file containing the layout block.
            - **args** (`str`) – Arguments passed to the CSL compiler.
            - **outdir** (`str`) – Output directory or file for compile artifacts.

        Returns
        :   String containing local path to compile artifacts.

        Return type
        :   `str`

## SdkLauncher

*class* `cerebras.sdk.client.``SdkLauncher`(*artifact\_path: str*, *\*\*kwargs*)
:   Bases: `object`

    The SdkLauncher API can be used to upload artifacts, run custom commands in
    the appliance, and use custom scripts written as if the system was not in
    appliance mode and the user were running directly from a worker node.

    The user must use the `%CMADDR%` template string to pass the system address
    to a run script.

    Parameters
    :   **artifact\_path** (`str`) – Path to a compiled artifact which will be transferred.
        and extracted in the appliance.

    Keyword Arguments
    :   - **simulator** (`bool`) –
          If `True`, runs the program on the simulator using a worker node of
          the Wafer-Scale Cluster.
          Default value is `False`, i.e., allocates and runs on a WSE.
        - **mgmt\_namespace** (`str`) –
          Appliance cluster namespace to which the job is submitted. Default is
          the default namespace.
        - **resource\_cpu** (`int`) –
          CPU cores on the WSC management node used by the compile job in units
          of 1/1000 CPU (default: 24000, or 24 cores)
        - **resource\_mem** (`int`) –
          Memory in bytes requested from the management node for the compile job
          (default: `64 << 30`, or 64 GiB)
        - **disable\_version\_check** (`bool`) – If `True`, ignore version
          differences between appliance client and server

    **Example**:

    > In the following example, an [`SdkLauncher`](#cerebras.sdk.client.SdkLauncher "cerebras.sdk.client.SdkLauncher") object is instantiated via a
    > context manager, with path to compile artifacts given by `artifact_path`.
    > `launcher.stage` transfers an additional file `additional_artifact.txt`
    > to the appliance. Next, `launcher.run` runs a command on the appliance
    > worker node which writes the contents of that file to stdout.
    > We then use `launcher.run` to run a host Python script `run.py`.
    > Notice that we specify the system’s CM address passed to this script via
    > the template string `%CMADDR`, which will be evaluated at runtime based
    > on the system allocated to this job. We then use `download_artifact`
    > to transfer a file back from the appliance.
    >
    > ```python
    > import json
    > import os
    >
    > from cerebras.sdk.client import SdkLauncher
    >
    > # read the compile artifact_path from the json file
    > with open("artifact_path.json", "r", encoding="utf8") as f:
    >     data = json.load(f)
    >     artifact_path = data["artifact_path"]
    >
    > # artifact_path contains the path to the compiled artifact.
    > # It will be transferred and extracted in the appliance.
    > # The extracted directory will be the working directory.
    > # Set simulator=False if running on CS system within appliance.
    > with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:
    >
    >     # Transfer an additional file to the appliance,
    >     # then write contents to stdout on appliance
    >     launcher.stage("additional_artifact.txt")
    >     response = launcher.run(
    >         "echo \"ABOUT TO RUN IN THE APPLIANCE\"",
    >         "cat additional_artifact.txt",
    >     )
    >     print("Test response: ", response)
    >
    >     # Run the original host code as-is on the appliance,
    >     # using the same cmd as when using the Singularity container
    >     response = launcher.run("cs_python run.py --name out --cmaddr %CMADDR%")
    >     print("Host code execution response: ", response)
    >
    >     # Fetch files from the appliance
    >     launcher.download_artifact("out.txt", "./output_dir/out.txt")
    >
    > ```

    `download_artifact`(*artifact\_name: str*, *out\_path: str*) → str
    :   Downloads an artifact from the appliance. If the artifact is a directory,
        a tarball of that directory will be created and transferred.

        Parameters
        :   - **artifact\_name** (`str`) – Name of the artifact to download.
            - **out\_path** (`str`) – Top-level CSL file containing the layout block.

        Returns
        :   The name of the file that has been written (can contain a `.tar.gz`
            extension if the `artifact_name` was a directory.)

        Return type
        :   `str`

    `stage(artifact_path: str):`
    :   Stages additional artifacts in the remote working directory within the appliance.

        Parameters
        :   **artifact\_path** (`str`) – Local path to the artifact to be staged on the appliance.

    `run`(*\*commands*)
    :   Run one or more shell commands on the appliance.

        Parameters
        :   **\*commands** – One or more command strings. Use the special placeholder
            `%CMADDR%` wherever a CS system address should be
            substituted. **All** positional arguments must be strings.

## SdkRuntime

> **Note**
> Note
> The `SdkRuntime` appliance bindings are deprecated. Use `SdkLauncher` to
> wrap an SDK host Python script instead.

*class* `cerebras.sdk.client.``SdkRuntime`(*artifact\_path: str*, *\*\*kwargs*)
:   Bases: `object`

    Manages the execution of SDK programs on the Cerebras Wafer-Scale Cluster
    appliance. The constructor analyzes the WSE ELFs in the `bindir`
    and prepares the WSE or simfabric for a run.

    [`SdkRuntime`](#cerebras.sdk.client.SdkRuntime "cerebras.sdk.client.SdkRuntime") must be used via a context manager.

    Parameters
    :   **artifact\_path** (`str`) – Path to ELF files compiled by [`SdkCompiler`](#cerebras.sdk.client.SdkCompiler "cerebras.sdk.client.SdkCompiler").
        The runtime collects the I/O and fabric parameters
        automatically, including height, width, number of
        channels, width of buffers,… etc.

    Keyword Arguments
    :   - **simulator** (`bool`) –
          If `True`, runs the program on simulator using a worker node of
          the Wafer-Scale Cluster.
          Default value is `False`, i.e., allocates and runs on a WSE.
        - **mgmt\_namespace** (`str`) –
          Appliance cluster namespace to which the job is submitted. Default is
          the default namespace.
        - **resource\_cpu** (`int`) –
          CPU cores on the WSC management node used by the compile job in units
          of 1/1000 CPU (default: 24000, or 24 cores)
        - **resource\_mem** (`int`) –
          Memory in bytes requested from the management node for the compile job
          (default: `64 << 30`, or 64 GiB)
        - **disable\_version\_check** (`bool`) – If `True`, ignore version
          differences between appliance client and server

    **Example**:

    > In the following example, an `SdkRuntime` runner object is instantiated
    > via a context manager, with path to compile artifacts given by
    > `artifact_path`. The compiled kernel code has exported symbols
    > `my_fn`, which names a function defined on all PEs in the program, and
    > `A`, which points to an array on all PEs. The context manager loads and
    > starts the program. Then, the function `my_fn` is launched. After this
    > function is launched, `A` on the device is copied back into `data`
    > on the host.
    >
    > ```python
    > import json
    > import os
    >
    > from cerebras.sdk.client import SdkRuntime
    >
    > # Read the artifact_path from the JSON file
    > with open("artifact_path.json", "r", encoding="utf8") as f:
    >     data = json.load(f)
    >     artifact_path = data["artifact_path"]
    >
    > # Instantiate a runner object using a context manager.
    > # Set simulator=False if running on CS system within appliance.
    > with SdkRuntime(artifact_path, simulator=False, disable_version_check=True) as runner:
    >     # Launch my_fn on device
    >     runner.launch('my_fn', nonblock=False)
    >
    >     # Copy A back from device
    >     symbol_A = runner.get_id("A")
    >     runner.memcpy_d2h(data, symbol_A, px, py, w, h, l,
    >                       streaming=False, data_type=memcpy_dtype,
    >                       order=memcpy_order, nonblock=False)
    >
    > ```

    `get_id`(*symbol: str*)
    :   See [SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

    `is_task_done`(*task\_handle: [Task](#cerebras.sdk.client.Task "cerebras.sdk.client.Task")*) → bool
    :   See [SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

    `launch`(*symbol: str*, *\*args*, *\*\*kwargs*) → [Task](#cerebras.sdk.client.Task "cerebras.sdk.client.Task")
    :   See [SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

    `memcpy_d2h`(*dest: numpy.ndarray*, *src: int*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.client.Task "cerebras.sdk.client.Task")
    :   See [SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

    `memcpy_h2d`(*dest: int*, *src: numpy.ndarray*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.client.Task "cerebras.sdk.client.Task")
    :   See [SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

    `task_wait`(*task\_handle: [Task](#cerebras.sdk.client.Task "cerebras.sdk.client.Task")*)
    :   See [SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

*class* `cerebras.sdk.client.``Task`
:   Handle to a task launched by [`SdkRuntime`](#cerebras.sdk.client.SdkRuntime "cerebras.sdk.client.SdkRuntime").

*class* `cerebras.appliance.pb.sdk.sdk_common_pb2.``MemcpyDataType`
:   Bases: `Enum`

    Specifies the data size for transfers using `SdkRuntime.memcpy_d2h()` and
    `SdkRuntime.memcpy_h2d()` copy mode.

    Values
    :   - **MEMCPY\_16BIT**
        - **MEMCPY\_32BIT**

*class* `cerebras.appliance.pb.sdk.sdk_common_pb2.``MemcpyOrder`
:   Bases: `Enum`

    Specifies mapping of data for transfers using `SdkRuntime.memcpy_d2h()` and
    `SdkRuntime.memcpy_h2d()`.

    Values
    :   - **ROW\_MAJOR**
        - **COL\_MAJOR**

## sdk\_utils

Utility functions for common operations with [`SdkRuntime`](#cerebras.sdk.client.SdkRuntime "cerebras.sdk.client.SdkRuntime").
Import from `cerebras.sdk.client.sdk_utils`.

See [sdk\_utils module](api-docs/sdkruntime-api.md#sdkruntime-sdk-utils).

## debug\_util

Utilities for parsing debug output and core files of a simulator run.
Import from `cerebras.sdk.client.debug_util`.

See [debug\_util module](api-docs/sdkruntime-api.md#sdkruntime-debug-util).

[previous

SdkLayout API Reference](api-docs/sdklayout-api.md "previous page")
