---
title: "SdkRuntime API Reference"
url: "https://sdk.cerebras.net/api-docs/sdkruntime-api"
crawled_at: "2026-02-14 23:55:37"
---

# SdkRuntime API Reference

> Source: [https://sdk.cerebras.net/api-docs/sdkruntime-api](https://sdk.cerebras.net/api-docs/sdkruntime-api)

This section presents the `SdkRuntime` Python host API reference and
associated utilities to develop kernels for the Cerebras Wafer-Scale Engine.

## sdkruntimepybind module

Python API for `SdkRuntime` functions.

### MemcpyDataType

*class* `cerebras.sdk.runtime.sdkruntimepybind.``MemcpyDataType`
:   Bases: `Enum`

    Specifies the data size for transfers using [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h") and
    [`SdkRuntime.memcpy_h2d()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d") copy mode.

    Values
    :   - **MEMCPY\_16BIT**
        - **MEMCPY\_32BIT**

### MemcpyOrder

*class* `cerebras.sdk.runtime.sdkruntimepybind.``MemcpyOrder`
:   Bases: `Enum`

    Specifies mapping of data for transfers using [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h") and
    [`SdkRuntime.memcpy_h2d()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d") copy mode.

    Values
    :   - **ROW\_MAJOR**
        - **COL\_MAJOR**

### SdkCompileArtifacts

*class* `cerebras.sdk.runtime.sdkruntimepybind.``SdkCompileArtifacts`(*artifacts\_path: str*)
:   Bases: `object`

    Specifies compile artifacts for execution.

    Parameters
    :   **artifacts\_path** (`str`) – Path to compile artifacts.

### SdkExecutionPlatform

*class* `cerebras.sdk.runtime.sdkruntimepybind.``SdkExecutionPlatform`
:   Bases: `object`

    Specifies the simulator or system target and architecture for execution.

    `is_simulation`()
    :   Queries if the execution platform is a simulator.

        Returns
        :   `True` if the execution platform is a simulator, `False` otherwise.

        Return type
        :   `bool`

    `is_system`()
    :   Queries if the execution platform is a real system.

        Returns
        :   `True` if the execution platform is a real system, `False` otherwise.

        Return type
        :   `bool`

### SdkRuntime

*class* `cerebras.sdk.runtime.sdkruntimepybind.``SdkRuntime`(*bindir: Union[pathlib.Path, str]*, *\*\*kwargs*)
:   Bases: `object`

    Manages the execution of SDK programs on the Cerebras Wafer Scale Engine
    (WSE) or simfabric. The constructor analyzes the WSE ELFs in the `bindir`
    and prepares the WSE or simfabric for a run.
    Requires CM IP address and port for WSE runs.

    Parameters
    :   **bindir** (`Union[pathlib.Path, str]`) – Path to ELF files which is compiled by `cslc`.
        The runtime collects the I/O and fabric parameters
        automatically, including height, width, number of channels,
        width of buffers,… etc.

    Keyword Arguments
    :   - **cmaddr** (`str`) –
          `'IP_ADDRESS:PORT'` string of CM. Omit this `kwarg` to run on
          simfabric.
        - **suppress\_simfab\_trace** (`bool`) –
          If `True`, suppresses generation of `simfab_traces` when running.
          Default value is `False`, i.e., `simfab_traces` are produced.
        - **simfab\_numthreads** (`int`) –
          Number of threads to use if running on simfabric.
          Maximum value is `64`. Default value is `5`, i.e., the simulator
          uses 5 threads.
        - **msg\_level** (`str`) –
          Message logging output level. Available output levels are `DEBUG`,
          `INFO`, `WARNING`, and `ERROR`. Default value is `WARNING`.

    **Example**:

    > In the following example, an `SdkRuntime` runner object is
    > instantiated. If `args.cmaddr` is non-empty, then the kernel code will
    > run on the WSE pointed to by that address; otherwise, the kernel code
    > will run on simfabric. The compiled kernel code in the directory
    > `args.name` has exported symbols `A` and `B` pointing to arrays on
    > the device. After loading the code and starting the run with `load()`
    > and `run()`, data on the host stored in `data` is copied to `A`
    > on the device, and then `B` on the device is copied back into `data`
    > on the host.
    >
    > ```python
    > runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    > symbol_A = runner.get_id("A")
    > symbol_B = runner.get_id("B")
    > runner.load()
    > runner.run()
    > runner.memcpy_h2d(symbol_A, data, px, py, w, h, l,
    >                   streaming=False, data_type=memcpy_dtype,
    >                   order=memcpy_order, nonblock=False)
    > runner.memcpy_d2h(data, symbol_B, px, py, w, h, l,
    >                   streaming=False, data_type=memcpy_dtype,
    >                   order=memcpy_order, nonblock=False)
    >
    > ```

    `__init__`(*bindir: Union[pathlib.Path, str]*, *platform: [SdkExecutionPlatform](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")*, *\*\*kwargs*) → None
    :   Constructor variant that takes a path to compiled ELF files and an execution platform
        specification. Takes same kwargs as above.

    `__init__`(*artifacts: [SdkCompileArtifacts](#cerebras.sdk.runtime.sdkruntimepybind.SdkCompileArtifacts "cerebras.sdk.runtime.sdkruntimepybind.SdkCompileArtifacts")*, *platform: [SdkExecutionPlatform](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")*, *\*\*kwargs*) → None
    :   Constructor variant that takes a compile artifacts specification and execution platform
        specification. Takes same kwargs as above.

    `coord_logical_to_physical`(*logical\_coords: int, int*)
    :   Convert a logical coordinate to a physical coordinate.
        For a program with fabric offsets (`offset_x`, `offset_y`),
        and program rectangle coordinate (`x`, `y`), this function
        returns (`offset_x + x`, `offset_y + y`).

        Parameters
        :   **logical\_coords** – Tuple containing logical coordinates.

        Returns
        :   Tuple containing physical coordinates.

        Return type
        :   `(int, int)`

    `dump_core`(*corefile: str*)
    :   Dump the core of a simulator run, to be used for debugging with `csdb`. Note that the
        specified name of the corefile MUST be “corefile.cs1” to use with `csdb`, and this method
        can only be called after a blocking [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") API call, or after calling
        [`SdkRuntime.stop()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.stop "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.stop").

        Parameters
        :   **corefile** – Name of corefile. Must be “corefile.cs1” to use with `csdb`.

    `get_id`(*symbol: str*) → int
    :   Retrieve the integer representation of an exported symbol which is
        exported in the kernel. Possible symbols include a data tensor or a
        host-callable function.

        Parameters
        :   **symbol** (`str`) – The exported name of the symbol.

        Returns
        :   Integer representation of exported symbol

        Return type
        :   `int`

    `get_port_id`(*port\_name: str*) → PortId
    :   Part of the [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API.

        Retrieve the integer representation of a program port for streaming
        data via [`SdkRuntime.send()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send") or `SdkRuntime.receive()`.

        Parameters
        :   **symbol** (`str`) – The exported name of the symbol.

        Returns
        :   Integer representation of program data port.

        Return type
        :   `PortId`

    `is_task_done`(*task\_handle: [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")*) → bool
    :   Query if task `task_handle` is complete.

        Parameters
        :   **task\_handle** ([`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")) – Handle to a task previously launched by [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime").

        Returns
        :   `True` if task is done, and `False` otherwise.

        Return type
        :   `bool`

    `launch`(*symbol: str*, *\*args*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Trigger a host-callable function defined in the kernel, with type checking for arguments.

        Parameters
        :   **symbol** (`str`) – The exported name of the symbol corresponding to a host-callable function.

        Positional Arguments
        :   Matches the arguments of the host-callable function. [`SdkRuntime.launch()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.launch "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.launch")
            will perform type checking on the arguments.

        Keyword Arguments
        :   **nonblock** (`bool`) –
            Nonblocking if `True`, blocking otherwise.

        Returns
        :   Handle to the task launched by [`SdkRuntime.launch()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.launch "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.launch").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

        **Example**:

        > Consider a kernel which defines a host-callable function `fn_foo`
        > by:
        >
        > ```csl
        > comptime {
        >   @export_symbol(fn_foo);
        > }
        >
        > ```
        >
        > The host calls `fn_foo` by
        > `runner.launch("fn_foo", nonblock=False)`.

    `load`()
    :   Load the binaries to simfabric or WSE. It may takes 80+ seconds to load
        the binaries onto the WSE.

    `memcpy_d2h`(*dest: numpy.ndarray*, *src: int*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Receive a host tensor from the device.
        The data is received from the region of interest (ROI) which is
        a bounding box starting at coordinate `(px, py)` with width `w` and
        height `h`.

        Parameters
        :   - **dest** (`numpy.ndarray`) – A 3-D host tensor `A[h][w][elem_per_pe]`, wrapped in a 1-D array
              according to keyword argument `order`.
            - **src** (`int`) – A user-defined color if keyword argument `streaming=True`,
              symbol of a device tensor otherwise.
            - **px** (`int`) – `x`-coordinate of start point of the ROI.
            - **py** (`int`) – `y`-coordinate of start point of the ROI.
            - **w** (`int`) – Width of the ROI.
            - **h** (`int`) – Height of the ROI.
            - **elem\_per\_pe** (`int`) – Number of elements per PE.
              The data type of an element is 16-bit and 32-bit only.
              If the tensor has `k` elements per PE, `elem_per_pe` is `k`
              even if the data type is 16-bit.
              If the data type is 16-bit, the user has to extend the tensor to
              a 32-bit one, with zero filled in the higher 16 bits.

        Keyword Arguments
        :   - **streaming** (`bool`) –
              Streaming mode if `True`, copy mode otherwise.
            - **data\_type** ([`MemcpyDataType`](#cerebras.sdk.runtime.sdkruntimepybind.MemcpyDataType "cerebras.sdk.runtime.sdkruntimepybind.MemcpyDataType")) –
              32-bit if `MemcpyDataType.MEMCPY_32BIT` or 16-bit if
              `MemcpyDataType.MEMCPY_16BIT`.
              Note that this argument has no effect if `streaming` is `True`,
              and the user must handle the data appropriately in the receiving
              wavelet-triggered task.
              Additionally, the underlying type of the tensor `dest` must be
              32-bit. The tensor must be extended to a 32-bit one with zero
              filled in the higher 16 bits.
            - **order** ([`MemcpyOrder`](#cerebras.sdk.runtime.sdkruntimepybind.MemcpyOrder "cerebras.sdk.runtime.sdkruntimepybind.MemcpyOrder")) –
              Row-major if `MemcpyOrder.ROW_MAJOR` or column-major if
              `MemcpyOrder.COL_MAJOR`.
            - **nonblock** (`bool`) –
              Nonblocking if `True`, blocking otherwise.

        Returns
        :   Handle to the task launched by [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

    `memcpy_h2d`(*dest: int*, *src: numpy.ndarray*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Send a host tensor to the device.
        The data is distributed into the region of interest (ROI) which is a
        bounding box starting at coordinate `(px, py)` with width `w` and
        height `h`.

        Parameters
        :   - **dest** (`int`) – A user-defined color if keyword argument `streaming=True`,
              symbol of a device tensor otherwise.
            - **src** (`numpy.ndarray`) – A 3-D host tensor `A[h][w][elem_per_pe]`, wrapped in a 1-D array
              according to parameter `order`.
            - **px** (`int`) – `x`-coordinate of start point of the ROI.
            - **py** (`int`) – `y`-coordinate of start point of the ROI.
            - **w** (`int`) – Width of the ROI.
            - **h** (`int`) – Height of the ROI.
            - **elem\_per\_pe** (`int`) – Number of elements per PE.
              The data type of an element is 16-bit and 32-bit only.
              If the tensor has `k` elements per PE, `elem_per_pe` is `k`
              even if the data type is 16-bit.
              If the data type is 16-bit, the user has to extend the tensor to
              a 32-bit one, with zero filled in the higher 16 bits.

        Keyword Arguments
        :   See [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h") keyword arguments.

        Returns
        :   Handle to the task launched by [`SdkRuntime.memcpy_h2d()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

    `memcpy_h2d_colbcast`(*dest: int*, *src: numpy.ndarray*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Broadcast a row of host data down columns of PEs.
        The data is distributed across the first row in the region of interest (ROI), which is a
        bounding box starting at coordinate `(px, py)` with width `w` and
        height `h`, and then broadcast down each column of the ROI.

        Parameters
        :   - **dest** (`int`) – A user-defined color if keyword argument `streaming=True`,
              symbol of a device tensor otherwise.
            - **src** (`numpy.ndarray`) – A 2-D host tensor `A[w][elem_per_pe]`, wrapped in a 1-D array
              according to parameter `order`.
            - **px** (`int`) – `x`-coordinate of start point of the ROI.
            - **py** (`int`) – `y`-coordinate of start point of the ROI.
            - **w** (`int`) – Width of the ROI.
            - **h** (`int`) – Height of the ROI.
            - **elem\_per\_pe** (`int`) – Number of elements per PE.
              The data type of an element is 16-bit and 32-bit only.
              If the tensor has `k` elements per PE, `elem_per_pe` is `k`
              even if the data type is 16-bit.
              If the data type is 16-bit, the user has to extend the tensor to
              a 32-bit one, with zero filled in the higher 16 bits.

        Keyword Arguments
        :   See [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h") keyword arguments.

        Returns
        :   Handle to the task launched by [`SdkRuntime.memcpy_h2d_colbcast()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d_colbcast "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d_colbcast").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

    `memcpy_h2d_rowbcast`(*dest: int*, *src: numpy.ndarray*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Broadcast a column of host data across rows of PEs.
        The data is distributed across the first column in the region of interest (ROI), which is a
        bounding box starting at coordinate `(px, py)` with width `w` and
        height `h`, and then broadcast down each column of the ROI.

        Parameters
        :   - **dest** (`int`) – A user-defined color if keyword argument `streaming=True`,
              symbol of a device tensor otherwise.
            - **src** (`numpy.ndarray`) – A 2-D host tensor `A[h][elem_per_pe]`, wrapped in a 1-D array
              according to parameter `order`.
            - **px** (`int`) – `x`-coordinate of start point of the ROI.
            - **py** (`int`) – `y`-coordinate of start point of the ROI.
            - **w** (`int`) – Width of the ROI.
            - **h** (`int`) – Height of the ROI.
            - **elem\_per\_pe** (`int`) – Number of elements per PE.
              The data type of an element is 16-bit and 32-bit only.
              If the tensor has `k` elements per PE, `elem_per_pe` is `k`
              even if the data type is 16-bit.
              If the data type is 16-bit, the user has to extend the tensor to
              a 32-bit one, with zero filled in the higher 16 bits.

        Keyword Arguments
        :   See [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h") keyword arguments.

        Returns
        :   Handle to the task launched by [`SdkRuntime.memcpy_h2d_rowbcast()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d_rowbcast "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d_rowbcast").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

    `memcpy_h2d_stride`(*dest: int*, *src: numpy.ndarray*, *px: int*, *py: int*, *w: int*, *h: int*, *elem\_per\_pe: int*, *row\_stride: int*, *col\_stride: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Send a host tensor to the device with a stride pattern across receiving PEs.
        The data is distributed into the region of interest (ROI) which is a
        bounding box starting at coordinate `(px, py)` with width `w` and height `h`.
        Across a given row, `row_stride` determines the stride between receiving PEs
        within the ROI, and across a given column, `col_stride` determines the
        stride between receiving PEs.
        The first row and column to which data is sent is given by the PE `(px, py)`
        at the top-left of the ROI.

        We denote by `xi` and `eta` the number of columns and rows to which elements
        will be sent in the ROI, respectively.
        Since the ROI is `w` PEs wide and `h` PEs tall, `xi` and `eta` are given by
        `xi = 1 + floor((w - 1) / row_stride)` and `eta = 1 + floor((h - 1) / col_stride)`.

        As an example, consider an ROI starting at (0, 0) with width 6 and height 8,
        and row and column strides 3 and 2, respectively. Then PEs with x coordinate
        0 or 3 and y coordinate 0, 2, 4, 6 will receive data from the host.
        In this case, `xi = 2` and `eta = 4`.

        Parameters
        :   - **dest** (`int`) – A user-defined color if keyword argument `streaming=True`,
              symbol of a device tensor otherwise.
            - **src** (`numpy.ndarray`) – A 3-D host tensor `A[xi][eta][elem_per_pe]`, wrapped in a 1-D array
              according to parameter `order`.
            - **px** (`int`) – `x`-coordinate of start point of the ROI.
            - **py** (`int`) – `y`-coordinate of start point of the ROI.
            - **w** (`int`) – Width of the ROI.
            - **h** (`int`) – Height of the ROI.
            - **elem\_per\_pe** (`int`) – Number of elements per PE.
              The data type of an element is 16-bit and 32-bit only.
              If the tensor has `k` elements per PE, `elem_per_pe` is `k`
              even if the data type is 16-bit.
              If the data type is 16-bit, the user has to extend the tensor to
              a 32-bit one, with zero filled in the higher 16 bits.
            - **row\_stride** (`int`) – Stride between PEs within a row in the ROI. Since the ROI
              is `w` PEs wide, the number of columns to which elements will be sent is
              `xi = 1 + floor((w - 1) / row_stride)`.
            - **col\_stride** – Stride between PEs within a column in the ROI. Since the ROI
              is `h` PEs tall, the number of rows to which elements will be sent is
              `eta = 1 + floor((h - 1) / col_stride)`.

        Keyword Arguments
        :   See [`SdkRuntime.memcpy_d2h()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_d2h") keyword arguments.

        Returns
        :   Handle to the task launched by [`SdkRuntime.memcpy_h2d_stride()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d_stride "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.memcpy_h2d_stride").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

    `receive`(*port: Union[PortId, str]*, *dest: numpy.ndarray*, *n\_wavelets: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Part of the py:class`SdkRuntime` direct link API.

        Receive `n_wavelets` wavelets via the program port `port` into array `dest`.

        Parameters
        :   - **port** (`Union[PortId, str]`) – Program port from which data will be received. Can be specified by a numerical
              port ID or by name.
            - **dest** (`numpy.ndarray`.) – Destination array into which the data will be received.
            - **n\_wavelets** (`int`) – Number of wavelets to receive.

        Keyword Arguments
        :   - **nonblock** (`bool`) –
              Nonblocking if `True`, blocking otherwise.

        Returns
        :   Handle to the task launched by [`SdkRuntime.receive()`](#cerebras.sdk.runtime.sdkruntime.pybind.SdkRuntime.receive "cerebras.sdk.runtime.sdkruntime.pybind.SdkRuntime.receive").

        Return type
        :   `Task`

    `receive_tofile`(*port: Union[PortId, str]*, *outfile: str*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Part of the `SdkRuntime` direct link API.

        Receive data via the program port `port` and write to a file named `outfile`.

        Parameters
        :   - **port** (`Union[PortId, str]`) – Program port from which data will be received. Can be specified by a numerical
              port ID or by name.
            - **outfile** (`str`.) – Name of file to which received output is written.

        Keyword Arguments
        :   - **nonblock** (`bool`) –
              Nonblocking if `True`, blocking otherwise.

        Returns
        :   Handle to the task launched by [`SdkRuntime.receive_tofile()`](#cerebras.sdk.runtime.sdkruntime.pybind.SdkRuntime.receive_tofile "cerebras.sdk.runtime.sdkruntime.pybind.SdkRuntime.receive_tofile").

        Return type
        :   `Task`

    `report_port_infos`()
    :   Part of the [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API.

        Reports the port name, color and absolute coordinate of every program data port.

    `run`()
    :   Start the simfabric or WSE run and wait for commands from the host
        runtime.

    `send`(*port: Union[PortId, str]*, *src: numpy.ndarray*, *n\_wavelets: int*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Part of the [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API.

        Stream `n_wavelets` wavelets from `src` to the device via the port `port`.

        Parameters
        :   - **port** (`Union[PortId, str]`) – Target program port in which to stream data. Can be specified by a numerical
              port ID or by name.
            - **src** (`numpy.ndarray`.) – Input source array whose contents will be streamed to the device.
            - **n\_wavelets** (`int`) – Number of wavelets to send.

        Keyword Arguments
        :   - **nonblock** (`bool`) –
              Nonblocking if `True`, blocking otherwise.

        Returns
        :   Handle to the task launched by [`SdkRuntime.send()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send").

        Return type
        :   [`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")

    `send`(*port: Union[PortId, str]*, *src: numpy.ndarray*, *\*\*kwargs*) → [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")
    :   Part of the [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API.

        Same as above when `src.dtype` is exactly `np.int32`, `np.uint32` or
        `np.float32`. In that case, the runtime infers `n_wavelets` from `len(src)`.

    `stop`()
    :   Wait for all pending commands (data transfers and kernel function calls)
        to complete and then stop simfabric or WSE. After this call is complete,
        no new commands will be accepted for this [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") object.

        [`SdkRuntime.stop()`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.stop "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.stop") must be called to end a program. Otherwise, the runtime will
        emit an error.

    `task_wait`(*task\_handle: [Task](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")*)
    :   Wait for the task `task_handle` to complete.

        Parameters
        :   **task\_handle** ([`Task`](#cerebras.sdk.runtime.sdkruntimepybind.Task "cerebras.sdk.runtime.sdkruntimepybind.Task")) – Handle to a task previously launched by [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime").

### SdkTarget

*class* `cerebras.sdk.runtime.sdkruntimepybind.``SdkTarget`
:   Bases: `Enum`

    Specifies a target compilation architecture.

    Values
    :   - **WSE2**
        - **WSE3**

### SimfabConfig

*class* `cerebras.sdk.runtime.sdkruntimepybind.``SimfabConfig`(*num\_threads: int = 16*, *suppress\_trace: bool = False*, *dump\_core: bool = False*, *core\_path: Optional[Union[pathlib.Path, str]] = None*)
:   Bases: `object`

    Specifies simfab configuration for simulator runs.

    Parameters
    :   - **num\_threads** (`int`) – Number of CPU threads used by the simulator. Maximum is 64.
        - **suppress\_trace** (`bool`) – If `True`, suppresses generation of `simfab_traces` when running.
        - **dump\_core** (`bool`) – If `True`, produces a coredump after execution ends.
        - **core\_path** (`Union[pathlib.Path, str, None]`) – Name of produced coredump. `None` (default) is `out.core`.

### Task

*class* `cerebras.sdk.runtime.sdkruntimepybind.``Task`
:   Handle to a task launched by [`SdkRuntime`](#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime").

### get\_platform

`cerebras.sdk.runtime.sdkruntimepybind.``get_platform`(*cmaddr: Optional[str] = None*, *config: SimfabConfig = SimfabConfig()*, *target: SdkTarget = SdkTarget::WSE3*) → [SdkExecutionPlatform](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")
:   Constructs an [`SdkExecutionPlatform`](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform") object configured by simulator or system settings
    and target architecture.

    Parameters
    :   - **cmaddr** (`Union[str, None]`) – CM address in `"IP_ADDRESS:PORT"` format. `None` (default) or the empty
          string chooses the simulator.
        - **config** ([`SimfabConfig`](#cerebras.sdk.runtime.sdkruntimepybind.SimfabConfig "cerebras.sdk.runtime.sdkruntimepybind.SimfabConfig")) – Simulator configuration object. Ignored when `cmaddr` is provided.
        - **target** ([`SdkTarget`](#cerebras.sdk.runtime.sdkruntimepybind.SdkTarget "cerebras.sdk.runtime.sdkruntimepybind.SdkTarget")) – Target architecture for the simulator or system.

    Returns
    :   A configured execution platform object.

    Return type
    :   [`SdkExecutionPlatform`](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")

### get\_simulator

`cerebras.sdk.runtime.sdkruntimepybind.``get_simulator`(*config: SimfabConfig = SimfabConfig()*, *target: SdkTarget = SdkTarget::WSE3*) → [SdkExecutionPlatform](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")
:   Constructs an [`SdkExecutionPlatform`](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform") object for simulator.

    Parameters
    :   - **config** ([`SimfabConfig`](#cerebras.sdk.runtime.sdkruntimepybind.SimfabConfig "cerebras.sdk.runtime.sdkruntimepybind.SimfabConfig")) – Simulator configuration object.
        - **target** ([`SdkTarget`](#cerebras.sdk.runtime.sdkruntimepybind.SdkTarget "cerebras.sdk.runtime.sdkruntimepybind.SdkTarget")) – Target architecture for the simulator.

    Returns
    :   A configured execution platform object.

    Return type
    :   [`SdkExecutionPlatform`](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")

### get\_system

`cerebras.sdk.runtime.sdkruntimepybind.``get_system`(*cmaddr: str*) → [SdkExecutionPlatform](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")
:   Constructs an [`SdkExecutionPlatform`](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform") object for a real system.

    Parameters
    :   - **cmaddr** (`str`) – CM address in `"IP_ADDRESS:PORT"` format.
        - **config** ([`SimfabConfig`](#cerebras.sdk.runtime.sdkruntimepybind.SimfabConfig "cerebras.sdk.runtime.sdkruntimepybind.SimfabConfig")) – Simulator configuration object.
        - **target** ([`SdkTarget`](#cerebras.sdk.runtime.sdkruntimepybind.SdkTarget "cerebras.sdk.runtime.sdkruntimepybind.SdkTarget")) – Target architecture for the simulator.

    Returns
    :   A configured execution platform object.

    Return type
    :   [`SdkExecutionPlatform`](#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")

## sdk\_utils module

Utility functions for common operations with `SdkRuntime`.
Import from `cerebras.sdk.sdk_utils`.

### calculate\_cycles

`cerebras.sdk.sdk_utils.``calculate_cycles`(*timestamp\_buf: numpy.ndarray*) → numpy.int64:
:   Converts values in `timestamp_buf` returned from device into a human-readable
    elapsed cycle count.

    Parameters
    :   **timestamp\_buf** (`numpy.ndarray`) – array returned from device containing elapsed timestamp data

    Returns
    :   Elapsed cycle count.

    Return type
    :   `numpy.int64`

    **Example**:

    > Consider the following CSL snippet which records timestamps and produces a single
    > array to copy back to the host, to generate an elapsed cycle count:
    >
    > ```csl
    > // import time module and create timestamp buffers
    > const timestamp = @import_module("<time>");
    > var tsc_end_buf = @zeros([timestamp.tsc_size_words]u16);
    > var tsc_start_buf = @zeros([timestamp.tsc_size_words]u16);
    >
    > // create elapsed timer buffer and advertise to host
    > var timer_buf = @zeros([3]f32);
    > var ptr_timer_buf: [*]f32 = &timer_buf;
    >
    > timestamp.enable_tsc();
    > // record starting timestamp
    > timestamp.get_timestamp(&tsc_start_buf);
    >
    > // perform some operation for which you want to calculate elapsed cycles
    >
    > // record ending timestamp
    > timestamp.get_timestamp(&tsc_end_buf);
    > timestamp.disable_tsc();
    >
    > var lo_: u16 = 0;
    > var hi_: u16 = 0;
    > var word: u32 = 0;
    >
    > lo_ = tsc_start_buf[0];
    > hi_ = tsc_start_buf[1];
    > timer_buf[0] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );
    >
    > lo_ = tsc_start_buf[2];
    > hi_ = tsc_end_buf[0];
    > timer_buf[1] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );
    >
    > lo_ = tsc_end_buf[1];
    > hi_ = tsc_end_buf[2];
    > timer_buf[2] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );
    >
    > ```
    >
    > Then the elapsed cycles can be calculated on the host with:
    >
    > ```python
    > # Get symbol for timer_buf on device
    > symbol_timer_buf = runner.get_id("timer_buf")
    >
    > # Copy back timer_buf from all width x height PEs
    > data = np.zeros((width*height*3, 1), dtype=np.uint32)
    > runner.memcpy_d2h(data, symbol_timer_buf, 0, 0, width, height, 3, streaming=False,
    >   data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    > elapsed_time_hwl = data.view(np.float32).reshape((height, width, 3))
    >
    > # Print elapsed cycles for each PE
    > for pe_x in range(width):
    >   for pe_y in range(height):
    >     cycle_cnt = sdk_utils.calculate_cycles(elapsed_time_hwl[pe_y, pe_x, :])
    >     print("Elapsed cycles on PE ", pe_x, ", ", pe_y, ": ", cycle_cnt)
    >
    > ```

### input\_array\_to\_u32

`cerebras.sdk.sdk_utils.``input_array_to_u32`(*arr: numpy.ndarray*, *sentinel: Optional[int]*, *fast\_dim\_sz: int*) → numpy.ndarray
:   Converts a 16-bit tensor to a 32-bit tensor of type `u32` for use with `memcpy`.
    The parameter `sentinel` distiguishes two different extensions of 16-bit data.
    If `sentinel` is `None`, zero-pad the upper 16 bits.
    If `sentinel` is not `None`, pack the index of the innermost dimension of the array
    into the upper 16-bits.

    Parameters
    :   - **arr** (`numpy.ndarray`) – A numpy array with 2 or 4 bytes per element.
        - **sentinel** (`Optional[int]`) – For 16-bit input data, if this parameter is not `None`,
          pack the index of the innermost dimension into
          the high bits of the 32-bit wavelet.
          If sentinel is None, then the high bits are zeros.
        - **fast\_dim\_sz** (`int`) – If `sentinel` is not `None`, specifies size of fastest-changing
          dimension for generating the index.

    Returns
    :   Numpy view into `arr` with specified numpy data type.

    Return type
    :   `numpy.ndarray.view`

### memcpy\_view

`cerebras.sdk.sdk_utils.``memcpy_view`(*arr: numpy.ndarray*, *datatype: numpy.dtype*) → numpy.ndarray.view
:   Returns a 32, 16 or 8 bit view of a 32 bit numpy array
    (only the lower 16 or 8 bits of each 32 bit word in the
    last two cases).

    Parameters
    :   - **arr** (`numpy.ndarray`) – A numpy array with 4 bytes per element on which the numpy view will be created.
        - **datatype** (`numpy.dtype`) – The numpy data type which should be used in the output view.
          The itemsize must be 1, 2, or 4 bytes.

    Returns
    :   Numpy view into `arr` with specified numpy data type.

    Return type
    :   `numpy.ndarray.view`

    **Example**:

    > [`memcpy_view()`](#cerebras.sdk.sdk_utils.memcpy_view "cerebras.sdk.sdk_utils.memcpy_view") simplifies the use of various precision data types when copying
    > between host and device. Consider the following Python host code which creates
    > a `float16` view into a numpy array. Note that this array *must* be 32-bit.
    > The user can fill the array with `float16` data,
    > and copy it to an array on the device with CSL data type `f16`.
    >
    > ```python
    > x_symbol = runner.get_symbol('x')
    > # This container array must be 32-bit
    > x_container = np.zeros(N, dtype=np.uint32)
    >
    > x = sdk_utils.memcpy_view(x_container, np.float16)
    > x.fill(0.5)
    >
    > runner.memcpy_h2d(x_symbol, x_container, 0, 0, 1, 1, N,
    >             streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
    >             order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    >
    > ```

## debug\_util module

Utilities for parsing debug output and core files of a simulator run.
Import from `cerebras.sdk.debug.debug_util`.

### debug\_util

*class* `cerebras.sdk.debug.debug_util.``debug_util`(*bindir: Union[pathlib.Path, str]*)
:   Bases: `object`

    Loads ELF files in `bindir` in order to dump symbols for debugging.

    The user does not need to export the symbols in the kernel. [`debug_util`](#cerebras.sdk.debug.debug_util.debug_util "cerebras.sdk.debug.debug_util.debug_util")
    dumps the core and looks for the symbols in the ELFs. If the symbol at
    `Px.y` is not found in the corresponding ELF, [`debug_util`](#cerebras.sdk.debug.debug_util.debug_util "cerebras.sdk.debug.debug_util.debug_util")
    emits an error.

    The most common errors are either: 1) a wrong coordinate passed in
    [`debug_util.get_symbol()`](#cerebras.sdk.debug.debug_util.debug_util.get_symbol "cerebras.sdk.debug.debug_util.debug_util.get_symbol"), or 2) a correct coordinate, but the symbol has been
    removed due to compiler optimization. One can use `readelf` to check if
    the symbol exists or not. If not, the user can export the symbol in the
    kernel to keep the symbol in the ELF.

    The functionality of this class is only supported in the simulator.

    **Example**:

    > ```python
    > from cerebras.sdk.debug.debug_util import debug_util
    >
    > # run the app
    > # dirname is the path to ELFs
    > simulator = SdkRuntime(dirname)
    > simulator.load()
    > simulator.run()
    > ...
    > simulator.stop()
    >
    > # retrieve symbols after the run
    > debug_mod = debug_util(dirname)
    > # assume the core rectangle starts at P4.1, the dimension is
    > # width-by-height and we want to retrieve the symbol y for every PE
    > core_offset_x = 4
    > core_offset_y = 1
    > for py in range(height):
    >   for px in range(width):
    >     t = debug_mod.get_symbol(core_offset_x+px, core_offset_y+py, 'y', np.float32)
    >     print(f"At (py, px) = {py, px}, symbol y = {t}")
    >
    > ```

    `get_symbol`(*col: int*, *row: int*, *symbol: str*, *dtype: numpy.dtype*) → numpy.ndarray
    :   Read the value of `symbol` of given type at given PE coordinates.
        Note that each call to this function scans the whole fabric, so prefer
        [`debug_util.get_symbol_rect()`](#cerebras.sdk.debug.debug_util.debug_util.get_symbol_rect "cerebras.sdk.debug.debug_util.debug_util.get_symbol_rect") over calling this in a loop.

        Parameters
        :   - **px** (`int`) – `x`-coordinate of the PE, indexed from the northwest corner
              of the entire fabric (NOT the program rectangle)
            - **py** (`int`) – `y`-coordinate of the PE, indexed from the northwest corner
              of the entire fabric (NOT the program rectangle)
            - **symbol** (`str`) – Name of the symbol to be read.
            - **dtype** (`numpy.dtype`) – Numpy data type of values contained by symbol.

        Returns
        :   Numpy array of output values read at symbol.

        Return type
        :   `numpy.ndarray`

    `get_symbol_rect`(*rectangle: Tuple[Tuple[int, int], Tuple[int, int]]*, *symbol: str*, *dtype: numpy.dtype*) → numpy.ndarray
    :   Read the value of `symbol` of given type for a rectangle of PEs.

        Parameters
        :   - **rectangle** (`Tuple[Tuple[int, int], Tuple[int, int]]`) – Rectangle specified as `((col, row), (width, height))`,
              indexed from the northwest corner of the entire fabric
              (NOT the program rectangle)
            - **symbol** (`str`) – Name of the symbol to be read.
            - **dtype** (`numpy.dtype`) – Numpy data type of values contained by symbol.

        Returns
        :   Numpy array of output values read at symbol. The first two dimensions of the
            returned array are PE coordinates `(column, row)` relative to the rectangle.

        Return type
        :   `numpy.ndarray`

    `read_trace`(*px: int*, *py: int*, *name: str*) → list
    :   Parse a CSL trace buffer with name `name` at the given PE coordinates.

        Parameters
        :   - **px** (`int`) – `x`-coordinate of the PE, indexed from the northwest corner
              of the entire fabric (NOT the program rectangle)
            - **py** (`int`) – `y`-coordinate of the PE, indexed from the northwest corner
              of the entire fabric (NOT the program rectangle)
            - **name** (`str`) – Name of the trace buffer to be read.

        Returns
        :   Heterogenous list of trace values.

        Return type
        :   `list`

        **Example**:

        > Consider a device kernel which initializes a trace buffer with the CSL
        > `debug` library and uses it to record values:
        >
        > ```csl
        > const debug_mod = @import_module("<debug>", .{.key = "my_trace", .buffer_size = 100});
        >
        > fn foo() void {
        >   debug_mod.trace_timestamp();
        >   debug_mod.trace_string("Bar");
        >   debug_mod.trace_i16(1);
        > }
        >
        > ```
        >
        > Then the trace can be read in the host code with:
        >
        > ```python
        > trace_output = debug_mod.read_trace(4, 1, 'my_trace')
        > print(trace_output)
        >
        > ```
        >
        > If `foo` was executed only once, then `trace_output` will be
        > a heterogenous list containing a timestamp, the string “Bar”,
        > and the number 1.

[previous

SDK GUI](debug/sdk-gui.md "previous page")
[next

SdkLayout API Reference](api-docs/sdklayout-api.md "next page")
