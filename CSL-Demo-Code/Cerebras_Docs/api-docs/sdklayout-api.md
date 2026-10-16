---
title: "SdkLayout API Reference"
url: "https://sdk.cerebras.net/api-docs/sdklayout-api"
crawled_at: "2026-02-14 23:55:40"
---

# SdkLayout API Reference

> Source: [https://sdk.cerebras.net/api-docs/sdklayout-api](https://sdk.cerebras.net/api-docs/sdklayout-api)

This section presents the `SdkLayout` program layout specification Python API.

Note that this API is part of the `sdkruntimepybind` module documented in
[SdkRuntime API Reference](api-docs/sdkruntime-api.md#sdkruntime-api-reference).

## CodeRegion

*class* `cerebras.sdk.runtime.sdkruntimepybind.``CodeRegion`
:   Bases: `object`

    Specifies a code region.

    `color`(*name: str*) → [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")
    :   Create and return a new color that is scoped within a region.

        Parameters
        :   **name** (`str`) – Name to be assigned to the color.

        Returns
        :   A new color scoped within this region.

        Return type
        :   [`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")

    `color`(*name: str*, *value: int*) → [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")
    :   Create and return a new color (with a value) that is scoped within a region.

        Parameters
        :   - **name** (`str`) – Name to be assigned to the color.
            - **value** (`int`) – Value to be assigned to the color.

        Returns
        :   A new color scoped within this region.

        Return type
        :   [`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")

    `create_input_port`(*color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *edge: [Edge](#cerebras.sdk.runtime.sdkruntimepybind.Edge "cerebras.sdk.runtime.sdkruntimepybind.Edge")*, *routes: List[[RoutingPosition](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]*, *data\_size: int*, *prefix: str = ''*) → [PortHandle](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")
    :   Given a color, an orientation (i.e., `edge`), a set of output routes, a size, and an
        optional prefix, create and return a new input communication port. The optional prefix
        can be used to create unique ports with the same color. Ports must be unique, otherwise an
        exception is thrown.

        Parameters
        :   - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color of the input port.
            - **edge** ([`Edge`](#cerebras.sdk.runtime.sdkruntimepybind.Edge "cerebras.sdk.runtime.sdkruntimepybind.Edge")) – Edge on which to create the input port.
            - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of output routes for the input port.
            - **data\_size** (`int`) – The size of the port’s data, used to verify port compatibility.
            - **prefix** (`str`) – Optional prefix to port’s name, which allows creation of unique ports with the
              same color.

        Returns
        :   Handle to the created input port.

        Return type
        :   [`PortHandle`](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")

    `create_output_port`(*color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *edge: [Edge](#cerebras.sdk.runtime.sdkruntimepybind.Edge "cerebras.sdk.runtime.sdkruntimepybind.Edge")*, *routes: List[[RoutingPosition](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]*, *data\_size: int*, *prefix: str = ''*)
    :   Given a color, an orientation (i.e., `edge`), a set of input routes, a size, and an
        optional prefix, create and return a new output communication port. The optional prefix
        can be used to create unique ports with the same color. Ports must be unique, otherwise an
        exception is thrown.

        Parameters
        :   - **edge** ([`Edge`](#cerebras.sdk.runtime.sdkruntimepybind.Edge "cerebras.sdk.runtime.sdkruntimepybind.Edge")) – Edge on which to create the output port.
            - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of input routes for the output port.
            - **data\_size** (`int`) – The size of the port’s data, used to verify port compatibility.
            - **prefix** (`str`) – Optional prefix to port’s name, which allows creation of unique ports with the
              same color.

        Returns
        :   Handle to the created output port.

        Return type
        :   [`PortHandle`](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")

    `paint`(*coord: IntVector*, *color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *routes: List[[RoutingPosition](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]*)
    :   Set the routing for a given color on a single PE within this region.

        Parameters
        :   - **coord** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – Coordinate on which color will be painted.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color to be painted on this coordinate.
            - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of routing positions which will be applied to this color.

    `paint_all`(*color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *routes: List[[RoutingPosition](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]*)
    :   Set the routing for a given color on all PEs of the region.

        Parameters
        :   - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color to be painted on this region.
            - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of routing positions which will be applied to this color.

    `paint_all`(*color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *routes: List[[RoutingPosition](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]*, *edge\_routes: List[[EdgeRouteInfo](#cerebras.sdk.runtime.sdkruntimepybind.EdgeRouteInfo "cerebras.sdk.runtime.sdkruntimepybind.EdgeRouteInfo")]*)
    :   Set the routing for a given color on all PEs of the region with special routing on one or
        all region edges.

        Parameters
        :   - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color to be painted on this region.
            - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of routing positions which will be applied to this color.
            - **edge\_routes** (List[[`EdgeRouteInfo`](#cerebras.sdk.runtime.sdkruntimepybind.EdgeRouteInfo "cerebras.sdk.runtime.sdkruntimepybind.EdgeRouteInfo")]) – List of routing positions to be applied to this color on the region’s
              edges.

    `paint_range`(*rect: IntRectangle*, *color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *routes: List[[RoutingPosition](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]*)
    :   Set the routing for a given color on a contiguous rectangular subset of PEs within this
        region.

        Parameters
        :   - **rect** ([`IntRectangle`](#cerebras.geometry.geometry.IntRectangle "cerebras.geometry.geometry.IntRectangle")) – Rectangular subset of PEs within this region on which color will be painted.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color to be painted on this rectangle.
            - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of routing positions which will be applied to this color.

    `place`(*x: int*, *y: int*)
    :   Place code region at specific coordinates `(x, y)`.

        Parameters
        :   - **x** (`int`) – x-coordinate at which core region will be placed.
            - **y** (`int`) – y-coordinate at which core region will be placed.

    `set_param`(*coord: IntVector*, *name: str*, *value: int*)
    :   Set an unsigned integer parameter on a single PE within this region.

        Parameters
        :   - **coord** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – Coordinate on which parameter will be set.
            - **name** (`str`) – Name of the parameter.
            - **value** (`int`) – Unsigned integer value of the parameter.

    `set_param`(*coord: IntVector*, *color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*)
    :   Set a color parameter on a single PE within this region.

        Parameters
        :   - **coord** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – Coordinate on which parameter will be set.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color value of the parameter.

    `set_param`(*coord: IntVector*, *name: str*, *value: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*)
    :   Set the value of parameter `name` on a single PE within this region with the value of
        color `value`.

        Parameters
        :   - **coord** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – Coordinate on which parameter will be set.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Name of the parameter.
            - **color** – Color value of the parameter.

    `set_param_all`(*name: str*, *value: int*)
    :   Set an unsigned integer parameter on all PEs of the region.

        Parameters
        :   - **name** (`str`) – Name of the parameter.
            - **value** (`int`) – Unsigned integer value of the parameter.

    `set_param_all`(*color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*)
    :   Set a color parameter on all PEs of the region.

        Parameters
        :   **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color value of the parameter.

    `set_param_all`(*name: str*, *value: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*)
    :   Set the value of parameter `name` on all PEs of the region with the value of color
        `value`.

        Parameters
        :   - **name** (`str`) – Name of the parameter.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color value of the parameter.

    `set_param_range`(*rect: IntRectangle*, *name: str*, *value: int*)
    :   Set an unsigned integer parameter on a contiguous rectangular subset of PEs within this
        region.

        Parameters
        :   - **rect** ([`IntRectangle`](#cerebras.geometry.geometry.IntRectangle "cerebras.geometry.geometry.IntRectangle")) – Rectangular subset of PEs within this region on which parameter will be set.
            - **name** (`str`) – Name of the parameter.
            - **value** (`int`) – Unsigned integer value of the parameter.

    `set_param_range`(*rect: IntRectangle*, *color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*)
    :   Set a color parameter on a contiguous rectangular subset of PEs within this region.

        Parameters
        :   - **rect** ([`IntRectangle`](#cerebras.geometry.geometry.IntRectangle "cerebras.geometry.geometry.IntRectangle")) – Rectangular subset of PEs within this region on which parameter will be set.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color value of the parameter.

    `set_param_range`(*rect: IntRectangle*, *name: str*, *value: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*)
    :   Set the value of parameter `name` on a contiguous rectangular subset of PEs within this
        region with the value of color `value`.

        Parameters
        :   - **rect** ([`IntRectangle`](#cerebras.geometry.geometry.IntRectangle "cerebras.geometry.geometry.IntRectangle")) – Rectangular subset of PEs within this region on which parameter will be set.
            - **name** (`str`) – Name of the parameter.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color value of the parameter.

    `set_symbol_all`(*symbol: str*, *data: np.ndarray*, *width: int*, *height: int*)
    :   Given a symbol name `symbol` and `data` with a given 2D shape specified by `width`
        and `height`, store `data` uniformly across the PEs of this code region. The 2D shape of
        `data` must be a multiple of the code region’s dimensions or an error will be emitted.

        Parameters
        :   - **symbol** (`str`) – Name of the symbol.
            - **data** (`np.ndarray`) – 2D data array to be applied to the symbol. Data must be of type `np.int32`,
              `np.uint32`, `np.float32`, `np.int16`, or `np.uint16`.
            - **width** (`int`) – Width of 2D data array.
            - **height** (`int`) – Height of 2D data array.

## Color

*class* `cerebras.sdk.runtime.sdkruntimepybind.``Color`(*name: str*, *value: Optional[int] = None*)
:   Bases: `object`

    Represents a color with an optional user-specified physical value. Objects of this class can be
    used for routing and can also be used to set microcode parameter values. If a physical value is
    not provided, then a physical value will be allocated automatically by the compiler.

    Parameters
    :   - **name** (`str`) – Name given to the color.
        - **value** (`Optional[int]`) – Physical value given to the color. If not provided, then a value will be allocated
          automatically by the compiler. The maximum value is 23.

    `get_global_name`() → str
    :   Returns the global name of the color. If this color is attached to a region, then the name
        returned takes the form `region_name` + `_` + `name`.

        Returns
        :   Global name of the color.

        Return type
        :   `str`

    `get_local_param_name`() → str
    :   Returns the name of the color.

        Returns
        :   Name of the color.

        Return type
        :   `str`

    `get_value`() → Optional[int]
    :   Returns the color’s physical value if one has been assigned. Otherwise, returns `None`.

        Returns
        :   Physical value of the color.

        Return type
        :   `Optional[int]`

## Edge

*class* `cerebras.sdk.runtime.sdkruntimepybind.``Edge`
:   Bases: `Enum`

    Represents edge positions along the boundary of a code region.

    Values
    :   - **TOP**
        - **BOTTOM**
        - **LEFT**
        - **RIGHT**

## EdgeRouteInfo

*class* `cerebras.sdk.runtime.sdkruntimepybind.``EdgeRouteInfo`
:   Bases: `object`

    For a given code region, represents the routing positions in one of the region’s four edges.

## SdkLayout

*class* `cerebras.sdk.runtime.sdkruntimepybind.``SdkLayout`(*platform: SdkExecutionPlatform*, *\*\*kwargs*)
:   Bases: `object`

    Specifies a program layout. This API allows the user to define retangular code regions, define
    color routing and switching, automatically allocate colors, and automatically route between code
    regions.

    Parameters
    :   **platform** ([`SdkExecutionPlatform`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform "cerebras.sdk.runtime.sdkruntimepybind.SdkExecutionPlatform")) – Execution platform specification.

    Keyword Arguments
    :   - **msg\_level** (`str`) –
          Message logging output level. Available output levels are `DEBUG`,
          `INFO`, `WARNING`, and `ERROR`. Default value is `WARNING`.

    `__init__`(*fabric\_file: Union[pathlib.Path, str]*, *\*\*kwargs*)
    :   Constructor variant that takes a path to a fabric JSON file which is used to define the
        compile target and execution platform. Takes same kwargs as above.

        Parameters
        :   **fabric\_file** (`Union[pathlib.Path, str]`) – Path to a fabric JSON file.

    `__init__`(*target: [SdkTarget](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkTarget "cerebras.sdk.runtime.sdkruntimepybind.SdkTarget")*, *\*\*kwargs*)
    :   Constructor variant that takes a target architecture. Takes same kwargs as above.

        Parameters
        :   **target** ([`SdkTarget`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkTarget "cerebras.sdk.runtime.sdkruntimepybind.SdkTarget")) – Target architecture for compilation.

    `compile`(*out\_prefix: str*) → [SdkCompileArtifacts](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkCompileArtifacts "cerebras.sdk.runtime.sdkruntimepybind.SdkCompileArtifacts")
    :   Compile this layout and produce artifacts with a given path prefix.

        Parameters
        :   **out\_prefix** (`str`) – Path to which artifacts will be produced.

    `connect`(*tx: [PortHandle](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")*, *rx: [PortHandle](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")*)
    :   Automatically connect two ports.

        Parameters
        :   - **tx** ([`PortHandle`](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")) – Transmitting output data port which will send data to `rx`.
            - **rx** ([`PortHandle`](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")) – Receiving input data port which will receive data from `tx`.

    `create_code_region`(*source: str*, *name: str*, *width: int*, *height: int*) → [CodeRegion](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")
    :   Create a code region.

        Parameters
        :   - **source** (`str`) – Path to code source file.
            - **name** (`str`) – Name of the created code region.
            - **width** (`int`) – Width in PEs of the created code region.
            - **height** (`int`) – Height in PEs of the created code region.

        Returns
        :   The created code region object.

        Return type
        :   [`CodeRegion`](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")

    `create_input_stream`(*port: [PortHandle](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")*, *io\_loc: Optional[IntVector] = None*, *io\_buffer\_size: int = 1024*) → str
    :   Sets up an input stream from the host to a 1-PE region at `io_loc` and then to input port
        `port`. If `io_loc` is not provided, an available location will be automatically picked.
        `io_buffer_size` can be provided to specify the buffer size at `io_loc`. Returns the name
        of the stream’s port which can be used by the [`SdkRuntime`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API method
        [`SdkRuntime.send()`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send").

        Parameters
        :   - **port** ([`PortHandle`](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")) – Handle to input port.
            - **io\_loc** (Optional[[`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")]) – PE location on wafer which receives input stream data. If not provided, a
              location is automatically chosen.
            - **io\_buffer\_size** (`int`) – Buffer size allocated at `io_loc`. Default is 1024.

        Returns
        :   Name of created input stream’s port.

        Return type
        :   `str`

    `create_input_stream_from_loc`(*loc: IntVector*, *color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *prefix: str = ''*) → str
    :   Sets up an input stream from the host to `loc` on a given color `color` assuming that
        a code region is already defined at `loc` to consume the incoming data. An optional
        prefix can be provided to uniquely identify the stream in case of naming conflicts. Returns
        the name of the stream’s port which can be used by the [`SdkRuntime`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API
        method [`SdkRuntime.send()`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime.send").

        Parameters
        :   - **loc** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – PE location of existing input port into which data will be streamed.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color on which stream transmits data.
            - **prefix** (`str`) – Optional prefix prepended to created stream’s name.

        Returns
        :   Name of created input stream’s port.

        Return type
        :   `str`

    `create_output_stream`(*port: [PortHandle](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")*, *io\_loc: IntVector = None*, *io\_buffer\_size: int = 1024*) → str
    :   Sets up an output stream from output port `port` to a 1-PE region at `io_loc` and then
        to the host. If `io_loc` is not provided, an available location will be automatically
        picked. `io_buffer_size` can be provided to specify the buffer size at `io_loc`. Returns
        the name of the stream’s port which can be used by the [`SdkRuntime`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API
        methods `SdkRuntime.receive()` and `SdkRuntime.receive_tofile()`.

        Parameters
        :   - **port** ([`PortHandle`](#cerebras.sdk.runtime.sdkruntimepybind.PortHandle "cerebras.sdk.runtime.sdkruntimepybind.PortHandle")) – Handle to output port.
            - **io\_loc** (Optional[[`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")]) – PE location on wafer which send out output stream data. If not provided, a
              location is automatically chosen.
            - **io\_buffer\_size** (`int`) – Buffer size allocated at `io_loc`. Default is 1024.

        Returns
        :   Name of created output stream’s port.

        Return type
        :   `str`

    `create_output_stream_from_loc`(*loc: IntVector*, *color: [Color](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")*, *prefix: str = ''*) → str
    :   Sets up an output stream to the host from `loc` on a given color `color` assuming that
        a code region is already defined at `loc` to produce the outgoing data. An optional
        prefix can be provided to uniquely identify the stream in case of naming conflicts. Returns
        the name of the stream’s port which can be used by the [`SdkRuntime`](api-docs/sdkruntime-api.md#cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime "cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime") direct link API
        methods `SdkRuntime.receive()` and `SdkRuntime.receive_tofile()`.

        Parameters
        :   - **loc** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – PE location of existing output port from which data will be streamed.
            - **color** ([`Color`](#cerebras.sdk.runtime.sdkruntimepybind.Color "cerebras.sdk.runtime.sdkruntimepybind.Color")) – Color on which stream transmits data.
            - **prefix** (`str`) – Optional prefix prepended to created stream’s name.

        Returns
        :   Name of created output stream’s port.

        Return type
        :   `str`

    `hstack`(*children: List[[CodeRegion](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]*) → int
    :   Place child code regions horizontally and relative to the first child in the given list, and
        return the width of the resulting code region.

        Parameters
        :   **children** (List[[`CodeRegion`](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]) – Code regions to be placed.

        Returns
        :   Width of resulting code region.

        Type
        :   `int`

    `hstack`(*children: List[[CodeRegion](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]*, *origin: IntVector*) → int
    :   Place child code regions horizontally and relative to a specified origin, and return the
        width of the resulting code region.

        Parameters
        :   - **children** (List[[`CodeRegion`](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]) – Code regions to be placed.
            - **origin** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – PE coordinate which serves as origin for placed code regions.

        Returns
        :   Width of resulting code region.

        Type
        :   `int`

    `vstack`(*children: List[[CodeRegion](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]*) → int
    :   Place child code regions vertically and relative to the first child in the given list, and
        return the height of the resulting code region.

        Parameters
        :   **children** (List[[`CodeRegion`](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]) – Code regions to be placed.

        Returns
        :   Height of resulting code region.

        Type
        :   `int`

    `vstack`(*children: List[[CodeRegion](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]*, *origin: IntVector*) → int
    :   Place child code regions vertically and relative to a specified origin, and return the
        height of the resulting code region.

        Parameters
        :   - **children** (List[[`CodeRegion`](#cerebras.sdk.runtime.sdkruntimepybind.CodeRegion "cerebras.sdk.runtime.sdkruntimepybind.CodeRegion")]) – Code regions to be placed.
            - **origin** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – PE coordinate which serves as origin for placed code regions.

        Returns
        :   Height of resulting code region.

        Type
        :   `int`

## PortHandle

*class* `cerebras.sdk.runtime.sdkruntimepybind.``PortHandle`
:   Bases: `object`

    Handle to a program input or output data port.

## Route

*class* `cerebras.sdk.runtime.sdkruntimepybind.``Route`
:   Bases: `Enum`

    Represents route directions.

    Values
    :   - **RAMP**
        - **EAST**
        - **WEST**
        - **NORTH**
        - **SOUTH**

## RoutingPosition

*class* `cerebras.sdk.runtime.sdkruntimepybind.``RoutingPosition`
:   Bases: `object`

    Represents a single routing position, which can consist of one or more route values
    for input and one or more route values for output.

    `set_input`(*routes: List[[Route](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]*)
    :   Set a list of routes as the input route position.

        Parameters
        :   **routes** (List[[`Route`](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]) – List of routes to be set as the input route position.

    `set_output`(*routes: List[[Route](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]*)
    :   Set a list of routes as the output route position.

        Parameters
        :   **routes** (List[[`Route`](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]) – List of routes to be set as the output route position.

    `add_input`(*route: [Route](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")*)
    :   Add a route to the input route position.

        Parameters
        :   **route** ([`Route`](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")) – Route to be added to the input route position.

    `add_output`(*route: [Route](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")*)
    :   Add a route to the output route position.

        Parameters
        :   **route** ([`Route`](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")) – Route to be added to the output route position.

    `get_input`() → List[[Route](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]
    :   For this routing position object, return a list of all input routes in the input route
        position.

        Returns
        :   List of all routes in the input route position.

        Return type
        :   List[[`Route`](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]

    `get_output`() → List[[Route](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]
    :   For this routing position object, return a list of all output routes in the output route
        position.

        Returns
        :   List of all routes in the output route position.

        Return type
        :   List[[`Route`](#cerebras.sdk.runtime.sdkruntimepybind.Route "cerebras.sdk.runtime.sdkruntimepybind.Route")]

## get\_edge\_routing

`cerebras.sdk.runtime.sdkruntimepybind.``get_edge_routing`(*edge: Edge*, *routes: List[RoutingPosition]*) → EdgeRouteInfo
:   Construct an edge routing info object from a given edge and routing positions.

    Parameters
    :   - **edge** ([`Edge`](#cerebras.sdk.runtime.sdkruntimepybind.Edge "cerebras.sdk.runtime.sdkruntimepybind.Edge")) – Edge of code region.
        - **routes** (List[[`RoutingPosition`](#cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition "cerebras.sdk.runtime.sdkruntimepybind.RoutingPosition")]) – List of routing positions to be applied to edge.

    Returns
    :   Object containing edge routing info.

    Return type
    :   [`EdgeRouteInfo`](#cerebras.sdk.runtime.sdkruntimepybind.EdgeRouteInfo "cerebras.sdk.runtime.sdkruntimepybind.EdgeRouteInfo")

## Geometry

*class* `cerebras.geometry.geometry.``IntRectangle`(*origin: [IntVector](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")*, *dims: [IntVector](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")*)
:   Bases: `object`

    Defines a rectangle of values.

    Parameters
    :   - **origin** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – Origin of rectangle’s northwest corner.
        - **dims** ([`IntVector`](#cerebras.geometry.geometry.IntVector "cerebras.geometry.geometry.IntVector")) – Width and height of rectangle.

*class* `cerebras.geometry.geometry.``IntVector`(*x: int*, *y: int*)
:   Bases: `object`

    Wraps a tuple of two integer values, often used to specify coordinates or offsets.

    Parameters
    :   - **x** (`int`) – x-coordinate
        - **y** (`int`) – y-coordinate

[previous

SdkRuntime API Reference](api-docs/sdkruntime-api.md "previous page")
[next

SDK Appliance API Reference](api-docs/appliance-api.md "next page")
