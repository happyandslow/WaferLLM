---
title: "SDK GUI"
url: "https://sdk.cerebras.net/debug/sdk-gui"
crawled_at: "2026-02-14 23:54:57"
---

# SDK GUI

> Source: [https://sdk.cerebras.net/debug/sdk-gui](https://sdk.cerebras.net/debug/sdk-gui)

Follow the documentation to launch `sdk_debug_shell visualize`, pointing to your test artifact directory. Open the URL that the command shows.

The below diagram shows a summary of the functionality:

> [![../_images/sdk-gui-example.png](../_images/sdk-gui-example.png)](../_images/sdk-gui-example.png)

The artifacts from your directory will be loaded and shown visually for debugging.

## Change directory

If you wish to load a different directory, enter the new directory click **Submit**.

> [![../_images/change-directory.png](../_images/change-directory.png)](../_images/change-directory.png)

## Dark and light modes

To switch between dark and light modes, click the icons at the top right.

Switch to dark mode:

> [![../_images/dark.png](../_images/dark.png)](../_images/dark.png)

Switch to light mode:

> [![../_images/light.png](../_images/light.png)](../_images/light.png)

## Routes and colors

To view the routes on the fabric, select the color IDs.

To view all the routes, click **Select All**.

> [![../_images/sdk-gui-routs-and-colors.png](../_images/sdk-gui-routs-and-colors.png)](../_images/sdk-gui-routs-and-colors.png)

Expanding/collapsing a color ID shows/hides a list of associated color names.

> [![../_images/sdk-color-names.png](../_images/sdk-color-names.png)](../_images/sdk-color-names.png)

## Fabric view

Move the fabric around by dragging it. To zoom in and out of the fabric, use the mouse scroll functionality.

> [![../_images/sdk-gui-fabric-view.png](../_images/sdk-gui-fabric-view.png)](../_images/sdk-gui-fabric-view.png)[![../_images/sdk-gui-zoom.png](../_images/sdk-gui-zoom.png)](../_images/sdk-gui-zoom.png)

- To recenter the fabric, click the Re-center button.

  > [![../_images/sdk-gui-re-center.png](../_images/sdk-gui-re-center.png)](../_images/sdk-gui-re-center.png)
- To zoom into a particular process element (PE), enter a specific **PE Coordinate**.

  > [![../_images/sdk-gui-pe-coordinate.png](../_images/sdk-gui-pe-coordinate.png)](../_images/sdk-gui-pe-coordinate.png)

For pre-execution artifacts, only routing information is available for viewing. For post-execution artifacts, you can view additional information as documented below.

Double-click a PE to open the instruction trace, source code, wavelet trace and symbols panels.

> [![../_images/sdk-gui-double-click.png](../_images/sdk-gui-double-click.png)](../_images/sdk-gui-double-click.png)

To zoom in and center the selected PE, click the bottom-right of the page

> [![../_images/sdk-gui-selected-pe.png](../_images/sdk-gui-selected-pe.png)](../_images/sdk-gui-selected-pe.png)[![../_images/sdk-gui-selected-pe-zoom.png](../_images/sdk-gui-selected-pe-zoom.png)](../_images/sdk-gui-selected-pe-zoom.png)

When hovering over the fabric, the coordinates of the currently hovered PE are displayed on the bottom info pane:

> [![../_images/sdk-gui-selected-pe-coordinates.png](../_images/sdk-gui-selected-pe-coordinates.png)](../_images/sdk-gui-selected-pe-coordinates.png)

If your work directory uses a fabric size greater than 20 x 20, there is a scalable fabric view:

> [![../_images/sdk-gui-scalable-fabric-view.png](../_images/sdk-gui-scalable-fabric-view.png)](../_images/sdk-gui-scalable-fabric-view.png)

The full dimensional view is shown on the right fabric. The left fabric reflects the area highlighted by the dark square on the right fabric. A dark square is drawn by mouse click on the right fabric surface.

> [![../_images/sdk-gui-left-fabric.png](../_images/sdk-gui-left-fabric.png)](../_images/sdk-gui-left-fabric.png)[![../_images/sdk-gui-center-fabric.png](../_images/sdk-gui-center-fabric.png)](../_images/sdk-gui-center-fabric.png)[![../_images/sdk-gui-right-fabric.png](../_images/sdk-gui-right-fabric.png)](../_images/sdk-gui-right-fabric.png)

Left, right, top, and bottom buttons allow you to navigate across the full dimensional view.

> [![../_images/sdk-gui-directional-buttons.png](../_images/sdk-gui-directional-buttons.png)](../_images/sdk-gui-directional-buttons.png)

Control moving the selected factory area to the top-left by hitting the **Re-corner** button.

> [![../_images/sdk-gui-re-corner-button.png](../_images/sdk-gui-re-corner-button.png)](../_images/sdk-gui-re-corner-button.png)

The location of the selected routes on the selected fabric area on the left is highlighted in red on the right / full view of the fabric

> [![../_images/sdk-gui-right-full-view.png](../_images/sdk-gui-right-full-view.png)](../_images/sdk-gui-right-full-view.png)[![../_images/sdk-gui-red-square-view.png](../_images/sdk-gui-red-square-view.png)](../_images/sdk-gui-red-square-view.png)

To zoom in/out/reset the left fabric, use the zoom control buttons on the top-right.

> [![../_images/sdk-gui-zoom-in-out-reset-buttons.png](../_images/sdk-gui-zoom-in-out-reset-buttons.png)](../_images/sdk-gui-zoom-in-out-reset-buttons.png)

Mouse scroll is not supported to zoom in or out for a scalable fabric view.

> > **Note**
> > Note
> > The fabric cannot be moved by dragging for a scalable fabric view.

## Instruction traces

You can view the instructions, tasks, statistics, uthreads or micro threads from the instruction trace panel.

To view the instructions, from the dropdown list, select **Instructions**.

> [![../_images/sdk-gui-instruction-trace.png](../_images/sdk-gui-instruction-trace.png)](../_images/sdk-gui-instruction-trace.png)

To view the tasks, from the dropdown list, select **Tasks**.

> [![../_images/sdk-gui-instruction-tasks.png](../_images/sdk-gui-instruction-tasks.png)](../_images/sdk-gui-instruction-tasks.png)

To view the statistics related to the instructions, from the dropdown list, select **Statistics**.

> [![../_images/sdk-gui-instruction-statistics.png](../_images/sdk-gui-instruction-statistics.png)](../_images/sdk-gui-instruction-statistics.png)

To view the micro-threads, from the dropdown list, select **Uthreads**.

> [![../_images/sdk-gui-instruction-uthreads.png](../_images/sdk-gui-instruction-uthreads.png)](../_images/sdk-gui-instruction-uthreads.png)

## Source code

To view the source code for an instruction, follow the steps below:

- To view all the instructions, in the instruction trace panel, select **Instructions**.
- Click an instruction that has the source code file and line number.

In the **Source Code** panel, you are navigated to the line number corresponding to the instruction.

> [![../_images/sdk-gui-src-example.png](../_images/sdk-gui-src-example.png)](../_images/sdk-gui-src-example.png)

## Wavelet traces

Wavelet traces are displayed in the wavelet trace panel.

- From the color dropdown, select a color. To select all colors, click **Select All**.
- From the wavelet format dropdown, select a format. This formats the header and data fields of the wavelet.

  > [![../_images/sdk-gui-wavelet-traces.png](../_images/sdk-gui-wavelet-traces.png)](../_images/sdk-gui-wavelet-traces.png)

## Timeline: Wavelet traces

To show the timeline for a wavelet, click a color and a PE. The timeline for wavelet appears in the bottom pane.

> [![../_images/sdk-gui-timeline-wavelet.png](../_images/sdk-gui-timeline-wavelet.png)](../_images/sdk-gui-timeline-wavelet.png)

The timeline has a list of PEs to the left and its corresponding timeline area with wavelet entries to the right. The axis on the top shows the cycle range of the view area. The rectangular bar represents a wavelet. The width of the rectangular bar represents how long the wavelet takes to travel from the current PE to the destination PE. The status of the wavelet is represented by the color of the bar.

### Legend terminology

- **Active**: Sender PE can send a wavelet without any backpressure or delay. Shown as green.
- **Delayed**: The switch/router (represented by the circle) output is congested due to multiple input sources. Shown as yellow.
- **Backpressure**: Destination PE is busy and cannot receive any wavelets from sender PE. The sender PE is back-pressured. Shown as red.
- **Idle**: There is no wavelet in the router. Shown as empty spaces in the timeline for a PE.
- **Wavelet Type**:
  :   - **Control wavelet**: Wavelet that changes control, such as routing to a compute element (CE). Shown as blue.
      - **Data wavelet**: Wavelet that sends data. The active, delayed, and back-pressured wavelets are data wavelets.

## Timeline: Instruction traces

To view the timeline of instruction traces, click the instruction trace icon next to the header title.

> [![../_images/sdk-gui-instruction-trace-timeline.png](../_images/sdk-gui-instruction-trace-timeline.png)](../_images/sdk-gui-instruction-trace-timeline.png)

The timeline area has rectangular bars depicting a timeline item. The timeline item depicts the number of cycles taken to execute a line of source code. There are tasks running in a PE that executes the source code. The color of the timeline item is linked to the task that is executing. The color legend below the timeline area shows the task numbers corresponding to the timeline color.

## Timeline: Combined view

Combined view shows both the wavelet traces and the instruction traces associated with the PE and color. Whenever a new selection is made, two items are added to the timeline area: one corresponding to the timeline of wavelet traces, and the other corresponding to the timeline of instruction traces.

> [![../_images/sdk-gui-timeline-combined-view.png](../_images/sdk-gui-timeline-combined-view.png)](../_images/sdk-gui-timeline-combined-view.png)

## Timeline: Navigation controls

You can delete a PE from the timeline view. Hover over the PE and click the “trash” icon. To clear all the PEs from the timeline view, click the “trash” icon in the upper-right corner.

By default, the timeline view shows only the first few hundred cycles. Not all cycles are shown in the initial view. To view the left or right, you can scroll the timeline. There is a red rectangular bar above the timeline area. The red bar represents the current view area. Dragging the red bar left or right moves the timeline view correspondingly.

You can also zoom out to show more cycles in the view or zoom in to show fewer cycles in the view.

> [![../_images/sdk-gui-timeline-navigation-controls.png](../_images/sdk-gui-timeline-navigation-controls.png)](../_images/sdk-gui-timeline-navigation-controls.png)

There are four buttons related to zoom in the top-right corner. The “**-**” button zooms out; the “**+**” button zooms in. To bring the zoom back to the default zoom, use the **Reset** button. The **Zoom to fit** button brings all the cycles into view.

The green bars indicate Active wavelet traces that don’t have a delay. (See Legend terminology above).

Because the rectangular bars representing the timeline are scaled, the green bars may not be visible in this view depending on the time range selected.

Another way to zoom the timeline view is to resize the red bar that is at the top of the timeline. By resizing the red bar, you can zoom the timeline view below in and out.

## Debug pane

To show the details in the debug pane to the right, hover over a timeline item. For wavelets, the detail pane shows the cycles that the wavelet took to reach the next PE, a wavelet ID that is shown for debug purposes, the status of the wavelet, the direction (East, West, North, South), and whether it is a control wavelet. The pane also shows index and data fields.

> [![../_images/sdk-gui-debug-pane.png](../_images/sdk-gui-debug-pane.png)](../_images/sdk-gui-debug-pane.png)

To keep a timeline item in the debug pane, click on it. Clicking anywhere on the timeline area unselects the timeline item and removes it from the debug pane.

## Settings dialog

You can enter a new work directory in the textbox at the top of the application. When you click the **Submit** button, the **Settings Dialog** appears.

> [![../_images/sdk-gui-settings-dialog.png](../_images/sdk-gui-settings-dialog.png)](../_images/sdk-gui-settings-dialog.png)

The Settings dialog has four settings: **Compile directory**, **Run directory**, **Core file path**, and **Source directory**. The application pre-populates these settings for you when you provide a work directory. Of these four settings, only the **Compile directory** is required.

Three of the settings contain dropdown menus: the **Compile directory**, **Run directory**, and **Core file path**. Within the work directory, there may be multiple compile directories, run directories and core file paths. You can switch between multiple values for each in a dropdown. You can also clear the dropdown and enter a custom value for these settings.

If there is no run directory provided, then the app cannot show the instruction trace, wavelet trace or the timeline view.

If there is no core file path provided, the routes shown in the fabric are compile time routes, and not the routes of the actual simulation.

To save the settings, click the **Save** button. To revert to the previous settings, click the **Close** button.

## Terminal

To view the **Terminal** page, click the terminal icon on the top right. This page provides a UNIX command console for the root folder where the SDK-GUI web server is running.

> [![../_images/sdk-gui-terminal.png](../_images/sdk-gui-terminal.png)](../_images/sdk-gui-terminal.png)

## Panes

You can toggle panes on and off.

> [![../_images/sdk-gui-pane-toggles.png](../_images/sdk-gui-pane-toggles.png)](../_images/sdk-gui-pane-toggles.png)

Use the resize handle to resize panes.

> [![../_images/sdk-gui-pane-resize-handle.png](../_images/sdk-gui-pane-resize-handle.png)](../_images/sdk-gui-pane-resize-handle.png)

You can expand panes to a separate page. To do this, click the **Expand** icon.

> > [![../_images/sdk-gui-pane-expand.png](../_images/sdk-gui-pane-expand.png)](../_images/sdk-gui-pane-expand.png)
>
> When you click the expand icon, the pane appears as a full page, as shown below.
>
> > [![../_images/sdk-gui-pane-expanded.png](../_images/sdk-gui-pane-expanded.png)](../_images/sdk-gui-pane-expanded.png)
>
> You can move from one pane to another. Click the icon on the left navigation bar.
>
> > [![../_images/sdk-gui-pane-left-nav.png](../_images/sdk-gui-pane-left-nav.png)](../_images/sdk-gui-pane-left-nav.png)

- The first icon (Home) shows the original view with all the panes.
- The second icon shows the instructions trace pane along with the source code pane.
- The third icon shows the source code pane.
- The fourth icon shows the wavelet trace pane.
[previous

Debugging Guide](debug/debugging.md "previous page")
[next

SdkRuntime API Reference](api-docs/sdkruntime-api.md "next page")
