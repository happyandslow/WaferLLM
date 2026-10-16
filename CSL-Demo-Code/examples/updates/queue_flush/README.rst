Reuse queues: output queue
==========================

On WSE-3, each output queue can only bind to a color. The application
sharing multiple colors on an output queue on WSE-2 no longer works on
WSE-3. The user may not be able to port this application on WSE-3 because
of insufficient output queues.

WSE-3 allows the user to repurpose the input/output queues with a different
color by `@queue_flush`. This new feature can resolve the insufficient
queues with a reasonable cost.

This example demonstrates how to reuse an output queue in the sender and
an input queue in the receiver on the WSE-3 architecture.

There are two PEs in this example, a sender (queue_flush_sender.csl) on
the left and a receiver (queue_flush_receiver.csl) on the right.
The sender sends two data sets to the receiver via C10 and C12. Both data
sets are handled by the output queue 2.

The output queue must be drained before it can be reused. However the
user only knows when the microthread finishes because the microthread can
activate a local task or unblock a color to notify the user about its
termination. The termination of a microthread only confirms that it
pushes all data into the output queue. The corresponding output queue is
usually non-empty at the time the microthread terminates.
WSE-3 provides an event handling mechanism to fill the gap between the
termination of a microthread and when the output queue becomes empty.

Here is how it works:
The user calls `@queue_flush(oq)` after the microthread operation, for
example, `@mov32` which sends the data to an output queue `oq`. When
`oq` becomes empty, WSE-3 will trigger T29, the same way a teardown
wavelet does. The user has to register a handler, `foo`, which can
repurpose the output queue. `@set_empty_queue_handler(foo, oq)` binds
the handler `foo` to the output queue `oq`.
The handler `foo` does three things:
1. reconfigure the color field of the output queue `oq` by
output_queue_config.encode_output_queue(oq, new color)
2. next operation after the output has reconfigured the color
`transmit2_id` is activated to send the data2.
3. clear the state of queue flush
`queue_flush.exit(oq)` is necessary, otherwise the subsequent T29 will
call `foo` again.

The sender has the following sequence:
1. send data1 to output queue `oq` which binds to color `out_color1`.
2. `done_transmit1` is triggered when data1 has been sent to `oq`.
3. `done_transmit1` reconfigures the color field of `oq` and calls
`@queue_flush(oq)`.
4. whole data1 has been moved out to the router, i.e. `oq` becomes empty.
5. T29 is triggered because `oq` is empty.
6. T29 calls the handler foo().
7. `foo()` binds `oq` with `out_color2`.
8. `foo()` activates `transmit2_id` to send data2.
9. `foo()` calls `queue_flush.exit()` to clear the queue flush state.
10. `transmit2_id` is triggered to send out data2.
11. `transmit2_id` send data2 to output queue `oq` which binds to color
`out_color2`.
12. `exit()` is triggered when data2 has been pushed into `oq`.

Be careful not to issue `@queue_flush(oq)` before `@mov32`, otherwise
it is possible that WSE-3 immediately triggers T29 when `oq` is empty
because `@mov32` is not started yet to push data into `oq`.

The `queue_flush.exit()` is necessary to clear the state of queue flush.
Otherwise the subsequent T29 due to D2Hs will call `foo` again, and then
the kernel will stall.

The receiver receives two data sets from C10 and C12 sequentially. It is
safe to reuse the input queue. We just reconfigure input queue `iq` after
data1 is received into `result1`. Then we can continue receiving data2
on the same `iq`. This example does not need to use `@queue_flush(iq)`
because `iq` is already empty at the time `done_receive1` is triggered.
`done_receive1` can reconfigure `iq` safely.

The receiver introduces a dummy loop to increase the delay between
`@queue_flush(oq)` and T29 so we can confirm if such T29 is triggered
by emptyness of `oq` easily.
