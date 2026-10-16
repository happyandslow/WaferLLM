"""
Code for polling for a condition
"""
import logging
import time


def poll_fn_until_true(
    poll_fn,
    args=None,
    kwargs=None,
    min_poll_interval=None,
    max_poll_interval=None,
    timeout=None,
):
    """
    Calls poll_fn(*args, **kwargs) in a loop, until the return value
    evaluates to True in a boolean context.
    (Note that it does not have to actually be a bool; it could be a str, int,
    list, etc, as long as it can be cast to a bool.)

    We use a simple back-off algorithm, polling frequently at first, then
    polling less frequently as the wait time increases, so we trade off
    between responsiveness and efficiency.  (e.g.  if something finishes in a
    few seconds, we want to notice quickly, but if something is going to take
    45 minutes, it's wasteful to poll every second.)

    :param callable poll_fn: Function to poll (call repeatedly)
    :param list args: list of arguments to pass: poll_fn(*args)
    :param dict kwargs: dict of keyword arguments to pass: poll_fn(**kwargs)
    :param int min_poll_interval:
      Shortest time in seconds to pause between calling the poll_fn().
      e.g. if poll_fn() is expensive and we expect to poll for a long time,
      we should set it to a longer time.
    :param int max_poll_interval:
      Longest time in seconds to pause between calling the poll_fn().
    :param int timeout:
      We'll raise TimeoutError if poll_fn() doesn't return True value before
      this timeout expires.  Specified in seconds.

    Returns the non-False return value.

    Example: wait for a file to appear:
      >>> import os
      >>> from cerebras.common.poll import poll_fn_until_true
      >>> poll_fn_until_true(os.path.exists, args=["/tmp/foo.txt"], timeout=60)

    Example: use the return value:
      >>> from cerebras.common.poll import poll_fn_until_true
      >>> def poll_fn():
      ...     print("Check...")
      ...     global counter
      ...     counter += 1
      ...     if counter >= 2:
      ...         return "SOMETHING"  # Evaluates to True
      ...     return ""  # Evaluates to False
      ...
      >>> counter = 0
      >>> ret = poll_fn_until_true(poll_fn)
      Check...
      Check...
      >>> assert ret == "SOMETHING"
    """
    logger = logging.getLogger(__name__)
    # Process args
    if not args:
        args = []
    if not kwargs:
        kwargs = {}
    try:
        poll_fn_name = poll_fn.__name__
    except AttributeError:
        poll_fn_name = str(poll_fn)
    # By default, poll every second at first:
    if not min_poll_interval:
        min_poll_interval = 1
    # For long wait times, we start backing off (increasing our sleep time)
    # up to this maximum.  Setting it too low is inefficient, since we assume
    # calling poll_fn() has a cost.  But setting it too high increases run
    # durations, etc.  Default that seems reasonable for our use cases:
    if not max_poll_interval:
        max_poll_interval = 30
    # We don't want to start our poll duration backoff until it's no longer
    # a "quick" wait, otherwise it feels sluggish to humans.  More than about
    # 15 seconds no longer seems "quick", so start our backoff then:
    backoff_start = 15
    start_time = time.time()
    # Loop
    time_elapsed = 0
    sleep_duration = min_poll_interval
    ret = None
    while True:
        # Poll
        ret = poll_fn(*args, **kwargs)
        time_elapsed = time.time() - start_time
        msg = f"Poll {poll_fn_name} args={str(args)} kwargs={str(kwargs)}"
        logger.debug(msg)
        if ret:
            break
        # Check for timeout
        if timeout and time_elapsed > timeout:
            msg = f"Exceeded timeout {timeout} polling with"
            msg += f" {poll_fn_name} args={str(args)} kwargs={str(kwargs)}"
            raise TimeoutError(msg)
        #
        # Sleep
        time.sleep(sleep_duration)
        time_elapsed = time.time() - start_time
        if time_elapsed > backoff_start:
            # If this seems to be taking a long time, start sleeping for longer
            # intervals to be more efficient, but don't sleep past a timeout
            # if it's expiring soon.
            sleep_duration = min(sleep_duration + 1, max_poll_interval)
            if timeout:
                time_remaining = timeout - time_elapsed
                if time_remaining > 0:
                    sleep_duration = min(sleep_duration, time_remaining)
    return ret
