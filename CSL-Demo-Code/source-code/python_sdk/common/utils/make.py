""" GNU make helpers.
"""
import os
import re
import signal

class JobClient:
    """
    Class to become a client for the GNU Make job server.
    See https://www.gnu.org/software/make/manual/html_node/Job-Slots.html for
    details on how the Make job server works.

    At a high level, the MAKEFLAGS environment variable will be set with a
    special jobserver field like: --jobserver-fds=3,4.  In this case this means
    file descriptor 3 should be read one character at a time to get a token,
    and when the token is done, it should be returned on file descriptor 4.

    You can think of the tokens as representing additional CPU's beyond what
    is already running.  Consider having a multi-process program running that
    is cognizant of the job server.  That program can run one process without
    needing any tokens, because the ability to run a single process is implicit.
    However if you can use more CPU's, then you should try to obtain a token
    from the job server.  Once you have the token you can launch the extra jobs,
    and return the tokens when done.

    If MAKEFLAGS does not indicate that a job server is active, then this
    class will be able to get and put any number of tokens without blocking.
    The reason for this is to have this class act as a no-op for cases where
    the job server is not in use.  When the job server *is* in use, this class
    should act as a limiter for how much parallelism can be acheived.

    The way to use this class would be:

    with JobClient(init_tokens=3) as jc:
        # do some stuff, assuming you have jc.num_tokens() tokens acquired
        # inside this block.  Note that we will be blocked until we get
        # all tokens requested in init_tokens.

    with JobClient(init_tokens=0) as jc:
        while True:
            # The following will timeout after 5 seconds, so you might
            # get anywhere between 0 and 5 tokens.
            got_tokens = jc.try_get_tokens(5, timeout=5)
            print("I got %d tokens", got_tokens)
            # do something based on the number of tokens you got
            jc.put_tokens(got_tokens)
    """

    class TimeoutError(Exception):
        """
        Simple exception for timeout.
        """
        pass

    def __init__(self, init_tokens=1, init_timeout=0):
        """
        Note we do not actually try to get the tokens until __enter__ is called.
        An init_timeout of 0 means to block until all tokens are obtained.
        Otherwise, you will live with whatever tokens that you got by the time
        the timer expires.

        :param int init_tokens: number of tokens to try to get initially
        :param int init_timeout: timeout associated with the initial tokens.

        """
        self.read_fileno = -1
        self.write_fileno = -1
        self.read_fd = None
        self.write_fd = None
        self.chars = [] # chars represents all tokens that have been obtained
        self.init_tokens = init_tokens
        self.init_timeout = init_timeout

        if "MAKEFLAGS" in os.environ:
            makeflags = os.environ["MAKEFLAGS"]
            match = re.search(r"--jobserver-(fds|auth)=(\d+),(\d+)", makeflags)
            if match:
                # Since these are shared pipes, we duplicate the file
                # descriptors so that we can safely close without impacting
                # other processes.
                self.read_fileno  = os.dup(int(match.group(2)))
                self.write_fileno = os.dup(int(match.group(3)))
            else:
                return

    def get_tokens(self, num_tokens=1):
        """
        Grab num_tokens tokens.  Note that this will BLOCK if tokens are not
        available.

        :param int num_tokens: number of tokens to get
        """
        # this will block
        while num_tokens:
            if self.read_fd:
                # read will block if tokens unavailable!
                self.chars.append(self.read_fd.read(1))
            else:
                self.chars.append("?")
            # print("GET token!")
            num_tokens -= 1

    def try_get_tokens(self, num_tokens=1, timeout=1):
        """
        Try to grab num_tokens tokens.  If the timeout count expires, then this
        method will return, having grabbed as many tokens as it could during
        that time.

        :param int num_tokens: number of tokens to get
        :param int timeout: number of seconds before timeout
        :returns: number of tokens we were able to get
        """
        def handler(_signum, _frame):
            """
            Local handler to raise error on timeout.
            """
            raise TimeoutError()
        init_tokens = self.num_tokens()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            self.get_tokens(num_tokens)
        except TimeoutError:
            pass
        finally:
            signal.alarm(0)
        # return number of tokens we were able to get
        return self.num_tokens() - init_tokens

    def num_tokens(self):
        """
        Number of tokens currently acquired.
        """
        return len(self.chars)

    def put_tokens(self, num_tokens=1):
        """
        Put num_tokens tokens back.

        :param int num_tokens: number of tokens to put
        """
        while num_tokens and self.chars:
            if self.write_fd:
                self.write_fd.write(self.chars.pop())
            else:
                char = self.chars.pop()
                if char != "?":
                    raise RuntimeError("Unexpected token!")
            # print("PUT token!")
            num_tokens -= 1

    def open(self):
        """
        Open file descriptors associated with the job server.
        """
        if self.read_fileno >= 0 and self.write_fileno >= 0:
            # print("Open for read:", self.read_fileno)
            # print("Open for write:", self.write_fileno)

            self.read_fd = os.fdopen(self.read_fileno, "rb", buffering=0)
            self.write_fd = os.fdopen(self.write_fileno, "wb", buffering=0)
            return self
        return None

    def close(self):
        """
        Close file descriptors associated with the job server.
        """
        if self.read_fd:
            self.read_fd.close()
        if self.write_fd:
            self.write_fd.close()

    def __enter__(self):
        """
        Enter context.
        """
        if not self.open():
            return self
        if self.init_timeout == 0:
            self.get_tokens(self.init_tokens)
        else:
            self.try_get_tokens(self.init_tokens, self.init_timeout)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """
        Exit context.
        All tokens should be returned.
        """
        self.put_tokens(len(self.chars))
        self.close()

if __name__ == '__main__':
    # Very very incomplete and manual testing
    #
    # Makefile:
    # all: \
    # 	a.out \
    # 	b.out \
    # 	c.out \
    # 	d.out
    #
    #
    # %.out:
    # 	+python3 make.py
    # 	touch $@
    #
    # clean:
    # 	rm *.out
    #
    # Imagine the Makefile above exists.  Then a "make clean; make -j2 all"
    # should activate the job server, and tokens will be shared among the
    # job clients.
    import time
    with JobClient(init_timeout=1) as jc:
        print("sleep")
        time.sleep(5)
        #assert(False)
    with JobClient(init_timeout=1) as jc:
        print("sleep")
        time.sleep(5)
        #assert(False)
