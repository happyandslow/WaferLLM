"""
This code implements a remote-terminal websocket connection to /bin/bash

The counterpart to this server is the client code in src/viz-ui/csviz/term.html and
src/viz-ui/clients/term

"""
import fcntl
import html
import logging
import os
import pty
import select
import struct
import subprocess
import termios
import threading
import time
from http import HTTPStatus

import psutil

from cerebras.hwdebug_common.viz_service.HTTPWebSocketsHandler import (
    HTTPWebSocketsHandler,
)


class SimpleWebSocketServer(HTTPWebSocketsHandler):
    def set_winsize(self, fd, cols, rows, xpix=0, ypix=0):
        logging.info("Resize: %sx%s", cols, rows)
        winsize = struct.pack("HHHH", rows, cols, xpix, ypix)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

    def on_ws_message(self, message):
        if message is None:
            message = ''
        if type(message) is str:
            if message.startswith('ESCAPED|'):
                # format is: ESCAPED|RESIZE;199;024
                fields = message.split('|')[1].split(';')
                if fields[0] == "RESIZE":
                    self.set_winsize(self.fd, int(fields[1]), int(fields[2]))
            else:
                os.write(self.fd, message.encode('utf-8'))
        elif type(message) is bytes:
            if message.decode('utf-8').startswith('ESCAPED|'):
                # format is: ESCAPED|RESIZE;199;024
                fields = message.decode('utf-8').split('|')[1].split(';')
                if fields[0] == "RESIZE":
                    self.set_winsize(self.fd, int(fields[1]), int(fields[2]))
            else:
                os.write(self.fd, message)
        else:
            assert False

    def on_ws_closed(self):
        logging.info('%s', 'websocket closed')
        self.shell_closed = True

    def on_ws_connected(self):
        """If client asked for a web socket, then we need to connect up to a bash session"""
        logging.info("new client connected")
        logging.info(
            "System memory used (MB): %d", psutil.virtual_memory()[3] / 1000000
        )
        process = psutil.Process()
        logging.info(
            "Process memory (MB): %d, pid: %d",
            process.memory_info().rss / 1000000,
            os.getpid(),
        )

        # create child process attached to a pty we can read from and write to
        (child_pid, fd) = pty.fork()
        self.fd = fd
        if child_pid == 0:
            # this is the child process fork.
            # anything printed here will show up in the pty, including the output
            # of this subprocess
            logging.info("Starting /bin/bash")
            time.sleep(0.01)
            subprocess.run('/bin/bash')
            logging.info("Stopped /bin/bash")
        else:
            logging.info("Child pid: %d", child_pid)
            t = threading.Thread(
                name="xterm_poll_{child_pid}",
                target=self.poll_xterm,
                args=(fd,),
            )
            t.start()

    def poll_xterm(self, fd):
        self.send_message("Starting /bin/bash on system\r\n".encode('utf-8'))
        self.shell_closed = False
        self.shellfd = fd
        max_read_bytes = 1024 * 20
        try:
            while self.shell_closed is False:
                time.sleep(0.01)
                if fd:
                    timeout_sec = 0
                    (data_ready, _, _) = select.select(
                        [fd], [], [], timeout_sec
                    )
                    if data_ready:
                        output = os.read(fd, max_read_bytes)
                        logging.debug(
                            "Sending XTERM: %s", output.decode('utf-8')[0:50]
                        )
                        self.send_message(output)
        except Exception as err:
            logging.error("Exception processing bash client")

        logging.info("Aborting on_ws_connected monitor")

    def send_error(self, code, message=None, explain=None):
        """Send and log an error reply.

        Arguments are
        * code:    an HTTP error code
                   3 digits
        * message: a simple optional 1 line reason phrase.
                   *( HTAB / SP / VCHAR / %x80-FF )
                   defaults to short entry matching the response code
        * explain: a detailed message defaults to the long entry
                   matching the response code.

        This sends an error response (so it must be called before any
        output has been generated), logs the error, and finally sends
        a piece of HTML explaining the error to the user.

        """

        try:
            shortmsg, longmsg = self.responses[code]
        except KeyError:
            shortmsg, longmsg = '???', '???'
        if message is None:
            message = shortmsg
        if explain is None:
            explain = longmsg
        self.log_error("code %d, message %s", code, message)
        self.send_response(code, message)
        self.send_header('Connection', 'close')

        # Message body is omitted for cases described in:
        #  - RFC7230: 3.3. 1xx, 204(No Content), 304(Not Modified)
        #  - RFC7231: 6.3.6. 205(Reset Content)
        body = None
        if code >= HTTPStatus.OK and code not in (
            HTTPStatus.NO_CONTENT,
            HTTPStatus.RESET_CONTENT,
            HTTPStatus.NOT_MODIFIED,
        ):
            # HTML encode to prevent Cross Site Scripting attacks
            # (see bug #1100201)
            content = self.error_message_format % {
                'code': code,
                'message': html.escape(message, quote=False),
                'explain': '<pre>'
                + html.escape(explain, quote=False)
                + '</pre>',
            }
            body = content.encode('UTF-8', 'replace')
            self.send_header("Content-Type", self.error_content_type)
            self.send_header('Content-Length', str(len(body)))
        self.end_headers()

        if self.command != 'HEAD' and body:
            self.wfile.write(body)
