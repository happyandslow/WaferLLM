import json
import logging
import os
import posixpath
import socketserver
import sys
import time
import traceback
import urllib
from http import HTTPStatus
from typing import Tuple

import requests

from cerebras.hwdebug_common.viz_service.base import ERROR_CODES, HTTPJsonError
from cerebras.hwdebug_common.viz_service.PluginManager2 import PluginManager
from cerebras.hwdebug_common.viz_service.SimpleWebSocketServer import (
    SimpleWebSocketServer,
)


class MyRequestHandler(SimpleWebSocketServer):
    def __init__(
        self,
        logger: logging,
        mgr: PluginManager,
        proxy: bool,
        webdev: str,
        debug: bool,
        request: bytes,
        client_address: Tuple[str, int],
        server: socketserver.BaseServer,
    ):
        """ subclass the simple file http request handler
        to support the /api plugins """

        self._logger = logger
        self._mgr = mgr
        self._proxy = proxy
        self._webdev = webdev
        self._debug = debug
    
        # for reusing connections
        self.protocol_version = "HTTP/1.1"
        
        super().__init__(request, client_address, server)

    def doproxy(self, url_, qs, proxy_prefix):
        """ for testing, use the real viz server to diagnose issues """
        if "path" in qs:
            if not qs["path"][0].startswith("/"):
                qs["path"][0] = os.path.join(os.getcwd(), qs["path"][0])
        if "graphPath" in qs:
            if not qs["graphPath"][0].startswith("/"):
                qs["graphPath"][0] = os.path.join(
                    os.getcwd(), qs["graphPath"][0]
                )
        req = f"{proxy_prefix}{url_}"
        logging.info("Proxy %s", req)

        r = requests.get(
            req, params=qs, headers={"User-Agent": "curl/7.0", "Accept": "*/*"}
        )

        self.send_response(r.status_code)

        body = r.content
        for name_, val in r.headers.items():
            ##Transfer-Encoding: chunked
            if name_ == "Content-Encoding" and val == "gzip":
                pass  # body = zlib.compress(body)
            if name_ == "Transfer-Encoding" or name_ == "Content-Encoding":
                if val == "chunked" or val == "gzip":
                    pass
                else:
                    self.send_header(name_, val)

            else:
                self.send_header(name_, val)

        logging.info("status: %s", r.status_code)

        if r.ok:
            logging.info("getting content")
            body = r.content

            self.send_header("Content-Length", len(body))
            self.end_headers()

            self.wfile.write(body)
            logging.info("Sent %s bytes", len(body))
        else:
            self.end_headers()
        r.close()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def send_json_error(self, body, status=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        start = time.time()

        self.protocol_version = "HTTP/1.1"

        # `self.path` needs a `?` for proper parsing,
        # but `filescan` API path does not contain it:
        # `/api/v2/filescan&name=*.np[yz]&section=Numpy%20files`
        if "filescan" in self.path and "?" not in self.path:
            self.path = self.path.replace("&", "?", 1)

        req_args = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(req_args.query)
        path = req_args.path

        if path.startswith("/api"):
            if self._proxy:
                self.doproxy(path, qs, "http://viz.cerebras.aws")
                return

            try:
                obj_, svc = self._mgr.activate(path)
            except Exception as e:
                tb = traceback.format_exc()
                self.send_error(
                    HTTPStatus.NOT_FOUND, explain=str(e) + "\n\n" + tb
                )
                return
            finally:
                end = time.time()
                self._logger.info(f"GET {path}: {(end - start) * 1000}ms")                

            if obj_:
                logging.info("plugin %s: %s %s", svc, self.path, obj_)
                try:
                    if obj_.handle_get(self, self.path, req_args, qs):
                        return
                except HTTPJsonError as e:
                    data = {"hint": e.hint, "code": e.error_code}
                    data.update(e.data)
                    body = json.dumps(data)
                    self.send_json_error(body.encode("utf-8"), status=e.status)
                    return
                except Exception as e:
                    tb = traceback.format_exc()
                    data = {
                        "traceback": tb,
                        "exception": str(e),
                        "code": ERROR_CODES.EXCEPTION,
                        "hint": f"unexpected exception: {e} stack trace: {traceback}",
                    }

                    body = json.dumps(data)
                    try:
                        self.send_json_error(body.encode("utf-8"))
                    except OSError:
                        # Give up when error backtrace cannot be sent. Most likely there is a socket closure issue SW-73700
                        logging.warning("OSError captured but cannot send backtrace to gui frontend")

                    return
                finally:
                    end = time.time()
                    self._logger.info(f"GET {path}: {(end - start) * 1000}ms")

                
        else:
            if self._webdev:
                self.doproxy(path, qs, self._webdev)
                return

        super().do_GET()

        end = time.time()
        self._logger.info(f"GET {path}: {(end - start) * 1000}ms")

    def do_POST(self):
        self.protocol_version = "HTTP/1.1"
        req_args = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(req_args.query)
        path = req_args.path

        if path.startswith("/api"):

            obj_, svc = self._mgr.activate(path)

            if obj_:
                logging.info("plugin %s: %s %s", svc, self.path, obj_)
                try:
                    logging.warning("headers: %s", self.headers)
                    content_len = int(self.headers.get("Content-Length", 0))
                    post_body = self.rfile.read(content_len)
                    if obj_.handle_post(
                        self, self.path, req_args, qs, post_body
                    ):
                        return
                except Exception as e:
                    tb = traceback.format_exc()
                    data = {
                        "traceback": tb,
                        "code": "E0001",
                        "hint": f"unexpected exception: stack trace: {traceback}",
                    }

                    body = json.dumps(data).encode("utf-8")
                    self.send_error(
                        HTTPStatus.INTERNAL_SERVER_ERROR, str(e) + "<br>" + body
                    )
                    return

        super().do_POST()

    def log_message(self, format_, *args):
        """Log an arbitrary message.

        This is used by all other logging functions.  Override
        it if you have specific logging wishes.

        The first argument, FORMAT, is a format string for the
        message to be logged.  If the format string contains
        any % escapes requiring parameters, they should be
        specified as subsequent arguments (it's just like
        printf!).

        The client ip and current date/time are prefixed to
        every message.

        """

        if self._debug:
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (
                    self.address_string(),
                    self.log_date_time_string(),
                    format_ % args,
                )
            )

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            "DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT",
        )
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")

        super().end_headers()

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)
        """
        # abandon query parameters
        original = path

        req_args = urllib.parse.urlparse(original)
        qs = urllib.parse.parse_qs(req_args.query)
        path = req_args.path

        # handle request for server side file
        if path.startswith("/api/v2/services"):
            if "file" in qs:
                path = os.path.join(
                    self._mgr.path_prefix, qs["file"].replace("/", "_")
                )
                return path

            if path == "/api/v2/services":
                fn = "top"
            else:
                fn = path.replace("/api/v2/services/", "")
            if fn not in self._mgr.plugins:
                logging.info("Loading plugin %s", fn)
                self._mgr.load_plugin(fn)
            path = self._mgr.plugins[fn].gen_file(qs)
            return path

        if (path == "/api/v1/file") and qs["path"]:

            logging.info("API %s with path %s", path, qs["path"])
            path = qs["path"][0]
            spath = path.split("/")

            # dont allow .. in paths for security
            if ".." in spath:
                logging.error(".. illegal in path (%s)", path)
                return ""

            # strip any path, as only the directory is allowed
            if not path.startswith("/"):
                path = os.path.join(self._mgr.path_prefix, *spath)

            if os.path.isfile(path) is False:
                logging.error("File: %s not found", path)
            return path

        path = path.split("#", 1)[0]
        parsed = urllib.parse.unquote(path)
        path = posixpath.normpath(parsed)
        words = path.split("/")
        words = filter(None, words)
        path = self._mgr.web_dir
        for word in words:
            _, word = os.path.splitdrive(word)
            _, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)

        if os.path.isdir(path):
            self._logger.info("Path is to directory, converting to index")
            path = os.path.join(path, "index.html")
        else:
            if not os.path.isfile(path):
                self._logger.warning("Path %s is to a non-existent file", path)

        self._logger.info("get: %s to path:%s", original, path)
        return path
