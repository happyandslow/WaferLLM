"""
Base class for all cs_viz services

This class defines a base set of messages that are generally available for use

Custom messages from each plugin can also be generated 
with MsgId.NOTIFY with msg = "{plugin}.{type/data}"

Examples:
    MsgId.NOTIFY

"""

import glob
import gzip
import json
import logging
import os
from abc import ABC, abstractmethod
from http import HTTPStatus
from pathlib import Path

from google.protobuf.json_format import MessageToDict, MessageToJson

from cerebras.common.decorators import deprecated
from cerebras.pb.viz.viz_explorer_pb2 import VizMetaItem, VizMetaType


class NoRegistryEntry(Exception):
    pass


class CustomAdapter(logging.LoggerAdapter):
    """
    This example adapter expects the passed in dict-like object to have a
    'connid' key, whose value in brackets is prepended to the log message.
    """

    def process(self, msg, kwargs):
        return '[plugin: %s] %s' % (self.extra['plugin_name'], msg), kwargs


@deprecated
class EnumFiles:
    """
    DO NOT USE. I will slowly replace all usages of this with rglob_file in utils.
    This class is now replaced with a much simplier implementation:
    rglob_file in hwdebug_common.shell_lib.utils
    For only_last_file mode, just assign file = sorted(returned_list)[-1]

    Scan filesystem looking for viz files
    """

    def __init__(self, file_, only_last_file=False):
        self.file = file_
        self.only_last_file = only_last_file

    def search(self, prefix, path=None):
        """return a list if viz files in path"""
        if path is None:
            path = ""
        self.found = []
        self.find_file_(prefix, path, self.file)
        return self.found

    def find_file_(self, prefix, path, file_):
        """recursive search for files"""
        # recurse through directories looking for *_viz.json
        full = os.path.join(prefix, path, file_)
        list_of_files = glob.glob(
            full
        )  # * means all if need specific format then *.csv
        for x in range(len(list_of_files)):
            if list_of_files[x].startswith(prefix):
                list_of_files[x] = list_of_files[x][len(prefix) + 1 :]
        if len(list_of_files) > 0:
            if self.only_last_file:
                ret = sorted(list_of_files)[-1]
                self.found.append(ret)
            else:
                self.found += list_of_files

        # now look for more
        full = os.path.join(prefix, path, "*")
        list_of_files = glob.glob(full)
        for subdir in list_of_files:
            if os.path.isdir(subdir):
                self.find_file_(prefix, subdir[len(prefix) + 1 :], file_)
        return


class PluginBase(ABC):
    """base plugin"""

    def __init__(self, parent, name, viz_link):
        self.parent = parent
        self.name = name
        self.handler = None
        self.viz_link = viz_link
        self.logger = CustomAdapter(
            logging.getLogger(name), {'plugin_name': name}
        )
        self.INFO = self.logger.debug
        self.DEBUG = self.logger.debug
        self.WARNING = self.logger.warning
        self.ERROR = self.logger.error

    @property
    def workdir(self):
        return self.parent.path_prefix

    @property
    def directory(self):
        return self.parent.directory

    @property
    def stack_dir(self):
        """return the location of the stack directory (artifacts)"""
        return self.parent.stack_dir

    @stack_dir.setter
    def stack_dir(self, value):
        self.parent.stack_dir = value

    @property
    def core_file(self):
        return self.parent.core_file

    @core_file.setter
    def core_file(self, value):
        self.parent.core_file = value

    def enum_files(self, path, pattern, last_file=True):
        return EnumFiles(pattern, last_file).search(self.workdir, path)

    @abstractmethod
    def handle_get(self, handler, url, req, qs):
        """if this function returns true, then the response was sent and we are done"""
        return False

    def gen_file(self, msg_id, msg=None):
        """send message to all other plugins msg_id should be one of MsgId.{xxx} variables"""
        # you must define this in the subclass
        assert False

    def help(self):
        """return doc string for --help"""
        return self.__doc__

    def msg_to_dict(self, msg):
        # MessageToDict is faster than MessageToJSON. For loading default annotations,
        # time taken decreased from 45s to 35s by using MessageToDict instead of MessageToJSON.
        return MessageToDict(
            msg,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )

    def return_msg(self, msg, compress=False):
        self.return_json(
            MessageToJson(
                msg,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            ),
            compress,
        )

    def return_html(self, body):
        self.handler.send_response(HTTPStatus.OK)
        self.handler.send_header("Content-type", "text/html; charset=UTF-8")
        self.handler.send_header("Content-Length", len(body))
        #        self.handler.send_header("Last-Modified",
        #        self.handler.date_time_string(fs.st_mtime))
        self.handler.end_headers()
        self.handler.wfile.write(body)
        return

    def return_validation_error(self, body_json, status=HTTPStatus.BAD_REQUEST):
        if (type(body_json) is dict) or (type(body_json) is list):
            body = json.dumps(body_json).encode('utf-8')
        elif type(body_json) is str:
            wrapper_json = {'message': body_json}
            body = json.dumps(wrapper_json).encode('utf-8')
        else:
            assert False

        self.handler.send_response(status)
        self.handler.send_header("Content-type", "application/json")
        self.handler.send_header("Content-Length", len(body))
        self.handler.end_headers()
        self.handler.wfile.write(body)
        return

    def return_json(self, body_json, compress=False, status=HTTPStatus.OK):
        if (type(body_json) is dict) or (type(body_json) is list):
            body = json.dumps(body_json).encode('utf-8')
        elif type(body_json) is str:
            body = body_json.encode('utf-8')
        elif type(body_json) is bytes:
            body = body_json
        else:
            assert False

        if compress:
            # After trial and error, compression level of 2 offers max performance
            body = gzip.compress(body, 2)

        self.handler.send_response(HTTPStatus.OK)
        self.handler.send_header("Content-type", "application/json")
        self.handler.send_header("Content-Length", len(body))
        if compress:
            self.handler.send_header("Content-Encoding", "gzip")
        self.handler.end_headers()
        self.handler.wfile.write(body)
        return

    def return_file(self, filename, status=HTTPStatus.OK):
        if not Path(filename).exists():
            self.handler.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.handler.send_response(status)
        self.handler.send_header("Content-Type", "application/octet-stream")
        self.handler.send_header("Content-Length", os.path.getsize(filename))
        self.handler.send_header(
            'Content-Disposition', f'attachment; filename="{filename}"'
        )
        self.handler.end_headers()

        with open(filename, 'rb') as file:
            self.handler.wfile.write(file.read())

        return

    def return_runtime_error(
        self, error, status=HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        body_json = {"message": str(error)}
        body_str = json.dumps(body_json)
        body = body_str.encode("utf-8")
        self.handler.send_json_error(body, status)

    def get_das(self):
        return self.parent.get_das()


class VizMetaBase(PluginBase):
    def add_link(self, resp, cat, text, href):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.endpoint = href
        m1.meta_type = VizMetaType.LINK
        resp.data.append(m1)

    def add_button(self, resp, cat, text, href):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.endpoint = href
        m1.meta_type = VizMetaType.BUTTON
        resp.data.append(m1)

    def add_selection(self, resp, cat, text, href):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.endpoint = href
        m1.meta_type = VizMetaType.SELECT_ONE_ONLY
        resp.data.append(m1)

    def add_text(self, resp, cat, text):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.meta_type = VizMetaType.TEXT
        resp.data.append(m1)

    def find_viz(self, prefix, file=None):
        if (file is None) or os.path.isdir(os.path.join(prefix, file)):

            found = EnumFiles("*_viz.json", only_last_file=True).search(
                prefix, file
            )
            if len(found) > 1:
                self.logger.error(
                    "Too many viz folders found: %s", "\n  ".join(found)
                )
                return None
            if len(found) == 1:
                self.logger.debug("found latest file: %s", found[0])
                # remove prefix, if present
                name = found[0]

                file = name
            if len(found) == 0:
                self.logger.error("search for viz folder failed")
                return None

        else:
            name = file

        fullname = os.path.join(prefix, name)

        if not name or not os.path.isfile(fullname):
            self.logger.error("file: %s not found", file)
            return None

        if not name.endswith('_viz.json'):
            return None

        return name


class ERROR_CODES:
    EXCEPTION = "E0001"
    NO_ARTIFACT_DIR = "E0002"
    NO_CORE = "E0003"
    CORE_INVALID_FORMAT = "E0004"


class HTTPJsonError(BaseException):
    def __init__(
        self,
        error_code: ERROR_CODES,
        hint,
        data=None,
        http_status=HTTPStatus.BAD_REQUEST,
    ):
        self.error_code = error_code
        self.hint = hint
        self.data = data
        self.status = http_status
