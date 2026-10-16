#!/usr/bin/env python
"""@package Manifest
Simple wrapper on C++ manifest that allows serializing python classes
"""

import os

import logging
import google.protobuf.json_format as json_format
from google.protobuf.message import Message
from cerebras.common.manifest.manifestpy import Manifest as  CppManifest
import cerebras.common.manifest.manifestpy as manifestpy



class Manifest(CppManifest):
    """
    to derive from this class to load the objects:
    class myclass(manifest):
        def  __init__(fname, value):
            super(fname, value)
        def __init__():
            super("manifest.yml", "myclass")

    or you can just "use" the manifest
    class myclass()
        __init__():
            manifest("manifest.yml", "myclass", self)
   """

    def __init__(self, fname, preload=None):
        super().__init__(fname)
        self.fname = fname
        self.basedir = os.path.dirname(fname)
        # for convenience, copy DIRLOC_ attributes to this class
        # these are defined in hwdebug/tools/manifest/manifest_dir.hh
        self.DIRLOC = {}
        for key, val in dict.items(manifestpy.__dict__):
            if key.startswith('DIRLOC_'):
                setattr(self, key, val)
                self.DIRLOC[key[7:]] = val
        self._cpp_manifest = CppManifest(fname)
        if preload is not None:
            self.load(preload)

    def setAbsPath(self, path):
        '''
        Convert absolute paths to relative paths.  This will not use .. for
        paths before specified path, therefore the path specified should be
        at the top of the tree only base (builtin) types (classes) are supported
        '''
        self._cpp_manifest.setValue(self.DIRLOC_ABSPATH, path)

    def protobufLoad(self, location, proto_cls):
        """
        Parses the JSON dict, returns it as a protobuf object of the given type
        Raises ValueError if the given dict is invalid

        :param string location:
        :param class proto_cls:
        """
        if not issubclass(proto_cls, Message):
            raise ValueError(
                f"proto_cls must be a protobuf message type: "
                f"{type(proto_cls).__name__}"
            )
        try:
            json_dict = self._cpp_manifest.getValue(location)
            proto_msg = json_format.ParseDict(
                json_dict, proto_cls(), ignore_unknown_fields=False
            )
            return proto_msg
        except json_format.ParseError as exc:
            raise ValueError("Invalid JSON dict") from exc

    def protobufStore(self, location, proto_obj):
        """
        Serializes the given protobuf object to manifest

        :param google.protobuf.message.Message proto_msg:
        """
        if not isinstance(proto_obj, Message):
            raise ValueError("proto_msg must be a protobuf message")

        val = json_format.MessageToDict(
            proto_obj,
            including_default_value_fields=True,
            preserving_proto_field_name=True
        )

        self._cpp_manifest.setValue(location, val)

    def store(self, name, obj=None, relative="", filter_=None):
        """ store object manfiest
        Args:
            relative:  convert absolute paths to relative paths
            filter_: dont store keys defined in this array
            obj: object to store
            name: location to store the data, dot separated path
        """
        if obj is None:
            obj = self
            logging.info("using self dictionary")
        # try to use serializer from caller's object.  If that does not
        # work/exist, then use the object dictionary
        try:
            save = obj.manifest_serialize()
            logging.info("using manifest_serialize")
        #pylint: disable=broad-except
        except Exception:
            save = self.manifest_serialize(obj)
            logging.info("using passed in object")

        logging.info("saving: " + str(save))

        if hasattr(save, '__dict__') is True:
            save = save.__dict__

        if isinstance(save, dict):
            self._cpp_manifest.setValue(name, None)
            # handle dictionary, and potential sub-objects here
            for key, value in save.items():
                matching = True
                if filter_ is not None:
                    matching = [s for s in filter_ if key in s]
                if matching:
                    logging.info("saving dict: " + key + ": " + str(value))
                    if isinstance(
                        value,
                        (list, int, float, complex, bool, str, bytes, type(None))
                    ):
                        self._cpp_manifest.setValue(name + "." + key, value)
                    else:
                        # recurse, saving objects
                        self.store(name + "." + key, value)

        else:
            if isinstance(
                save,
                (list, int, float, complex, bool, str, bytes, type(None))
            ):
                self._cpp_manifest.setValue(name, save)
            else:
                logging.warning("ignoring attribute: %s", str(save))

        if relative:
            self.replacepath(self._cpp_manifest, relative)

    def relative(self, path):
        """ return relative path based on absPath in manifest """
        return self.data.path_relative(path)

    def manifest_serialize(self, obj=None):
        """ return json file data from object """
        if obj is None:
            obj = self
        logging.info("serializing self...  " + str(obj))

        # if it is a class with a __dict__ then we can serialize the class
        if hasattr(obj, '__dict__') is True:
            obj = obj.__dict__
        else:
            # these are the only objs we can serialize...
            if not isinstance(
                obj,
                (list, int, float, complex, bool, str, bytes, type(None))
            ):
                obj = None

        return obj

    def get_value(self, name):
        """ get a value from the manifest.  Manifest is multi-process safe, however
            read-modify-write of the same value/path is not atomic.  if this is required,
            an update_value() method needs to be added.  read-modify-write of non-same paths
            is multi-process safe.
        """
        ret = None
        try:
            ret = self._cpp_manifest.getValue(name)
        except Exception as e: # pylint: disable=broad-except
            logging.debug("exception %s ignored reading %s, NULL returned, possibly ok", e, name)

        return ret

    def set_value(self, name, value):
        """ multi-process safe (different names only) replace value at location """
        ret = self._cpp_manifest.setValue(name, value)
        return ret

    def get_file_rel(self, name):
        """ get file name relative to ABSPATH set in manifest """
        ret = None
        try:
            ret = self._cpp_manifest.getValue(name)
            rel = self._cpp_manifest.getValue(self.DIRLOC_ABSPATH)
            ret = os.path.relpath(ret, rel)

        except Exception as e: # pylint: disable=broad-except
            logging.debug("exception %s ignored reading %s, NULL returned, possibly ok", e, name)

        return ret

    def load(self, name, obj=None):
        """ load an object into location """
        if obj is None:
            obj = self
        data = self._cpp_manifest.getValue(name)
        for key, value in dict.items(data):
            setattr(obj, key, value)

if __name__ == "__main__":
    # tests code if you don't import
    import yaml

    class test:
        """ test class """
        def __init__(self):
            self.a = "this is a"
            self.b = 123
            self.c = None
            self.d = None

        def add(self):
            """ test code"""
            self.d = "path"

        def manifest_serialize(self):
            """ test code"""
            return self.__dict__

    class test2(Manifest):
        """ test class that subclasses Manifest """
        def __init__(self, name):
            super().__init__("test.yml")
            super().load(name, self)

    # a bit of testing/use case examples
    y = test()
    y.c = "4"
    y.add()
    print(yaml.dump(y.manifest_serialize()))

    z = test()
    z.a = "test"
    z.b = 456
    m = Manifest("test.yml")
    m.store("y", y)
    m.store("z", z)

    n = test()
    m.load("z", n)
    print(n.__dict__)

    n1 = test2("z")
    print(n1.__dict__)
