import os
import threading
from pydoc import safeimport
from pathlib import Path
from typing import Dict

"""
Plugin manager module.  Responsible for caching plugins and providing links to other plugins

also caches a das context and directory of shared data
"""


class PluginManager:
    """ this class is responsible for managing loading and caching of plugins
        for csviz. """

    def __init__(self, logger, path_prefix, viz_link):
        self.directory = {}
        self.compile_directory = {}
        self.core_directory = {}
        self.run_directory: Dict[Path, Trace] = {}
        self.plugins = {}
        self.imports = {}
        self.plan_meta = None
        self.logger = logger
        self.path_prefix = path_prefix
        self.run_summaries = None
        self.current_summary = None
        self.current_stack_dir = None
        self.current_core_file = None
        self.viz_link = viz_link
        self.lock_ = threading.Lock()

    @property
    def stack_dir(self):
        return self.current_stack_dir

    @stack_dir.setter
    def stack_dir(self, value):
        path = os.path.join(self.path_prefix, value)
        self.current_stack_dir = value
        self.logger.debug("Setting plan meta to: %s", path)
        self.plan_meta = os.path.join(value, "plan.meta")
        plan_path = os.path.join(self.path_prefix, self.plan_meta)

    @property
    def lock(self):
        return self.lock_

    @property
    def run_summary(self):
        return self.current_summary

    @run_summary.setter
    def run_summary(self, value):
        if value:
            path = os.path.join(self.path_prefix, value)
            with open(path, 'r') as f:
                self.current_summary = json.load(f)
        else:
            self.current_summary = None

    @property
    def core_file(self):
        return self.current_core_file

    @core_file.setter
    def core_file(self, value):
        path = os.path.join(self.path_prefix, value)
        self.current_core_file = path
        self.logger.debug("Setting core file to: %s", path)

    def locate(self, path, forceload=0):
        """Locate an object by name or dotted path, importing as necessary."""
        parts = [part for part in path.split('.') if part]
        module, n = None, 0
        while n < len(parts):
            nextmodule = safeimport('.'.join(parts[: n + 1]), forceload)
            if nextmodule:
                module, n = nextmodule, n + 1
            else:
                break
        assert module
        _object = module
        for part in parts[n:]:
            try:
                _object = getattr(_object, part)
            except AttributeError:
                return None
        return _object

    def activate(self, path):
        # remove leading slash and api prefix
        if path.startswith('/'):
            path = path[1:]
        if path.startswith('api/'):
            path = path[4:]

        # split and remove api (first entry)
        p = path.split('/')
        svc = '.'.join(p)

        # must be a python plugin, so load that
        return self.load_plugin(svc), svc

    def load_plugin(self, plug_name):
        if plug_name.startswith('cerebras'):
            return self.load_plugin2(plug_name, plug_name)
        else:
            prefix = 'cerebras.hwdebug_common.viz_service.plugins'
            # find the bottom-most object that has a plugin
            parts = [part for part in plug_name.split('.') if part]
            n = len(parts)
            object_ = None
            while n > 0 and not object_:
                mod = '.'.join(parts[:n])
                object_ = self.load_plugin2(plug_name, prefix + '.' + mod)
                # also create real plugin link, if not present
                if object_ and mod not in self.plugins:
                    self.plugins[mod] = object_
                n = n - 1
        return object_

    def get_plugin(self, plug_name):
        if plug_name not in self.plugins:
            self.load_plugin(plug_name)

        return self.plugins[plug_name]

    def load_plugin2(self, plug_name, module_path):
        """ recursively load plugins """

        if plug_name not in self.plugins:
            self.logger.debug("Locate: %s", module_path)

            try:
                module = self.locate(module_path)
                self.logger.debug(
                    "Module: %s: name: %s", getattr(module, "Plugin"), plug_name
                )
                x = getattr(module, 'Plugin')
                alias = x.ALIAS
                self.plugins[plug_name] = x
                if alias:
                    self.plugins[alias] = x
            except AttributeError:
                return None

        return self.plugins[plug_name](self, plug_name, self.viz_link)

    def get_das(self):
        path = os.path.join(self.path_prefix, self.plan_meta)
