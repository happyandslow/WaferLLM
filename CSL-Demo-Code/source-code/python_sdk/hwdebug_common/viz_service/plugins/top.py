"""
Top plugin

this plugin is designed to find all avertized plugins that are functional
"""
from cerebras.hwdebug_common.viz_service.base import PluginBase


class Plugin(PluginBase):
    ALIAS = 'top'
