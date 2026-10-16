"""
top menu plugin
"""
import json

from cerebras.hwdebug_common.viz_service.base import PluginBase


class Plugin(PluginBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("V2/chart GET: %s", url)
        self.handler = handler

        body = {
            'type': 'bar',
            'data': {
                'labels': [
                    'Red',
                    'Blue',
                    'Yellow',
                    'Green',
                    'Purple',
                    'Orange',
                ],
                'datasets': [
                    {
                        'label': '# of Votes',
                        'data': [12, 19, 3, 5, 2, 3],
                        'backgroundColor': [
                            'rgba(255, 99, 132, 0.2)',
                            'rgba(54, 162, 235, 0.2)',
                            'rgba(255, 206, 86, 0.2)',
                            'rgba(75, 192, 192, 0.2)',
                            'rgba(153, 102, 255, 0.2)',
                            'rgba(255, 159, 64, 0.2)',
                        ],
                        'borderColor': [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(255, 159, 64, 1)',
                        ],
                        'borderWidth': 1,
                    }
                ],
            },
            'options': {'scales': {'y': {'beginAtZero': True}}},
        }
        body = json.dumps(body)
        self.return_json(body.encode('utf-8'))
        return True
