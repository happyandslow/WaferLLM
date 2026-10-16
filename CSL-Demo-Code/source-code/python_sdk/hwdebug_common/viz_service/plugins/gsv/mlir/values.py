"""
api/gsv/mlir/values
"""

import math
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.mlir_view_pb2 import MlirValuesRequest, MlirValuesResponse


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/values'

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = MlirValuesRequest()
        query_string_to_request(qs, req)

        if not Path(req.dump_location).exists():
            self.return_validation_error(
                f'Dump location {req.dump_location} does not exist'
            )
            return True

        log_path = Path(req.dump_location) / "logs"
        if not log_path.exists():
            self.return_validation_error(
                f'Tensorboard logs were not found in {req.dump_location}'
            )
            return True

        resp = MlirValuesResponse()

        for event_file in os.listdir(log_path):
            event_acc = EventAccumulator(os.path.join(log_path, event_file))
            event_acc.Reload()
            for scalar_key in event_acc.Tags()['scalars']:
                if not scalar_key.endswith('max'):
                    continue
                kernel_id = int(scalar_key.split('_')[0])
                if kernel_id in req.kernel_ids:
                    mlir_value = MlirValuesResponse.MlirValue()
                    scalar_events = event_acc.Scalars(scalar_key)
                    scalar_event = scalar_events[len(scalar_events) - 1]
                    mlir_value.kernel_id = kernel_id
                    mlir_value.operand_number = int(scalar_key.split('_')[1])
                    mlir_value.max_value = scalar_event.value
                    mlir_value.has_nan = math.isnan(scalar_event.value)
                    resp.values.append(mlir_value)

        dict = self.msg_to_dict(resp)
        self.return_json(dict)

        return True
