import dataclasses, json
import numpy as np


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)
