import mosaik_api_v3
from typing_extensions import override

META = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "Const": {
            "public": True,
            "params": ["value"],
            "attrs": ["value"],
        }
    },
}


class ConstSimulator(mosaik_api_v3.Simulator):
    def __init__(self) -> None:
        super().__init__(META)

    @override
    def init(self, sid, time_resolution, step_size=1):
        self.entities = {}
        self.step_size = step_size
        return self.meta

    @override
    def create(self, num, model, value):
        new_entities = []
        for n in range(len(self.entities), len(self.entities) + num):
            eid = f"Constant-{n}"
            self.entities[eid] = value
            new_entities.append(
                {
                    "eid": eid,
                    "type": "Const",
                }
            )
        return new_entities

    @override
    def step(self, time, inputs, max_advance):
        return time + self.step_size

    @override
    def get_data(self, request):
        return {eid: {"value": self.entities[eid]} for eid in request}
