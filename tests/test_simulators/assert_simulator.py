from copy import deepcopy

import mosaik_api_v3
from typing_extensions import override

META = {
    "version": "3.0",
    "type": "event-based",
    "models": {
        "Entity": {
            "public": True,
            "params": ["name", "expected"],
            "attrs": [],
            "any_inputs": True,
        },
    },
}


class AssertSimulator(mosaik_api_v3.Simulator):
    def __init__(self) -> None:
        super().__init__(META)

    @override
    def init(self, sid, time_resolution):
        self.entities = {}
        return self.meta

    @override
    def create(self, num, model, expected, name="Entity"):
        new_entities = []
        for _ in range(num):
            while name in self.entities:
                name += "'"
            self.entities[name] = deepcopy(expected)
            new_entities.append(
                {
                    "eid": name,
                    "type": "Entity",
                }
            )
        return new_entities

    @override
    def step(self, time, inputs, max_advance):
        for entity, expected in self.entities.items():
            if time in expected:
                expected_now = {
                    attr: sorted(values) for attr, values in expected[time].items()
                }
                entity_inputs = {
                    attr: sorted(values.values())
                    for attr, values in inputs[entity].items()
                }
                assert expected_now == entity_inputs, (
                    f"got {entity_inputs} instead of {expected_now} for entity "
                    f"{entity} at time {time}"
                )
                del inputs[entity]
        assert inputs == {}, f"remaining inputs at time {time}"
