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
                entity_inputs = inputs.pop(entity)
                for attr, values in expected[time].items():
                    expected_values = sorted(values)
                    received_values = sorted(entity_inputs.pop(attr).values())
                    for ev, rv in zip(expected_values, received_values, strict=True):
                        rel_err = abs((ev - rv) / ev)
                        assert rel_err < 0.005, (
                            f"at time {time} on port {entity}.{attr}, got {rv} instead "
                            f"of {ev} for (relative error of {rel_err})"
                        )
                assert entity_inputs == {}, (
                    f"remaining entity inputs for entity {entity} at time {time}: "
                    f"{entity_inputs}"
                )
        assert inputs == {}, f"remaining inputs at time {time}: {inputs}"
