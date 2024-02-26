from copy import deepcopy
import mosaik_api_v3


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
    }
}


class AssertSimulator(mosaik_api_v3.Simulator):
    def __init__(self) -> None:
        super().__init__(META)

    def init(self, sid, time_resolution):
        self.entities = {}
        return self.meta

    def create(self, num, model, expected, name="Entity"):
        new_entities = []
        for i in range(num):
            while name in self.entities:
                name += "'"
            self.entities[name] = deepcopy(expected)
            new_entities.append({
                "eid": name,
                "type": "Entity",
            })
        return new_entities

    def step(self, time, inputs, max_advance):
        for entity, expected in self.entities.items():
            if time in expected:
                expected_now = {
                    attr: sorted(values)
                    for attr, values in expected[time].items()
                }
                entity_inputs = {
                    attr: sorted(values.values())
                    for attr, values in inputs[entity].items()
                }
                for k in range(len(expected_now['P[MW]'])):
                    rel_err = abs((expected_now['P[MW]'][k] - entity_inputs['P[MW]'][k])/expected_now['P[MW]'][k])
                    assert rel_err < 0.005, \
                        f"got {entity_inputs['P[MW]'][k]} at P[MW] instead of {expected_now['P[MW]'][k]} for entity {k} with relativ error of {rel_err} " \
                        f"{entity} at time {time}"                 
                del inputs[entity]
        assert inputs == {}, f"remaining inputs at time {time}"
