from dataclasses import dataclass
import mosaik_api
import os
import pandapower as pp
import pandapower.timeseries.run_time_series as pprts
from typing import List

META = {
    "version": "3.0",
    "type": "time-based",
    "models": {
        "Grid": {
            "public": True,
            "params": ["grid", "params"],
        },
        "Bus": {
            "public": False,
            "params": [],
            "attrs": [
                "P_MW_gen",
                "P_MW_load",
            ],
        },
        "Load": {
            "public": False,
            "params": [],
            "attrs": [
                "P_MW",
            ],
        },
        "ControlledGen": {"public": True, "params": ["bus"], "attrs": {}},
    },
}


@dataclass
class ModelElementInfo:
    elem_name: str
    connected_buses: List[str]


MODEL_ELEM_MAP = {
    "Bus": ModelElementInfo("bus", []),
    "Load": ModelElementInfo("load", ["bus"]),
}


class Simulator(mosaik_api.Simulator):
    _net: pp.pandapowerNet

    def __init__(self):
        super().__init__(META)
        self._net = None  # type: ignore  # set in init()

    def init(self, sid, time_resolution):
        self._sid = sid
        return self.meta

    def create(self, num, model, **model_params):
        if model == "Grid":
            if num != 1:
                raise ValueError("must create exactly one Grid entity")
            return [self.create_grid(**model_params)]

        if not self._net:
            raise ValueError(f"cannot create {model} entities before creating Grid")

    def create_grid(self, grid, params={}):
        if self._net:
            raise ValueError("Grid was already created")

        self._net, self._profiles = load_grid(grid, params)

        child_entities = []
        for child_model, info in MODEL_ELEM_MAP.items():
            for elem_tuple in self._net[info.elem_name].itertuples():
                child_entities.append(
                    {
                        "type": child_model,
                        "eid": f"{child_model}-{elem_tuple.Index}",
                        "rel": [
                            f"Bus-{getattr(elem_tuple, bus)}"
                            for bus in info.connected_buses
                        ],
                    }
                )

        return {
            "eid": "Grid",
            "type": "Grid",
            "children": child_entities,
        }

    def setup_done(self):
        pass

    def step(self, time, inputs, max_advance):
        if self._profiles:
            apply_profiles(self._net, self._profiles, time)
        pp.runpp(self._net)
        return time + 1

    def get_data(self, output_request):
        return {
            entity: self.get_entity_attrs(entity, attrs)
            for entity, attrs in output_request.items()
        }

    def get_entity_attrs(self, entity, attrs):
        model, idx_str = entity.split("-")
        idx = int(idx_str)
        elem_table = self._net[MODEL_ELEM_MAP[model].elem_name]
        return {attr: elem_table[idx, attr] for attr in attrs}


def apply_profiles(net, profiles, step):
    """Apply element profiles for the given step to the grid.

    :param profiles: profiles for elements in the format returned by
        simbench's ``get_absolute_values`` function.
    :param step: the time step to apply
    """
    for (elm, param), series in profiles.items():
        net[elm].loc[:, param].update(series.loc[step])


def load_grid(grid, params):
    # Accept a pandapower grid
    if isinstance(grid, pp.pandapowerNet):
        return (grid, None)

    # Accept .json and .excel files
    ext = os.path.splitext(grid)
    if ext == ".json":
        return (pp.load_json(grid), None)
    if ext == ".xlsx":
        return (pp.load_excel(grid), None)

    # Try to load a default network
    try:
        return (getattr(pp.networks, grid)(**params), None)
    except AttributeError:
        pass

    try:
        import simbench as sb

        net = sb.get_simbench_net(grid)
        profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        return (net, profiles)
    except ImportError:
        raise ValueError(
            "Could not load requested grid. Did you spell the name correctly? "
            "Is simbench installed (if you're trying to load a simbench grid)?"
        )
