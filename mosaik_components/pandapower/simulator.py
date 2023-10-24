from __future__ import annotations

from dataclasses import dataclass
import sys
from loguru import logger
import mosaik_api_v3
from mosaik_api_v3.types import CreateResult, CreateResultChild, Meta, ModelDescription, OutputData, OutputRequest
import os
import pandapower as pp
import pandapower.networks
import pandas as pd
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# For META, see below. (Non-conventional order do appease the type
# checker.)


class Simulator(mosaik_api_v3.Simulator):
    _sid: str
    """This simulator's ID."""
    _net: pp.pandapowerNet
    """The pandapowerNet for this simulator."""
    bus_auto_elements: pd.DataFrame
    """A dataframe listing the automatically generated loads and sgens
    to support connecting entities from other simulators directly to
    grid nodes.
    
    The index of this dataframe corresponds to the bus index. The two
    columns "load" and "sgen" contain the index of the corresponding
    load and sgen in the load and sgen element tables.
    """

    def __init__(self):
        super().__init__(META)
        self._net = None  # type: ignore  # set in init()
        self.bus_auto_elements = None  # type: ignore  # set in setup_done()

    def init(self, sid: str, time_resolution: float):
        self._sid = sid
        return self.meta

    def create(self, num: int, model: str, **model_params: Any) -> List[CreateResult]:
        if model == "Grid":
            if num != 1:
                raise ValueError("must create exactly one Grid entity")
            return [self.create_grid(**model_params)]

        if not self._net:
            raise ValueError(f"cannot create {model} entities before creating Grid")
        
        if model == "ControlledGen":
            return [
                self.create_controlled_gen(**model_params)
                for _ in range(num)
            ]
        else:
            raise ValueError(f"no entities for the model {model} can be created")

    def create_grid(self, source: str, params: Dict[str, Any]={}) -> CreateResult:
        if self._net:
            raise ValueError("Grid was already created")

        self._net, self._profiles = load_grid(source, params)

        child_entities: List[CreateResultChild] = []
        for child_model, info in MODEL_TO_ELEMENT_INFO.items():
            for elem_tuple in self._net[info.elem].itertuples():
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
            "rel": [],
        }
    
    def create_controlled_gen(self, bus: int) -> CreateResult:
        idx = pp.create_gen(self._net, bus, p_mw=0.0)
        return {
            "type": "ControlledGen",
            "eid": f"ControlledGen-${idx}",
            "children": [],
            "rel": [],
        }

    def setup_done(self):
        load_indices = pp.create_loads(self._net, self._net.bus.index, 0.0)
        sgen_indices = pp.create_sgens(self._net, self._net.bus.index, 0.0)
        self.bus_auto_elements = pd.DataFrame(
            {
                "load": load_indices,
                "sgen": sgen_indices,
            },
            index=self._net.bus.index,
        )

    def get_model_and_idx(self, eid: str) -> Tuple[str, int]:
        # TODO: Maybe add a benchmark whether caching this in a dict is
        # faster
        model, idx_str = eid.split("-")
        return (model, int(idx_str))

    def step(self, time, inputs, max_advance):
        if self._profiles:
            apply_profiles(self._net, self._profiles, time)
        for eid, data in inputs.items():
            model, idx = self.get_model_and_idx(eid)
            info = MODEL_TO_ELEMENT_INFO[model]
            for attr, values in data.items():
                attr_info = info.in_attrs[attr]
                self._net[attr_info.target_elem or info.elem].at[
                    attr_info.idx_fn(idx, self), attr_info.column
                ] = attr_info.aggregator(values.values())

        pp.runpp(self._net)
        return time + 1

    def get_data(self, outputs: OutputRequest) -> OutputData:
        return {
            eid: self.get_entity_attrs(eid, attrs)
            for eid, attrs in outputs.items()
        }

    def get_entity_attrs(self, eid: str, attrs: List[str]) -> Dict[str, Any]:
        model, idx = self.get_model_and_idx(eid)
        info = MODEL_TO_ELEMENT_INFO[model]
        elem_table = self._net[f"res_{info.elem}"]
        return {attr: elem_table.at[idx, info.out_attrs[attr]] for attr in attrs}


@dataclass
class InAttrInfo:
    """Specificaction of an input attribute of a model.
    """
    column: str
    """The name of the column in the target element's dataframe
    corresponding to this attribute.
    """
    target_elem: Optional[str] = None
    """The name of the pandapower element to which this attribute's
    inputs are written. (This might not be the element type
    corresponding to the model to support connecting loads and sgens
    directly to the buses.)
    If None, use the element corresponding to the model.
    """
    idx_fn: Callable[[int, Simulator], int] = lambda idx, sim: idx
    """A function to transform the entity ID's index part into the
    index for the target_df.
    """
    aggregator: Callable[[Iterable[Any]], Any] = sum
    """The function that is used for aggregation if several values are
    given for this attribute.
    """


@dataclass
class ModelElementInfo:
    elem: str
    """The name of the pandapower element corresponding to this model.
    """
    connected_buses: List[str]
    """The names of the columns specifying the buses to which this
    element is connected.
    """
    in_attrs: Dict[str, InAttrInfo]
    """Mapping each input attr to the corresponding column in the
    element's dataframe and an aggregation function.
    """
    out_attrs: Dict[str, str]
    """Mapping each output attr to the corresponding column in the
    element's result dataframe.
    """


MODEL_TO_ELEMENT_INFO = {
    "Bus": ModelElementInfo(
        elem="bus",
        connected_buses=[],
        in_attrs={
            "P_gen[MW]": InAttrInfo(
                column="p_mw",
                target_elem="sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "sgen"],
            ),
            "P_load[MW]": InAttrInfo(
                column="p_mw",
                target_elem="load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "load"],
            ),
            "Q_gen[MVar]": InAttrInfo(
                column="q_mvar",
                target_elem="sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "sgen"],
            ),
            "Q_load[MVar]": InAttrInfo(
                column="q_mvar",
                target_elem="load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "load"],
            ),
        },
        out_attrs={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mw",
            "Vm[pu]": "vm_pu",
            "Va[deg]": "va_deg",
        },
    ),
    "Load": ModelElementInfo(
        elem="load",
        connected_buses=["bus"],
        in_attrs={},
        out_attrs={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
    ),
    "StaticGen": ModelElementInfo(
        elem="sgen",
        connected_buses=["bus"],
        in_attrs={},
        out_attrs={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
    ),
    "Gen": ModelElementInfo(
        elem="gen",
        connected_buses=["bus"],
        in_attrs={},
        out_attrs={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
            "Va[deg]": "va_degree",
            "Vm[pu]": "vm_pu",
        },
    ),
    "Line": ModelElementInfo(
        elem="line",
        connected_buses=["from_bus", "to_bus"],
        in_attrs={},
        out_attrs={
            "I[kA]": "i_ka",
            "loading[%]": "loading_percent",
        },
    ),
}


# Generate mosaik model descriptions out of the MODEL_TO_ELEMENT_INFO
ELEM_META_MODELS: Dict[str, ModelDescription] = {
    model: {
        "public": False,
        "params": [],
        "attrs": list(info.in_attrs.keys()) + list(info.out_attrs.keys()),
        "any_inputs": False,
        "persistent": [],
        "trigger": [],
    }
    for model, info in MODEL_TO_ELEMENT_INFO.items()
}


META: Meta = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "Grid": {
            "public": True,
            "params": ["source", "params"],
            "attrs": [],
            "any_inputs": False,
            "persistent": [],
            "trigger": [],
        },
        "ControlledGen": {
            "public": True,
            "params": ["bus"],
            "attrs": [],
            "any_inputs": False,
            "persistent": [],
            "trigger": [],
        },
        **ELEM_META_MODELS,
    },
    "extra_methods": [],
}


def apply_profiles(net: pp.pandapowerNet, profiles: Any, step: int):
    """Apply element profiles for the given step to the grid.

    :param profiles: profiles for elements in the format returned by
        simbench's ``get_absolute_values`` function.
    :param step: the time step to apply
    """
    for (elm, param), series in profiles.items():
        net[elm].loc[:, param].update(series.loc[step])  # type: ignore


def load_grid(source: Any, params: Dict[str, Any]) -> Tuple[pp.pandapowerNet, Any]:
    """Load a grid and the associated element profiles (if any).

    :param source: This can be:

        - a pandapower net object
        - the file name of a grid in JSON or Excel format
        - the name of a function in pandapower.networks
        - a simbench ID (if simbench is installed)
    
    :param params: The parameters to pass to the pandapower.networks
        function. (For the other source types, this does nothing.)

    :return: a tuple consisting of a :class:`pandapowerNet` and "element
        profiles" in the form that is returned by simbench's
        get_absolute_values function (or ``None`` if the loaded grid
        is not a simbench grid).
    """
    # Accept a pandapower grid
    if isinstance(source, pp.pandapowerNet):
        return (source, None)

    # Accept .json and .xlsx files
    try:
        _, ext = os.path.splitext(source)
        if ext == ".json":
            return (pp.from_json(source), None)
        elif ext == ".xlsx":
            return (pp.from_excel(source), None)
        else:
            # File ending does not indicate accepted file type.
            # Try the next option
            pass
    except UserWarning:
        # This exception is thrown by pandapower if file does not exist.
        # Try the next option
        pass

    # Try to load a default network
    try:
        return (getattr(pandapower.networks, source)(**params), None)
    except AttributeError:
        # The supplied network name is not the name of a function from
        # pandapower.networks.
        # Try the next option
        pass

    # Try simbench
    try:
        import simbench as sb

        net = sb.get_simbench_net(source)
        profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        return (net, profiles)
    except (ImportError, ValueError):
        raise ValueError(
            f"Could not load requested grid '{source}'. Did you spell the name "
            "correctly? If you're trying to load a simbench grid: Is simbench "
            "installed?"
        )
