from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import mosaik_api_v3
import pandas as pd
from typing_extensions import override

import pandapower as pp
import pandapower.networks

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mosaik_api_v3.types import (
        CreateResult,
        EntityId,
        InputData,
        Meta,
        ModelDescription,
        OutputData,
        OutputRequest,
        Time,
    )

# For META, see below. (Non-conventional order do appease the type
# checker.)


class Simulator(mosaik_api_v3.Simulator):
    _sid: str
    """This simulator's ID."""
    _step_size: int | None
    """The step size for this simulator. If ``None``, the simulator
    is running in event-based mode, instead.
    """
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

    _extra_info: dict[EntityId, Any]
    """Storage of the entity's extra_info for use in the
    `get_extra_info` extra_method. This should be removed once
    mosaik 3.3 (or later) is available more widely.
    """

    def __init__(self):
        super().__init__(META)
        self._net = None  # type: ignore  # set in init()
        self.bus_auto_elements = None  # type: ignore  # set in setup_done()
        self._extra_info = {}

    @override
    def init(self, sid: str, time_resolution: float, step_size: int | None = 900):
        self._sid = sid
        if not step_size:
            self.meta["type"] = "event-based"
        self._step_size = step_size
        return self.meta

    @override
    def create(self, num: int, model: str, **model_params: Any) -> list[CreateResult]:
        if model == "Grid":
            if num != 1:
                raise ValueError("must create exactly one Grid entity")
            return [self.create_grid(**model_params)]

        if not self._net:
            raise ValueError(f"cannot create {model} entities before creating Grid")

        if model == "ControlledGen":
            return [self.create_controlled_gen(**model_params) for _ in range(num)]

        raise ValueError(f"no entities for the model {model} can be created")

    def create_grid(self, **params: Any) -> CreateResult:
        if self._net:
            raise ValueError("Grid was already created")

        self._net, self._profiles = load_grid(params)

        child_entities: list[CreateResult] = []
        for child_model, spec in MODEL_TO_ELEMENT_SPECS.items():
            for elem_tuple in self._net[spec.elem].itertuples():
                eid = f"{child_model}-{elem_tuple.Index}"
                extra_info = {
                    "name": elem_tuple.name,
                    "index": elem_tuple.Index,
                    **spec.get_extra_info(elem_tuple, self._net),
                }
                child_entities.append(
                    {
                        "type": child_model,
                        "eid": eid,
                        "rel": [
                            f"Bus-{getattr(elem_tuple, bus)}"
                            for bus in spec.connected_buses
                        ],
                        "extra_info": extra_info,
                    }
                )
                self._extra_info[eid] = extra_info

        return {
            "eid": "Grid",
            "type": "Grid",
            "children": child_entities,
            "rel": [],
        }

    def get_extra_info(self) -> dict[EntityId, Any]:
        return self._extra_info

    def get_net(self) -> pp.pandapowerNet:
        return self._net

    def disable_elements(self, elements: list[str]) -> None:
        for eid in elements:
            model, idx = self.get_model_and_idx(eid)
            elem_spec = MODEL_TO_ELEMENT_SPECS[model]
            if not elem_spec.can_switch_off:
                raise ValueError(f"{model} elements cannot be disabled")
            self._net[elem_spec.elem].loc[idx, "in_service"] = False

    def enable_elements(self, elements: list[str]) -> None:
        for eid in elements:
            model, idx = self.get_model_and_idx(eid)
            elem_spec = MODEL_TO_ELEMENT_SPECS[model]
            if not elem_spec.can_switch_off:
                raise ValueError(f"{model} elements cannot be enabled")
            self._net[elem_spec.elem].loc[idx, "in_service"] = True

    def create_controlled_gen(self, bus: int) -> CreateResult:
        idx = pp.create_gen(self._net, bus, p_mw=0.0)
        return {
            "type": "ControlledGen",
            "eid": f"ControlledGen-{idx}",
            "children": [],
            "rel": [f"Bus-{bus}"],
        }

    @override
    def setup_done(self):
        # Create "secret" loads and sgens that are used when the user
        # provides real and reactive power directly to grid nodes.
        load_indices = pp.create_loads(self._net, self._net.bus.index, 0.0)
        sgen_indices = pp.create_sgens(self._net, self._net.bus.index, 0.0)
        self.bus_auto_elements = pd.DataFrame(
            {
                "load": load_indices,
                "sgen": sgen_indices,
            },
            index=self._net.bus.index,
        )

    def get_model_and_idx(self, eid: str) -> tuple[str, int]:
        # TODO: Maybe add a benchmark whether caching this in a dict is
        # faster
        model, idx_str = eid.split("-")
        return (model, int(idx_str))

    @override
    def step(self, time: Time, inputs: InputData, max_advance: Time) -> Time | None:
        if self._profiles:
            # TODO: Division by 900 here assumes a time_resolution of 1.
            apply_profiles(self._net, self._profiles, time // 900)
        for eid, data in inputs.items():
            model, idx = self.get_model_and_idx(eid)
            spec = MODEL_TO_ELEMENT_SPECS[model]
            for attr, values in data.items():
                attr_spec = spec.input_attr_specs[attr]
                self._net[attr_spec.target_elem or spec.elem].at[
                    attr_spec.idx_fn(idx, self), attr_spec.column
                ] = attr_spec.aggregator(values.values())

        pp.runpp(self._net)
        if self._step_size:
            return time + self._step_size

        return None

    @override
    def get_data(self, outputs: OutputRequest) -> OutputData:
        return {eid: self.get_entity_data(eid, attrs) for eid, attrs in outputs.items()}

    def get_entity_data(self, eid: str, attrs: list[str]) -> dict[str, Any]:
        model, idx = self.get_model_and_idx(eid)
        info = MODEL_TO_ELEMENT_SPECS[model]
        elem_table = self._net[f"res_{info.elem}"]
        return {
            attr: elem_table.at[idx, info.out_attr_to_column[attr]] for attr in attrs
        }


@dataclass
class InputAttrSpec:
    """Specificaction of an input attribute of a model."""

    column: str
    """The name of the column in the target element's dataframe
    corresponding to this attribute.
    """
    target_elem: str | None = None
    """The name of the pandapower element to which this attribute's
    inputs are written. (This might not be the element type
    corresponding to the model to support connecting loads and sgens
    directly to the buses.)
    If ``None``, use the element corresponding to the model.
    """
    idx_fn: Callable[[int, Simulator], int] = lambda idx, _sim: idx  # noqa: E731
    """A function to transform the entity ID's index part into the
    index for the target_df.
    """
    aggregator: Callable[[Iterable[Any]], Any] = sum
    """The function that is used for aggregation if several values are
    given for this attribute.
    """


@dataclass
class ModelToElementSpec:
    """Specification of the pandapower element that is represented by
    a (mosaik) model of this simulator.
    """

    elem: str
    """The name of the pandapower element corresponding to this model.
    """
    connected_buses: list[str]
    """The names of the columns specifying the buses to which this
    element is connected.
    """
    input_attr_specs: dict[str, InputAttrSpec]
    """Mapping each input attr to the corresponding column in the
    element's dataframe and an aggregation function.
    """
    out_attr_to_column: dict[str, str]
    """Mapping each output attr to the corresponding column in the
    element's result dataframe.
    """
    createable: bool = False
    """Whether this element can be created by the user."""
    params: list[str] = field(default_factory=list)
    """The mosaik params that may be given when creating this element.
    (Only sensible if ``createable=True``.)
    """
    get_extra_info: Callable[[Any, pp.pandapowerNet], dict[str, Any]] = (  # noqa: E731
        lambda _net, _idx: {}
    )
    """Function returning the extra info for this type of element given
    the net and the element's index.
    """
    can_switch_off: bool = False
    """Whether elements of this type may be switched off (and on) using
    the *disable_element* (*enable_element*) extra methods.
    """


MODEL_TO_ELEMENT_SPECS = {
    "Bus": ModelToElementSpec(
        elem="bus",
        connected_buses=[],
        input_attr_specs={
            "P_gen[MW]": InputAttrSpec(
                column="p_mw",
                target_elem="sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "sgen"],
            ),
            "P_load[MW]": InputAttrSpec(
                column="p_mw",
                target_elem="load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "load"],
            ),
            "Q_gen[MVar]": InputAttrSpec(
                column="q_mvar",
                target_elem="sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "sgen"],
            ),
            "Q_load[MVar]": InputAttrSpec(
                column="q_mvar",
                target_elem="load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "load"],
            ),
        },
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
            "Vm[pu]": "vm_pu",
            "Va[deg]": "va_degree",
        },
        get_extra_info=lambda elem_tuple, _net: {
            "nominal voltage [kV]": elem_tuple.vn_kv,
        },
    ),
    "Load": ModelToElementSpec(
        elem="load",
        connected_buses=["bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
        get_extra_info=lambda elem, _net: {
            "bus": elem.bus,
            **({"profile": elem.profile} if "profile" in elem._fields else {}),
        },
        can_switch_off=True,
    ),
    "StaticGen": ModelToElementSpec(
        elem="sgen",
        connected_buses=["bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
        get_extra_info=lambda elem, _net: {
            "bus": elem.bus,
            **({"profile": elem.profile} if "profile" in elem._fields else {}),
        },
        can_switch_off=True,
    ),
    "Gen": ModelToElementSpec(
        elem="gen",
        connected_buses=["bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
            "Va[deg]": "va_degree",
            "Vm[pu]": "vm_pu",
        },
        get_extra_info=lambda elem, _net: {
            "bus": elem.bus,
            **({"profile": elem.profile} if "profile" in elem._fields else {}),
        },
        can_switch_off=True,
    ),
    "ExternalGrid": ModelToElementSpec(
        elem="ext_grid",
        connected_buses=["bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
    ),
    "ControlledGen": ModelToElementSpec(
        elem="gen",
        connected_buses=["bus"],
        input_attr_specs={
            "P[MW]": InputAttrSpec(
                column="p_mw",
            )
        },
        out_attr_to_column={},
        createable=True,
        params=["bus"],
    ),
    "Line": ModelToElementSpec(
        elem="line",
        connected_buses=["from_bus", "to_bus"],
        input_attr_specs={},
        out_attr_to_column={
            "I[kA]": "i_ka",
            "loading[%]": "loading_percent",
        },
    ),
}


# Generate mosaik model descriptions out of the MODEL_TO_ELEMENT_INFO
ELEM_META_MODELS: dict[str, ModelDescription] = {
    model: {
        "public": info.createable,
        "params": info.params,
        "attrs": list(info.input_attr_specs.keys())
        + list(info.out_attr_to_column.keys()),
        "any_inputs": False,
    }
    for model, info in MODEL_TO_ELEMENT_SPECS.items()
}


META: Meta = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "Grid": {
            "public": True,
            "params": ["json", "xlsx", "net", "simbench", "network_function", "params"],
            "attrs": [],
            "any_inputs": False,
        },
        **ELEM_META_MODELS,
    },
    "extra_methods": [
        "get_extra_info",
        "get_net",
        "disable_elements",
        "enable_elements",
    ],
}


def apply_profiles(net: pp.pandapowerNet, profiles: Any, step: int):
    """Apply element profiles for the given step to the grid.

    :param profiles: profiles for elements in the format returned by
        simbench's ``get_absolute_values`` function.
    :param step: the time step to apply
    """
    for (elm, param), series in profiles.items():
        net[elm].update(series.loc[step].rename(param))  # type: ignore


def load_grid(params: dict[str, Any]) -> tuple[pp.pandapowerNet, Any]:
    """Load a grid and the associated element profiles (if any).

    :param params: A dictionary describing which grid to load. It should
        contain one of the following keys (or key combinations).

        - `"net"` where the corresponding value is a pandapowerNet
        - `"json"` where the value is the name of a JSON file in
          pandapower JSON format
        - `"xlsx"` where the value is the name of an Excel file
        - `"network_function"` giving the name of a function in
          pandapower.networks. In this case, the additional key
          `"params"` may be given to specify the kwargs to that function
        - `"simbench"` giving a simbench ID (if simbench is installed)

    :return: a tuple consisting of a :class:`pandapowerNet` and "element
        profiles" in the form that is returned by simbench's
        get_absolute_values function (or ``None`` if the loaded grid
        is not a simbench grid).

    :raises ValueError: if multiple keys are given in `params`
    """
    found_sources: set[str] = set()
    result: tuple[pp.pandapowerNet, Any] | None = None

    # Accept a pandapower grid
    if net := params.get("net", None):
        if isinstance(net, pp.pandapowerNet):
            result = (net, None)
            found_sources.add("net")
        else:
            raise ValueError("net is not a pandapowerNet instance")

    if json_path := params.get("json", None):
        result = (pp.from_json(json_path), None)
        found_sources.add("json")

    if xlsx_path := params.get("xlsx", None):
        result = (pp.from_excel(xlsx_path), None)
        found_sources.add("xlsx")

    if network_function := params.get("network_function", None):
        result = (
            getattr(pandapower.networks, network_function)(**params.get("params", {})),
            None,
        )
        found_sources.add("network_function")

    if simbench_id := params.get("simbench", None):
        import simbench as sb

        net = sb.get_simbench_net(simbench_id)
        profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        # Remove profile keys for element types that don't exist in the
        # grid. (The corresponding profiles will be empty, which would
        # result in indexing errors in `apply_profiles` later.)
        profiles = {
            (elm, col): df for (elm, col), df in profiles.items() if not net[elm].empty
        }
        result = (net, profiles)
        found_sources.add("simbench")

    if len(found_sources) != 1 or not result:
        raise ValueError(
            f"too many or too few sources specified for grid, namely: {found_sources}"
        )

    return result
