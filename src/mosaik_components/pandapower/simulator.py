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
        self.asymmetric_flow = False

    @override
    def init(self, sid: str, time_resolution: float, step_size: int | None = 900, asymmetric_flow: bool = False, **sim_params: Any) -> Meta:
        self._sid = sid
        if not step_size:
            self.meta["type"] = "event-based"
        self._step_size = step_size
        self.asymmetric_flow = asymmetric_flow
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
        
        asymmetric_sgen_indices = [
            pp.create_asymmetric_sgen(self._net, bus, p_a_mw=0.0, p_b_mw=0.0, p_c_mw=0.0)
            for bus in self._net.bus.index
        ]
        asymmetric_load_indices = [
            pp.create_asymmetric_load(self._net, bus, p_a_mw=0.0, p_b_mw=0.0, p_c_mw=0.0)
            for bus in self._net.bus.index
        ]
        
        self.bus_auto_elements = pd.DataFrame(
            {
                "load": load_indices,
                "sgen": sgen_indices,
                "asymmetric_load": asymmetric_load_indices,
                "asymmetric_sgen": asymmetric_sgen_indices,
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

        if self.asymmetric_flow:
            pp.runpp_3ph(self._net)
        else:
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
        table_name = f"res_{info.elem}"
        if self.asymmetric_flow:
            table_name = table_name + "_3ph"        
        elem_table = self._net[table_name]
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

    Specifications of this type are collected in the
    ``MODEL_TO_ELEMENT_SPECS`` dict, with the model name serving as the
    key. This dict is then used to both create the ``META`` for the
    ``init`` method and the grid's children during ``create``.
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
        lambda elem_tuple, net: {}
    )
    """Function returning the extra info for this type of element tuple
    and the entire net.
    """
    can_switch_off: bool = False
    """Whether elements of this type may be switched off (and on) using
    the ``disable_element`` (``enable_element``) extra methods.
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
            "P_a_gen[MW]": InputAttrSpec(
                column="p_a_mw",
                target_elem="asymmetric_sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_sgen"],
            ),
            "P_b_gen[MW]": InputAttrSpec(
                column="p_b_mw",
                target_elem="asymmetric_sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_sgen"],
            ),
            "P_c_gen[MW]": InputAttrSpec(
                column="p_c_mw",
                target_elem="asymmetric_sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_sgen"],
            ),
            "P_a_load[MW]": InputAttrSpec(
                column="p_a_mw",
                target_elem="asymmetric_load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_load"],
            ),
            "P_b_load[MW]": InputAttrSpec(
                column="p_b_mw",
                target_elem="asymmetric_load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_load"],
            ),
            "P_c_load[MW]": InputAttrSpec(
                column="p_c_mw",
                target_elem="asymmetric_load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_load"],
            ),
            "Q_a_gen[MVar]": InputAttrSpec(
                column="q_a_mvar",
                target_elem="asymmetric_sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_sgen"],
            ),
            "Q_b_gen[MVar]": InputAttrSpec(
                column="q_b_mvar",
                target_elem="asymmetric_sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_sgen"],
            ),
            "Q_c_gen[MVar]": InputAttrSpec(
                column="q_c_mvar",
                target_elem="asymmetric_sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_sgen"],
            ),
            "Q_a_load[MVar]": InputAttrSpec(
                column="q_a_mvar",
                target_elem="asymmetric_load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_load"],
            ),
            "Q_b_load[MVar]": InputAttrSpec(
                column="q_b_mvar",
                target_elem="asymmetric_load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_load"],
            ),
            "Q_c_load[MVar]": InputAttrSpec(
                column="q_c_mvar",
                target_elem="asymmetric_load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "asymmetric_load"],
            ),
        },
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
            "Vm[pu]": "vm_pu",
            "Va[deg]": "va_degree",
            "Vm_a[pu]": "vm_a_pu",
            "Va_a[deg]": "va_a_degree",
            "Vm_b[pu]": "vm_b_pu",
            "Va_b[deg]": "va_b_degree",
            "Vm_c[pu]": "vm_c_pu",
            "Va_c[deg]": "va_c_degree",
            "P_a[MW]": "p_a_mw",
            "Q_a[MVar]": "q_a_mvar",
            "P_b[MW]": "p_b_mw",
            "Q_b[MVar]": "q_b_mvar",
            "P_c[MW]": "p_c_mw",
            "Q_c[MVar]": "q_c_mvar",
            "Unbalance[%]": "unbalance_percent",
        },
        get_extra_info=lambda elem_tuple, _net: {
            "nominal voltage [kV]": elem_tuple.vn_kv,
        },
    ),
    "AsymmetricLoad": ModelToElementSpec(
        elem="asymmetric_load",
        connected_buses=["bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P_a[MW]": "p_a_mw",
            "Q_a[MVar]": "q_a_mvar",
            "P_b[MW]": "p_b_mw",
            "Q_b[MVar]": "q_b_mvar",
            "P_c[MW]": "p_c_mw",
            "Q_c[MVar]": "q_c_mvar"
        },
        get_extra_info=lambda elem, _net: {
            "bus": elem.bus,
            **({"profile": elem.profile} if "profile" in elem._fields else {}),
        },
        can_switch_off=True,
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
    "AsymmetricStaticGen": ModelToElementSpec(
        elem="asymmetric_sgen",
        connected_buses=["bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P_a[MW]": "p_a_mw",
            "Q_a[MVar]": "q_a_mvar",
            "P_b[MW]": "p_b_mw",
            "Q_b[MVar]": "q_b_mvar",
            "P_c[MW]": "p_c_mw",
            "Q_c[MVar]": "q_c_mvar",
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
            "P_a[MW]": "p_a_mw",
            "Q_a[MVar]": "q_a_mvar",
            "P_b[MW]": "p_b_mw",
            "Q_b[MVar]": "q_b_mvar",
            "P_c[MW]": "p_c_mw",
            "Q_c[MVar]": "q_c_mvar",
        },
    ),
    "Transformer": ModelToElementSpec(
        elem="trafo",
        connected_buses=["hv_bus", "lv_bus"],
        input_attr_specs={},
        out_attr_to_column={
            "P_hv[MW]": "p_hv_mw",
            "Q_hv[MVar]": "q_hv_mvar",
            "P_lv[MW]": "p_lv_mw",
            "Q_lv[MVar]": "q_lv_mvar",
            "Pl[MW]": "pl_mw",
            "Ql[MVar]": "ql_mvar",
            "I_hv[kA]": "i_hv_ka",
            "I_lv[kA]": "i_lv_ka",
            "Vm_hv[pu]": "vm_hv_pu",
            "Vm_lv[pu]": "vm_lv_pu",
            "Va_hv[degree]": "va_hv_degree",
            "Va_lv[degree]": "va_lv_degree",    
            "P_a_hv[MW]": "p_a_hv_mw",
            "Q_a_hv[MVar]": "q_a_hv_mvar",
            "P_b_hv[MW]": "p_b_hv_mw",
            "Q_b_hv[MVar]": "q_b_hv_mvar",
            "P_c_hv[MW]": "p_c_hv_mw",
            "Q_c_hv[MVar]": "q_c_hv_mvar",
            "P_a_lv[MW]": "p_a_lv_mw",
            "Q_a_lv[MVar]": "q_a_lv_mvar",
            "P_b_lv[MW]": "p_b_lv_mw",
            "Q_b_lv[MVar]": "q_b_lv_mvar",
            "P_c_lv[MW]": "p_c_lv_mw",
            "Q_c_lv[MVar]": "q_c_lv_mvar",
            "Pl_a[MW]": "pl_a_mw",
            "Ql_a[MVar]": "ql_a_mvar",
            "Pl_b[MW]": "pl_b_mw",
            "Ql_b[MVar]": "ql_b_mvar",
            "Pl_c[MW]": "pl_c_mw",
            "Ql_c[MVar]": "ql_c_mvar",
            "I_a_hv[kA]": "i_a_hv_ka",
            "I_a_lv[kA]": "i_a_lv_ka",
            "I_b_hv[kA]": "i_b_hv_ka",
            "I_b_lv[kA]": "i_b_lv_ka",
            "I_c_hv[kA]": "i_c_hv_ka",
            "I_c_lv[kA]": "i_c_lv_ka",
            "Loading[%]": "loading_percent",
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
            # Active power flows
            "P_from[MW]": "p_from_mw",
            "P_to[MW]": "p_to_mw",
            
            # Reactive power flows
            "Q_from[MVar]": "q_from_mvar",
            "Q_to[MVar]": "q_to_mvar",
            
            # Line losses
            "Pl_a[MW]": "p_a_l_mw",
            "Ql_a[MVar]": "q_a_l_mvar",
            "Pl_b[MW]": "p_b_l_mw",
            "Ql_b[MVar]": "q_b_l_mvar",
            "Pl_c[MW]": "p_c_l_mw",
            "Ql_c[MVar]": "q_c_l_mvar",
            
            # Current measurements
            "I_from[kA]": "i_from_ka",
            "I_to[kA]": "i_to_ka",
            "I[kA]": "i_ka",
            
            # Voltage magnitudes
            "Vm_from[pu]": "vm_from_pu",
            "Vm_to[pu]": "vm_to_pu",
            
            # Voltage angles
            "Va_from[deg]": "va_from_degree",
            "Va_to[deg]": "va_to_degree",
            
            "P_a_from[MW]": "p_a_from_mw",
            "Q_a_from[MVar]": "q_a_from_mvar",
            "P_b_from[MW]": "p_b_from_mw",
            "Q_b_from[MVar]": "q_b_from_mvar",
            "P_c_from[MW]": "p_c_from_mw",
            "Q_c_from[MVar]": "q_c_from_mvar",            
            "P_a_to[MW]": "p_a_to_mw",
            "Q_a_to[MVar]": "q_a_to_mvar",
            "P_b_to[MW]": "p_b_to_mw",
            "Q_b_to[MVar]": "q_b_to_mvar",
            "P_c_to[MW]": "p_c_to_mw",
            "Q_c_to[MVar]": "q_c_to_mvar",            
            "I_a_from[kA]": "i_a_from_ka",
            "I_b_from[kA]": "i_b_from_ka",
            "I_c_from[kA]": "i_c_from_ka",
            "I_n_from[kA]": "i_n_from_ka",
            "I_a_to[kA]": "i_a_to_ka",
            "I_b_to[kA]": "i_b_to_ka",
            "I_c_to[kA]": "i_c_to_ka",
            "I_n_to[kA]": "i_n_to_ka",
            "Loading[%]": "loading_percent",
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
