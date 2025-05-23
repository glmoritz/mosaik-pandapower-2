import json

import pytest
from loguru import logger
from mosaik import World


@pytest.fixture
def world():
    with World(
        {
            "Grid": {"python": "mosaik_components.pandapower:Simulator"},
            "Asserter": {"python": "test_simulators.assert_simulator:AssertSimulator"},
            "Const": {"python": "test_simulators.const_simulator:ConstSimulator"},
        }
    ) as world:
        yield world


def test_scenario(world: World):
    ppsim = world.start("Grid")
    # grid = ppsim.Grid(grid="1-MVLV-semiurb-5.220-1-no_sw")
    grid = ppsim.Grid(
        network_function="create_cigre_network_mv", params={"with_der": False}
    )
    asserter = world.start("Asserter").Entity(
        expected={0: {"P[MW]": [-45.0457317940754]}}
    )

    world.connect(grid.children[0], asserter, "P[MW]")

    world.run(until=1)


def test_scenario_simbench(world: World):
    ppsim = world.start("Grid")
    grid = ppsim.Grid(simbench="1-MVLV-semiurb-5.220-1-no_sw")
    extra_info = ppsim.get_extra_info()
    asserter = world.start("Asserter").Entity(
        expected={
            0: {
                "load": [0],
                "bus": [-1.486891509],
            },
        },
    )
    load = next(e for e in grid.children if e.type == "Load")
    load_bus_index = extra_info[load.eid]["bus"]
    bus = next(
        e
        for e in grid.children
        if e.type == "Bus" and extra_info[e.eid]["index"] == load_bus_index
    )
    world.connect(load, asserter, ("P[MW]", "load"))
    world.connect(bus, asserter, ("P[MW]", "bus"))
    ppsim.disable_elements([load.eid])
    world.run(until=1)


def test_scenario_2(world: World):
    ppsim = world.start("Grid")
    grid = ppsim.Grid(
        network_function="create_cigre_network_mv", params={"with_der": False}
    )
    logger.warning(f"{grid.children}")
    constsim = world.start("Const")
    const = constsim.Const(value=3)
    world.connect(const, grid.children[1], ("value", "P_gen[MW]"))
    world.run(until=1)


def test_inexistent_file(world: World):
    ppsim = world.start("Grid")
    with pytest.raises(UserWarning):
        ppsim.Grid(json="does_not_exist.json")


def test_invalid_grid(world: World):
    ppsim = world.start("Grid")
    with pytest.raises(json.decoder.JSONDecodeError):
        ppsim.Grid(json="tests/data/invalid_grid.json")
