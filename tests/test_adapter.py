import json

import pytest
from loguru import logger
from mosaik import World


@pytest.fixture
def world():
    world = World(
        {
            "Grid": {"python": "mosaik_components.pandapower:Simulator"},
            "Asserter": {"python": "test_simulators.assert_simulator:AssertSimulator"},
            "Const": {"python": "test_simulators.const_simulator:ConstSimulator"},
        }
    )
    yield world
    world.shutdown()


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
    ppsim.Grid(simbench="1-MVLV-semiurb-5.220-1-no_sw")
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
    world.shutdown()


def test_invalid_grid(world: World):
    ppsim = world.start("Grid")
    with pytest.raises(json.decoder.JSONDecodeError):
        ppsim.Grid(json="tests/data/invalid_grid.json")
    world.shutdown()
