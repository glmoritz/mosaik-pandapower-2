import json

from loguru import logger
import pytest

import mosaik
from mosaik import World


@pytest.fixture
def world():
    return World(
        {
            "Grid": {"python": "mosaik_components.pandapower:Simulator"},
            "Asserter": {"python": "assert_simulator:AssertSimulator"},
            "Const": {"python": "const_simulator:ConstSimulator"},
        }
    )


def test_scenario(world):
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


def test_scenario_2(world):
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


def test_problematic_profiles(world: World):
    ppsim = world.start("Grid", step_size=1)
    grid = ppsim.Grid(simbench="1-MV-rural--0-sw")
    world.run(until=10)
