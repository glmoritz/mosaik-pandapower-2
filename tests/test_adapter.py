from loguru import logger
import pytest

import mosaik
from mosaik import World


@pytest.fixture
def world():
    return World({
        "Grid": {"python": "mosaik_components.pandapower:Simulator"},
        "Asserter": {"python": "tests.assert_simulator:AssertSimulator"},
        "Const": {"python": "tests.const_simulator:ConstSimulator"},
    })


def test_scenario(world):
    ppsim = world.start("Grid")
    # grid = ppsim.Grid(grid="1-MVLV-semiurb-5.220-1-no_sw")
    grid = ppsim.Grid(source="create_cigre_network_mv", params={"with_der": False})
    asserter = world.start("Asserter").Entity(
        expected={
            0: {"P[MW]": [-45.045731794074804]}
        }
    )

    world.connect(grid.children[0], asserter, "P[MW]")

    world.run(until=1)


def test_scenario_2(world):
    ppsim = world.start("Grid")
    grid = ppsim.Grid(source="create_cigre_network_mv", params={"with_der": False})
    logger.warning(f"{grid.children}")
    constsim = world.start("Const")
    const = constsim.Const(value=3)
    world.connect(const, grid.children[1], ("value", "P_gen[MW]"))
    #world.connect(const, grid.children[2], ("value", "P_MW_load"))
    world.run(until=1)
    #assert 0 == 1

    
def test_inexistent_file(world: World):
    ppsim = world.start("Grid")
    with pytest.raises(ValueError) as exc:
        ppsim.Grid(source="does_not_exist.json")
    world.shutdown()


def test_invalid_grid(world: World):
    ppsim = world.start("Grid")
    with pytest.raises(ValueError) as exc:
        ppsim.Grid(source="tests/data/invalid_grid.json")
    world.shutdown()