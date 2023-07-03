import mosaik

def test_scenario():
    world = mosaik.World({
        "Grid": {"python": "mosaik_components.pandapower:Simulator"}
    })
    ppsim = world.start("Grid")
    # grid = ppsim.Grid(grid="1-MVLV-semiurb-5.220-1-no_sw")
    grid = ppsim.Grid(grid="create_cigre_network_mv", params={"with_der": False})

    world.run(until=1)
    
    
