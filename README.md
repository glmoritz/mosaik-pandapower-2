# mosaik-pandapower-2

This is an adapter for using the network calculation program [pandapower] in a
[mosaik] simulation.

This simulator is still work in progress. In particular, not every desirable
attribute is implemented yet. If you have need for a particular entity or
attribute, leave an [issue here].

[pandapower]: https://www.pandapower.org
[mosaik]: https://mosaik.offis.de
[issue here]: https://gitlab.com/mosaik/components/energy/mosaik-pandapower-2/-/issues


## Usage

### Installation

This package can be installed from PyPI as `mosaik-pandapower-2`.

This simulator supports simbench grids. If you want to use them, you need to
install the simbench package from PyPI.

pandapower can be sped up by also installing numba. As this can be tricky on
some systems (and not strictly necessary), it is not listed as an explicit
dependency.

### Setup

First, add the simulator to your `SIM_CONFIG` as normal:

```python
SIM_CONFIG = {
    "Pandapower": {"python": "mosaik_components.pandapower:Simulator"},
    ...
}
```

Having created your `world`, you can then start an instance of the simulator via

```python
pp_sim = world.start("Pandapower", step_size=900)
```

The `step_size` specifies at which steps this simulator runs. (The default is
`step_size=900`.) If you set `step_size` to `None`, the simulator will run in
event-based mode, i.e. it will step whenever it receives new input.

Finally, you can create the `Grid` entity. There are several ways of doing this:

- If you have a `pandapowerNet` instance `net` in your scenario and the
  pandapower simulator is running in the same Python instance, you can use that
  grid by calling

  ```python
  grid = pp_sim.Grid(net=net)
  ```

  Note that the intended use for this is that you set up the grid using
  pandapower's functions and then pass the `pandapowerNet` to the adapter. The
  adapter does not expect the supplied net to be changed by anything (but
  itself) afterwards. If you continue to tinker with the grid, your results may
  be incorrect.

- If the grid is in a JSON file (in pandapower’s format), you can call
  ```python
  grid = pp_sim.Grid(json=path_to_json)
  ```

- Similarly, if the grid is in an Excel file,
  ```python
  grid = pp_sim.Grid(xlsx=path_to_xlsx)
  ```

- If you want to use one of the network creation functions in
  `pandapower.networks`, you can specify
  ```python
  grid = pp_sim.Grid(network_function=function_name, params=params)
  ```
  where `function_name` is the name of the function as a string and `params`
  is a dictionary that will be used as the keyword arguments to that function
  (it will default to `{}` if not given).

- Finally, if you want to use a simbench grid,
  ```python
  grid = pp_sim.Grid(simbench=simbench_id)
  ```
  This method requires simbench to be installed in the same (virtual)
  environment as the adapter. This does not happen automatically when installing
  mosaik-pandapower-2, as we don't want to burden users who don't need it with
  this dependency.

In every case, you will get a `Grid` entity `grid`. The simulator supports only
one such entity. In case that you want to simulate several grids, you need to
start several instances of the simulator. (Design note: Requiring multiple grids
should be rare; restricting to one of them shortens the entity IDs of the
children as we don’t need to track to which grid they belong.)


### Identifying grid elements

The `Grid` entity `grid` will have a child entity for each supported element
present in the grid. You can access them via `grid.children`. You can filter for
specific types of entities by checking the child entity's `type` attribute.

The entity IDs of the children follow the format `ElementType-Index` where
`ElementType` is the same as the `type` and `Index` is the element's index
in the element's table in the `pandapowerNet` object.

If you are using mosaik 3.3 or later, each entity will also have an `extra_info`
field holding a dict with additional information about this entity. This will
always include the entity's index (so you don't need to parse its entity ID) and
its name in the pandapower grid. Depending on the element, other information is
included as well. If you are missing a field, feel free to leave an issue.

Earlier versions of mosaik do not support `extra_info` in this way. To still
enable you to access it, the simulator has a `get_extra_info` extra method,
which can be called like so, *after* creating the grid:

```python
extra_info = pp_sim.get_extra_info()
```

This is a dict mapping each entity ID to its entity's extra info.


### Connecting other simulators to the grid

The most common case when connecting entities to the grid is to connect them
as loads or static generators. (Essentially, this encompasses all entities
that provide real and reactive power values.) Therefore, this simulator is
optimized for this case and you can connect these entities directly to the `Bus`
entities. Use the `P_gen[MW]` and `Q_gen[MVar]` attributes if your simulator
follows the generator convention (i.e. generation is positive), and use the
`P_load[MW]` and `Q_load[MVar]` attributes if your simulator follows the
consumer convention (i.e. consumption is positive).

If your entity models a generator instead (i.e. it produces real power and
maintains a fixed voltage level instead of a fixed reactive power), the process
is slightly more involved. First, you need to create a `ControlledGen` entity
at the bus that you want to control, by calling

```python
gen = pp_sim.ControlledGen(bus=bus_index)
```

where `bus_index` is the index of the bus where the generator should be
connected. Then, you can connect your simulator to the `gen` entity.


### Table of entity types and their attributes

The following table describes the model that this simulator supports, the
attributes of these models, and whether they’re used as inputs or outputs.

| Model         | Attribute      | In/Out | Description                                            |
|---------------|----------------|--------|--------------------------------------------------------|
| Grid          |                |        | **meta entity representing the entire grid**           |
| Bus           |                |        | **a bus in the grid**                                  |
|               | `P_gen[MW]`    | In     | real power in MW, following generator convention       |
|               | `Q_gen[MVar]`  | In     | reactive power in MVar, following generator convention |
|               | `P_load[MW]`   | In     | real power in MW, following consumer convention        |
|               | `Q_load[MVar]` | In     | reactive power in MVar, following consumer convention  |
|               | `P[MW]`        | Out    | entire real power at this bus                          |
|               | `Q[MVar]`      | Out    | entire reactive power at this bus                      |
|               | `Vm[pu]`       | Out    | voltage magnitude at this bus (in per unit)            |
|               | `Va[deg]`      | Out    | voltage angle at this bus (in degrees)                 |
| Load          |                |        | **an existing load in the grid**                       |
|               | `P[MW]`        | Out    | the real power consumed by this load                   |
|               | `Q[MVar]`      | Out    | the reactive power consumed by this load               |
| StaticGen     |                |        | **an existing static generator in the grid**           |
|               | `P[MW]`        | Out    | the real power produced by this generator              |
|               | `Q[MVar]`      | Out    | the reactive power produced by this generator          |
| Gen           |                |        | **an existing generator in the grid**                  |
|               | `P[MW]`        | Out    | the real power produced by this generator              |
|               | `Q[MVar]`      | Out    | the reactive power produced by this generator          |
|               | `Vm[pu]`       | Out    | the voltage magnitude this generator tries to hold     |
|               | `Va[deg]`      | Out    | the voltage angle for this generator                   |
| ControlledGen |                |        |**a generator created by the user to control**          |
| Line          |                |        | **a line in the grid**                                 |
|               | `I[kA]`        | Out    | the current along the line                             |
|               | `loading[%]`   | Out    | the loading of the line                                |


### Getting the net

The simulator offers a `get_net` extra method which can be called on the simulator object in your mosaik scenario, once you have created the *Grid* entity.
It will return the internal `pandapowerNet` object for the grid.

**Note**: This method exists purely for visualization and debugging purposes.
If you change values on this objects, the simulation might crash or the results might be silently incorrect.

**Note**: This method only works if you run the adapter in the same Python process as your scenario (i.e. if you start the simulator using the `"python"` option in your sim config).
It is not possible to pass the `pandapowerNet` object between processes.


### Disabling existing elements

This simulator purposefully does not allow you to overwrite the values of loads and other elements that already exist in the grid.
However, you may switch them off entirely.
For this, use the `disable_elements` extra method.
If `loads` is a list of *Load* entities that you want to disable, you can achieve this via

```python
pp_sim.disable_elements([load.eid for load in loads])
```

This works by setting the element's *in_service* value to `False`.
You can undo this (or enable elements that are not in service in your grid file) by using the analogous `enable_elements` extra method.


## Development

For the development of this simulator, the following tools are employed:

-   [Hatch](https://hatch.pypa.io/latest/) is used as a packaging manager.
    This offers the following commands:

    -   `hatch fmt` to format the code (using ruff)
    -   `hatch run test:test` to run pytest in a test matrix consisting of Python versions 3.9 and 3.11 and mosaik versions 3.2 and 3.3.0b1.
    -   `hatch run python` for running Python.
    -   `hatch run` to run arbitrary commands in the managed virtualenv.

    Also, we use `hatch-vcs` to automatically deduce version numbers from git tags.
    Adding a new tag starting with v on the main branch should automatically release this on PyPI.


-   [pre-commit](https://pre-commit.com/) is used to run hooks before committing and pushing.
    Install pre-commit (I recommend `pipx`) and install the hooks using `pre-commit install`.
