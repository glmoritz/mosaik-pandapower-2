This is an adapter for using the network calculation program [pandapower] in a
[mosaik] simulation.

This simulator is still work in progress. In particular, not every desirable
attribute is implemented yet. If you have need for a particular entity or
attribute, leave an [issue here].

[pandapower]: https://www.pandapower.org
[mosaik]: https://mosaik.offis.de
[issue here]: https://gitlab.com/mosaik/components/energy/mosaik-pandapower-2/-/issues

## Usage

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

The `step_size` specifies at which steps this simulator runs. If you don’t give
a step size or set it to `None`, the simulator will run in event-based mode,
i.e. whenever it receives new input.

Finally, you can create the `Grid` entity. There are several ways of doing this:

- If you have a pandapowerNet instance `net` in your scenario and the pandapower
  simulator is running in the same Python instance, you can use that grid by
  calling
  
  ```python
  grid = pp_sim.Grid(net=net)
  ```
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

| Entity type   | Attribute      | In/Out | Description                                            |
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

