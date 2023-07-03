Currently, the adapter supports the following grid specifications:

- You can give the name of a grid file in JSON or Excel format (as supported by
  pandapower). The adapter will look for the file in its working directory.
- You can give the name of a function in pandapower.networks that returns a
  net. You can specify arguments to that function via the param `params`.
- You can give the name of a simbench case, provided that simbench is installed
  (for the Python instance running the adapter).
- You can pass a pandapowerNet object directly. This only works if the adapter
  is running in the same process as the scenario, i.e. if it has been started
  using the `"python"` method.