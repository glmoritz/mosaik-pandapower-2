
# To run a time-series simulation with simbench, apparently:

import simbench as sb
import pandapower as pp
import pandapower.timeseries.run_time_series as pprts

def test_timeseries():
    print()
    net = sb.get_simbench_net("1-MVLV-semiurb-5.220-1-no_sw")  # load simbench grid
    print(net.profiles.keys())
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    sb.apply_const_controllers(net, profiles)

    pp.create_load(net, bus=42, p_mw=0.005)
    ts_variables = pprts.init_time_series(net, time_steps=range(0, 1000), verbose=False)

    time_step = 0  # Step to run
    pprts.run_time_step(net, 0, ts_variables)

    print(net.res_load.loc[215:, :])

def test_timeseries_no_ctrl():
    net = sb.get_simbench_net("1-MVLV-semiurb-5.220-1-no_sw")  # load simbench grid
    ts_variables = pprts.init_time_series(net, time_steps=range(0, 1000), verbose=False)

    time_step = 0  # Step to run
    pprts.run_time_step(net, time_step, ts_variables)

    print(net.res_load.loc[215:, :])
