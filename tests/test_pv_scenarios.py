import json
import pathlib
import re

import numpy
import pulp
import pytest

from optimizer.app import app
from optimizer.optimizer import BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData

# a load in the evening, solar over the day, and a first time step that is cheap to import in.
# Everything below varies one piece of this and compares the answers.
GT = [500, 500, 500, 500, 3000, 3000]
FT = [1000, 2000, 4000, 2000, 0, 0]
P_N = [0.0002, 0.0004, 0.0004, 0.0004, 0.0005, 0.0005]


def build(pv_spread=0.0, p_N=None, p_E=0.00018, gt=None, ft=None, weights=None,
          prc_e_goal_pen=None, **battery):
    config = dict(charge_from_grid=True, discharge_to_grid=False, s_capacity=6000, s_min=0,
                  s_max=6000, s_initial=0, c_min=0, c_max=3000, d_max=3000, p_a=0.0)
    config.update(battery)
    series = TimeSeriesData(dt=[3600] * len(GT), gt=gt or GT, ft=ft or FT,
                            p_N=p_N or P_N, p_E=[p_E] * len(GT))
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='none', discharging_strategy='none'),
        grid=GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None),
        batteries=[BatteryConfig(**config)], time_series=series,
        eta_c=0.95, eta_d=0.95, M=1e6,
        pv_spread=pv_spread, pv_scenario_weights=weights, prc_e_goal_pen=prc_e_goal_pen)


@pytest.mark.parametrize('test_case', pathlib.Path('test_cases').glob('*.json'))
def test_a_zero_spread_answers_exactly_as_before(test_case: pathlib.Path):
    # the gate the whole change hangs on: asking for no spread has to reach the model that was
    # there before the scenarios, not an equivalent of it. Compared over the whole response, not
    # only the objective value, because two schedules of equal worth are the easy way to fail this.
    request = json.loads(test_case.read_text())['request']
    if request.get('pv_spread'):
        pytest.skip('the case plans over scenarios itself')
    client = app.test_client()

    without = client.post('/optimize/charge-schedule', json=request).json
    with_zero = client.post('/optimize/charge-schedule', json=dict(request, pv_spread=0.0)).json

    assert with_zero == without


def test_a_zero_spread_builds_the_single_scenario_model():
    # and it is one scenario carrying the forecast itself, under the variable names the model
    # always had. A suffixed name would be the same problem for the solver but not the same file,
    # and CBC picks its way through a degenerate plateau by the order it reads.
    model = build(pv_spread=0.0)
    model.create_model()

    assert len(model.scenarios) == 1
    assert model.scenarios[0].weight == 1.0
    assert model.scenarios[0].ft is model.time_series.ft
    assert not [var for var in model.problem.variables() if re.search(r'_s\d+$', var.name)]


def test_a_spread_replicates_the_horizon_per_scenario():
    single = build(pv_spread=0.0)
    single.create_model()
    spread = build(pv_spread=0.4)
    spread.create_model()

    # the shortfall scenario and the forecast itself, the expected one last because solve()
    # reports it. Only the downside is modelled, see _build_scenarios.
    assert len(spread.scenarios) == 2
    assert [round(sc.weight, 3) for sc in spread.scenarios] == [0.1, 0.9]
    assert spread.reported_scenario is spread.scenarios[-1]
    assert spread.reported_scenario.ft == FT
    # the first time step is the one being decided, so its yield is the same in both. The rest of
    # the horizon is scaled.
    assert [sc.ft[0] for sc in spread.scenarios] == [FT[0]] * 2
    assert [round(sc.ft[2]) for sc in spread.scenarios] == [2400, 4000]

    # every time indexed variable exists once per scenario, the peak and demand rate variables
    # stay single. The extra rows are the per scenario constraints plus the tie on the first step.
    assert len(spread.problem.variables()) == 2 * len(single.problem.variables())
    assert len(spread.problem.constraints) > 2 * len(single.problem.constraints)


def test_the_first_time_step_is_one_decision():
    # non anticipativity. The scenarios differ from the second step on, but the step the caller is
    # about to act on is decided before the yield is known and is therefore the same in all of them.
    model = build(pv_spread=0.5)
    assert model.solve()['status'] == 'Optimal'

    for key in ('n', 'e'):
        values = [pulp.value(sc.variables[key][0]) for sc in model.scenarios]
        assert numpy.allclose(values, values[0], atol=1e-6), f'{key} at t=0 differs: {values}'
    for key in ('c', 'd', 's'):
        values = [pulp.value(sc.variables[key][0][0]) for sc in model.scenarios]
        assert numpy.allclose(values, values[0], atol=1e-6), f'{key} at t=0 differs: {values}'

    # and they do go their own way afterwards, otherwise the test above proves nothing
    later = {tuple(round(pulp.value(var)) for var in sc.variables['n'][1:]) for sc in model.scenarios}
    assert len(later) > 1, f'every scenario imports the same: {later}'


def test_a_flat_tariff_schedule_is_left_alone_by_a_spread():
    # with one price for the whole horizon there is nothing to arbitrage, so the yield the schedule
    # is planned against cannot change it. The battery takes every Wh of solar either way.
    # Deliberately not capacity bound: charging then has a single optimum instead of a plateau the
    # spread could pick a different corner of.
    flat = dict(p_N=[0.0003] * len(GT), p_E=0.0001, gt=[0, 0, 0, 0, 0, 12000],
                s_capacity=10000, s_max=10000, c_max=4000, d_max=10000)
    reference = build(pv_spread=0.0, **flat).solve()

    for pv_spread in (0.4, 0.8):
        result = build(pv_spread=pv_spread, **flat).solve()
        assert result['status'] == 'Optimal'
        for key in ('charging_power', 'discharging_power'):
            assert numpy.allclose(result['batteries'][0][key], reference['batteries'][0][key], atol=1), \
                f'spread {pv_spread} changed {key}: {result["batteries"][0][key]}'
        for key in ('grid_import', 'grid_export'):
            assert numpy.allclose(result[key], reference[key], atol=1), \
                f'spread {pv_spread} changed {key}: {result[key]}'


def test_a_spread_moves_charging_into_the_cheap_first_step():
    # the first step is cheap to import in and the later ones are not, so a yield that may come out
    # low is worth insuring against while the insurance is affordable. Without a spread the battery
    # waits for the solar it was promised.
    charged = []
    imported = []
    for pv_spread in (0.0, 0.3, 0.5):
        result = build(pv_spread=pv_spread).solve()
        assert result['status'] == 'Optimal'
        charged.append(result['batteries'][0]['charging_power'][0])
        imported.append(result['grid_import'][0])

    assert charged == sorted(charged) and charged[-1] > charged[0] + 1, charged
    assert imported == sorted(imported) and imported[0] < 1 < imported[-1], imported


def test_a_realistic_goal_penalty_price_stops_the_worst_case_hoarding():
    # the goal penalty defaults to a hundred times the largest real rate of the request. That is
    # invisible while the goal is reachable, and every scenario below reaches it on its own, but
    # the low yield one only just: protecting it is then worth more than any amount of money, so
    # the battery keeps its charge and the load of the first step is bought at the expensive rate.
    hoarding = dict(gt=[3000, 500, 500, 500, 500, 500], ft=[0, 2000, 4000, 2000, 0, 0],
                    p_N=[0.0005, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003], p_E=0.0001,
                    charge_from_grid=False, s_initial=4000,
                    s_goal=[0, 0, 0, 0, 0, 6000])

    legacy = build(pv_spread=0.5, **hoarding).solve()
    assert legacy['grid_import'][0] > 2000, legacy['grid_import']
    assert legacy['batteries'][0]['discharging_power'][0] < 1000

    # a price per Wh in the range of the import price weighs the shortfall against money again
    realistic = build(pv_spread=0.5, prc_e_goal_pen=0.0006, **hoarding).solve()
    assert realistic['grid_import'][0] < 1, realistic['grid_import']
    assert realistic['batteries'][0]['discharging_power'][0] > 2500
    assert realistic['objective_value'] > legacy['objective_value']

    # and none of it shows without a spread, which is why the default could stand as it is: the
    # single scenario reaches the goal, the shortfall is zero and its price never enters
    assert build(pv_spread=0.0, **hoarding).solve() == \
        build(pv_spread=0.0, prc_e_goal_pen=0.0006, **hoarding).solve()


def test_scenario_weights_are_normalized():
    model = build(pv_spread=0.4, weights=[1, 3])
    assert [sc.weight for sc in model.scenarios] == [0.25, 0.75]


@pytest.mark.parametrize('weights', [[0.5, 0.5, 0.5], [0.5], [0.0, 1.0], [1.0, 0.0], [-0.5, 1.5]])
def test_a_weight_of_zero_is_refused_rather_than_solved(weights):
    # a scenario weighted 0 stays in the model but drops out of the objective, so the solver hands
    # its variables back unoptimized. One of the two is the schedule that gets reported, and a
    # returned schedule nothing optimized is worse than a rejected request.
    with pytest.raises(ValueError):
        build(pv_spread=0.4, weights=weights)


@pytest.mark.parametrize('payload', [
    {'pv_spread': 1.5},
    {'pv_spread': -0.1},
    {'pv_spread': 0.3, 'pv_scenario_weights': [0.3, 0.3, 0.4]},
    {'pv_spread': 0.3, 'pv_scenario_weights': [0.5]},
    {'pv_spread': 0.3, 'pv_scenario_weights': [-0.5, 1.5]},
    {'pv_spread': 0.3, 'pv_scenario_weights': [0, 1]},
    {'pv_spread': 0.3, 'pv_scenario_weights': [1, 0]},
])
def test_invalid_scenario_input_is_rejected(payload):
    request = json.loads(pathlib.Path('test_cases/012-early-charging-not-perfect.json').read_text())['request']
    response = app.test_client().post('/optimize/charge-schedule', json=dict(request, **payload))
    assert response.status_code == 400, response.json


def level_peaks(pv_spread=0.0, prc_p_exc_imp=None, **kw):
    """A grid import the limit cannot hold, on a strategy that levels the import peak."""
    series = TimeSeriesData(dt=[3600, 3600], gt=kw.pop('gt', [2000, 0]), ft=kw.pop('ft', [0, 0]),
                            p_N=[0.0003, 0.0003], p_E=[0.0001, 0.0001])
    return Optimizer(
        strategy=OptimizationStrategy(charging_strategy='attenuate_demand_peaks',
                                      discharging_strategy='none'),
        grid=GridConfig(p_max_imp=100, p_max_exp=None, prc_p_exc_imp=prc_p_exc_imp),
        batteries=[BatteryConfig(charge_from_grid=False, discharge_to_grid=False, s_capacity=1000,
                                 s_min=0, s_max=1000, s_initial=0, c_min=0, c_max=0, d_max=0,
                                 p_a=0.0)],
        time_series=series, eta_c=0.95, eta_d=0.95, M=1e6, pv_spread=pv_spread, **kw)


@pytest.mark.parametrize('pv_spread', [0.0, 0.4])
def test_the_levelled_peak_counts_the_power_above_the_grid_limit(pv_spread):
    # the peak the levelling strategies work on is the whole grid power of the side, including the
    # portion beyond p_max_imp. Reading the excess variables off the wrong dictionary silently
    # drops that portion and the strategy then levels a 2000 W peak as if it were the 100 W limit.
    model = level_peaks(pv_spread=pv_spread)
    assert model.solve()['status'] == 'Optimal'

    # 2000 Wh over an hour against a 100 W limit: 100 Wh of it is regular import, 1900 Wh excess
    assert pulp.value(model.reported_scenario.variables['n'][0]) == pytest.approx(100, abs=1)
    assert pulp.value(model.reported_scenario.variables['e_imp_lim_exc'][0]) == pytest.approx(1900, abs=1)
    assert pulp.value(model.variables['p_imp_peak']) == pytest.approx(2000, abs=1)


def test_the_reported_demand_charge_is_the_reported_schedule_s():
    # the demand rate peak is one variable for all scenarios, so it carries the worst of them.
    # Charging the reported schedule for it bills an import that schedule does not make: here the
    # expected scenario covers its load from solar and imports nothing at all.
    model = level_peaks(pv_spread=0.5, prc_p_exc_imp=0.001, gt=[0, 1000], ft=[0, 1000])
    result = model.solve()
    assert result['status'] == 'Optimal'

    # the shortfall scenario has to import, and above the limit, so the shared peak is positive
    assert pulp.value(model.variables['p_max_imp_exc']) > 100
    # the reported one does not import at all, and its reported value says so
    assert result['grid_import'] == pytest.approx([0, 0], abs=1)
    assert result['objective_value'] == pytest.approx(0, abs=1e-6)
