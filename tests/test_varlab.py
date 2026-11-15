import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from varlab import christoffersen, hs_var_es, kupiec, normal_var_es, t_var_es  # noqa: E402

rng = np.random.default_rng(0)


def test_normal_var_es_textbook_values():
    var, es = normal_var_es(0.0, 1.0, 0.99)
    assert abs(var - 2.3263) < 1e-4 and abs(es - 2.6652) < 1e-4  # standard normal 99% VaR / ES


def test_t_var_es_matches_monte_carlo():
    nu, a = 5.0, 0.975
    z = stats.t.rvs(nu, size=2_000_000, random_state=1) * np.sqrt((nu - 2) / nu)  # unit-variance t
    loss = -z
    mc_var = np.quantile(loss, a)
    var, es = t_var_es(0.0, 1.0, nu, a)
    assert abs(var - mc_var) < 0.01
    assert abs(es - loss[loss >= mc_var].mean()) < 0.02


def test_hs_var_es_on_known_sample():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    var, es = hs_var_es(losses, 0.95)
    assert abs(var - np.quantile(losses, 0.95)) < 1e-12 and es == losses[losses >= var].mean()


def test_kupiec_accepts_correct_and_rejects_wrong_coverage():
    good = (rng.random(5000) < 0.01).astype(int)
    bad = (rng.random(5000) < 0.03).astype(int)
    assert kupiec(good, 0.99) > 0.05
    assert kupiec(bad, 0.99) < 0.001


def test_christoffersen_detects_clustering():
    independent = (rng.random(5000) < 0.02).astype(int)
    clustered = np.zeros(5000, dtype=int)
    for s in range(100, 5000, 500):
        clustered[s:s + 10] = 1  # exceedances arrive in bursts
    assert christoffersen(independent) > 0.05
    assert christoffersen(clustered) < 0.001
