import numpy as np
from jax import random
from omegaconf import OmegaConf
from scipy import stats

from examples.build_malaria_dataset import _build_simulator

CONFIG = "examples/configs/malaria_general_pmix_v1.yaml"
BASE = "examples/configs/malaria_beta_nh1000_q50.yaml"


def test_only_p_init_differs_from_bob_priors():
    cfg, base = OmegaConf.load(CONFIG), OmegaConf.load(BASE)
    for name in ["num_hosts", "num_transmitted_alleles", "chi", "sw", "st"]:
        assert cfg.priors[name] == base.priors[name], name
    assert cfg.priors.p_init != base.priors.p_init
    assert cfg.seed != base.seed and cfg.output.store != base.output.store
    assert cfg.output.overwrite is False


def test_p_init_mixture():
    sim = _build_simulator(OmegaConf.load(CONFIG))
    p = np.asarray(sim.p_init.sample(random.key(0), (200_000,)))
    assert p.min() >= 1e-3 and p.max() <= .7
    # Half log-uniform on [0.001, 0.05], half uniform on [0.001, 0.7].
    low_share = .5 + .5 * (.05 - .001) / (.7 - .001)
    assert abs(np.mean(p < .05) - low_share) < .005
    assert abs(np.mean(p > .5) - .5 * .2 / .699) < .005
    low = p[p < .05]
    # Below 0.05 the density is a log-uniform plus a small flat part; the log-uniform dominates.
    assert stats.kstest(np.log(low[low < .01]), "uniform", args=(np.log(1e-3), np.log(.01) - np.log(1e-3))).statistic < .02


def test_general_config_simulates_p_init_extremes():
    sim = _build_simulator(OmegaConf.load(CONFIG))
    for p_init in (1e-3, .7):
        batch = sim.simulate_batch(random.key(1), num_generations=3, num_simulations=8,
                                   num_transmitted_alleles=5, p_init=p_init)
        counts = np.asarray(batch.num_hosts_by_state)
        np.testing.assert_array_equal(counts.sum(-1), np.full((8, 3), 1000))
