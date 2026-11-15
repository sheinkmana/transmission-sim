import numpy as np
import pytest
from jax import random
from omegaconf import OmegaConf

from dl4bi_sps.priors import Prior
from examples.build_malaria_dataset import _build_simulator, make_histogram_dataset


def test_requested_beta_marginals_and_q_rounding():
    cfg = OmegaConf.load("examples/configs/malaria_beta_nh1000_q50.yaml")
    sim = _build_simulator(cfg)
    for i, (name, target) in enumerate([("sw", -.2), ("st", .25), ("chi", 1/3), ("p_init", .25)]):
        x = np.asarray(getattr(sim, name).sample(random.key(i), (100_000,)))
        assert abs(x.mean() - target) < .006
        assert np.all((x > -1) & (x <= 0)) if name == "sw" else np.all((x >= 0) & (x <= 1))
    q = np.asarray(sim.num_transmitted_alleles.sample(random.key(9), (100_000,)))
    assert np.issubdtype(q.dtype, np.integer)
    np.testing.assert_array_equal(np.unique(q), np.arange(1, 51))
    # Exact mass for rounded log-uniform, including half-width endpoint bins.
    k = np.arange(1, 51)
    expected = np.log(np.minimum(50, k+.5)/np.maximum(1, k-.5))/np.log(50)
    np.testing.assert_allclose(np.bincount(q, minlength=51)[1:]/len(q), expected, atol=.004)
    np.testing.assert_array_equal(sim.num_hosts.sample(random.key(8),(10,)), np.full(10,1000))
    assert cfg.output.overwrite is False


@pytest.mark.parametrize("q", [1, 50])
def test_requested_config_simulates_q_endpoints(q):
    sim = _build_simulator(OmegaConf.load("examples/configs/malaria_beta_nh1000_q50.yaml"))
    batch = sim.simulate_batch(random.key(q), num_generations=3,
                              num_simulations=8, num_transmitted_alleles=q, p_init=.25)
    counts = np.asarray(batch.num_hosts_by_state)
    assert counts.shape == (8,3,3*q+2)
    assert (counts >= 0).all()
    np.testing.assert_array_equal(counts.sum(-1), np.full((8,3),1000))
    ds = make_histogram_dataset(batch,np.arange(8),np.arange(3),42)
    assert ds.num_hosts_by_state.dtype == np.uint16
