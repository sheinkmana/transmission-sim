#!/bin/bash
#SBATCH --job-name=malaria-1m-beta-q50
#SBATCH --output=sbatch/logs-sim/beta_q50_%A_%a.out
#SBATCH --error=sbatch/logs-sim/beta_q50_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-2%1
#SBATCH --chdir=/nfs/jbailey5/baileyweb/cmeiersc/Projects/transmission-sim
set -euo pipefail

repo=/nfs/jbailey5/baileyweb/cmeiersc/Projects/transmission-sim
run_dir="$repo/outputs/transmission_120gen_1m_beta_nh1000_q50_v1"
cd "$run_dir/source"
export PYTHONPATH="$run_dir/source"
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
# Reuse same-shape compilations across the three sequential split jobs.
export JAX_COMPILATION_CACHE_DIR="$run_dir/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

case "${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}" in
  0) split=train; batches=7813; seed=2026111501 ;;
  1) split=validation; batches=782; seed=2026111502 ;;
  2) split=test; batches=782; seed=2026111503 ;;
  *) exit 1 ;;
esac

sha256sum --check SHA256SUMS
test ! -e "$run_dir/${split}.zarr"
# Exercise the largest Q at production batch size before writing a split.
"$repo/.venv/bin/python" -u - <<'PY'
import time
import jax
import numpy as np
from omegaconf import OmegaConf
from examples.build_malaria_dataset import _build_simulator
assert all(device.platform == 'gpu' for device in jax.devices()), jax.devices()
print('Devices:', jax.devices(), flush=True)
sim = _build_simulator(OmegaConf.load('examples/configs/malaria_beta_nh1000_q50.yaml'))
start = time.monotonic()
batch = sim.simulate_batch(jax.random.key(98765), num_generations=120,
                          num_simulations=128, num_transmitted_alleles=50, p_init=.25)
counts = np.asarray(batch.num_hosts_by_state)
assert counts.shape == (128,120,152)
assert np.all(counts >= 0) and np.all(counts.sum(-1) == 1000)
print('Q=50 GPU preflight passed; seconds:', time.monotonic()-start, flush=True)
PY

"$repo/.venv/bin/python" -u -m examples.build_malaria_dataset \
  --config-name malaria_beta_nh1000_q50 \
  seed="$seed" num_batches="$batches" \
  output.store="$run_dir/${split}.zarr" \
  hydra.run.dir="$run_dir/hydra/${split}"
