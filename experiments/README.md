# Experiments

This directory contains **experiment wrappers and benchmark adapters**. The core VBO-MI training loop remains in `src/vbo_mi`:
## Synthetic benchmarks

`synthetic/run_synthetic_suite.py` runs VBO-MI on the synthetic functions used in the paper:

- Ackley (10D)
- Levy (20D)
- Griewank (20D)
- Hartmann-6D
- Rastrigin (20D)

Fast smoke test:

```bash
python experiments/synthetic/run_synthetic_suite.py \
  --tasks ackley --seeds 0 --warmup 2 --iterations 5 --batch-size 16
```


## Real-world tasks

`real_world/` contains development adapters for the three real-world tasks in the paper:

- Lunar Lander
- Pest Control
- Rover Trajectory Planning
