# VBO-MI

**VBO-MI: A Differentiable Black-Box Optimization Framework via Variational Mutual Information Estimation**

This repository contains an implementation of VBO-MI together with the **seven baseline surrogate methods** used in the accompanying TMLR study:

- Gaussian Process (GP)
- Deep Kernel Learning (DKL)
- Infinite-width Bayesian Neural Network (I-BNN / NNGP)
- Hamiltonian Monte Carlo (HMC)
- Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)
- Linearized Laplace Approximation (LLA)
- Focal Sparse Gaussian Process

The implementation is organized so that the proposed method and the baselines can be installed and tested from one repository while retaining separate model packages and experiment scripts.

## Repository contents

```text
VBO-MI/
├── src/
│   ├── vbo_mi/                  # Proposed VBO-MI method
│   └── vbo_baselines/           # Seven baseline surrogate methods
├── benchmarks/                  # Reserved for task-specific benchmark adapters
├── configs/                     # Paper-oriented configurations
├── experiments/                 # Experiment notes / shared experiment material
├── scripts/
│   ├── vbo_mi/                  # VBO-MI runs and MI validation
│   └── baselines/               # Baseline smoke and suite runners
├── tests/                       # Automated tests
├── docs/
│   └── Baseline_Audit.md        # Audit of the preliminary baseline notebook
├── legacy/                      # Original supplied development notebooks/scripts
├── notebooks/
├── README.md
├── pyproject.toml
├── requirements.txt
└── LICENSE_NOTICE.md
```

## Installation

For the complete research environment:

```bash
pip install -e ".[all]"
```

For the core packages only:

```bash
pip install -e .
```

LLA requires `laplace-torch`; the LLA dependency is kept optional because the other methods do not require it:

```bash
pip install -e ".[lla]"
```

## VBO-MI

VBO-MI uses three neural components:

1. **Action-Net** $E_\\phi$: maps Gaussian seeds to bounded candidate actions.
2. **Helper** $D_\\theta$: a pairwise LSTM + MLP critic used to estimate the Donsker--Varadhan variational mutual-information bound.
3. **Surrogate** $\\hat f_\\psi$: a small MLP trained on the replay buffer and used to provide the exploitation signal without querying the black-box objective inside the internal update loops.

The main implementation follows the algorithm in the paper: warm-up oracle evaluations, surrogate pre-training, helper updates, surrogate updates, action-net updates with helper/surrogate parameters frozen, and then one real oracle evaluation batch per main iteration.


### Run a small VBO-MI example

```bash
python scripts/vbo_mi/run_ackley.py --warmup 2 --iterations 5 --batch-size 16
```

### Run the paper-oriented synthetic configuration

The supplied manuscript uses batch size 60 and 30 warm-up iterations for the main synthetic benchmarks, with 70 main iterations for Ackley, Levy and Griewank.

```bash
python scripts/vbo_mi/run_ackley.py --warmup 30 --iterations 70 --batch-size 60
```

The corresponding repository configuration is in `configs/paper_synthetic.yaml`.


The manuscript frames Ackley, Levy and Griewank as maximization problems by optimizing the negative of their conventional minimization objective.

## Baselines

The baseline package is `vbo_baselines`. All seven methods share the same common BO interface and common qEI acquisition optimizer.

The baseline protocol encoded from the supplied TMLR Appendix C.1.2 is:

- Monte-Carlo Expected Improvement (qEI)
- Sobol initial designs
- 10 initial points for synthetic tasks
- 50 initial points for Lunar Lander and Rover Trajectory
- 20 initial points for Pest Control
- GP: Matérn-5/2, length-scale prior Gamma(3,6), output-scale prior Gamma(2.0, 0.15)
- DKL: 3-layer Tanh MLP, width 128, Matérn-5/2 base kernel
- I-BNN: NNGP kernel, depth 3, ReLU, weight variance 10.0, bias variance 1.6
- HMC: 3-layer Tanh MLP, width 128
- SGHMC: 3-layer Tanh MLP, width 128, adaptive step size, heteroskedastic likelihood output
- LLA: 3-layer Tanh MLP, width 128
- Focal Sparse GP: Matérn-5/2 with 50 inducing variables for synthetic tasks and 200 for real-world tasks

The manuscript states that HMC, SGHMC, LLA, DKL and I-BNN follow Li et al. (2024), while Focal Sparse GP follows Wei et al. (2024). Details that are not explicit in the supplied manuscript remain configurable instead of being silently invented.

### Run the baseline smoke test

```bash
python scripts/baselines/smoke_test.py
```

### Run the baseline suite smoke test

```bash
python scripts/baselines/run_smoke_suite.py --task branin
```

## MI diagnostic experiment

The repository includes a small Gaussian-channel MI validation script:

```bash
python scripts/vbo_mi/run_ar_lstm_validation.py
```

This is a diagnostic experiment, not one of the main TMLR benchmark runs.

## Tests

```bash
pytest -q
```

## Legacy code

The original supplied development files are preserved under `legacy/`:

- `legacy/mycodework_original.py`
- `legacy/untitled50_baselines.py`

They are kept for provenance and comparison; the modular packages under `src/` are the preferred entry points.

## References

- VBO-MI: *A Differentiable Black-Box Optimization Framework via Variational Mutual Information Estimation*.
- Li et al., *A Study of Bayesian Neural Network Surrogates for Bayesian Optimization*: https://github.com/yucenli/bnn-bo
- Wei et al., *Scalable Bayesian Optimization via Focalized Sparse Gaussian Processes*: https://github.com/yunyuewei/FocalBO

## Citation

