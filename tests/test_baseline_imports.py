def test_package_imports():
    import vbo_baselines
    from vbo_baselines.baselines.gp import ExactGPBaseline
    from vbo_baselines.baselines.dkl import DKLBaseline
    from vbo_baselines.baselines.ibnn import IBNNBaseline
    from vbo_baselines.baselines.focal_sparse_gp import FocalSparseGPBaseline
    from vbo_baselines.baselines.hmc import HMCBaseline
    from vbo_baselines.baselines.sghmc import SGHMCBaseline
    from vbo_baselines.baselines.lla import LLABaseline

    assert all(
        cls is not None
        for cls in [
            ExactGPBaseline,
            DKLBaseline,
            IBNNBaseline,
            FocalSparseGPBaseline,
            HMCBaseline,
            SGHMCBaseline,
            LLABaseline,
        ]
    )
