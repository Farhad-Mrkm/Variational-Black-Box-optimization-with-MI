def test_both_packages_import():
    import vbo_mi
    import vbo_baselines

    assert vbo_mi.VBOMI is not None
    assert vbo_baselines.ExactGPBaseline is not None
