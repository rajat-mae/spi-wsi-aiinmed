def test_imports():
    import spi_wsi_aiinmed
    from spi_wsi_aiinmed.prompt_robustness import jaccard
    assert spi_wsi_aiinmed.__version__
    assert jaccard({1}, {1, 2}) == 0.5
