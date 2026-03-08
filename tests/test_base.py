"""
Tests for base utilities: input conversion, validation, helpers.
These don't require CatBoost.
"""

import numpy as np
import polars as pl
import pytest

from insurance_distributional.base import (
    _clip_hessians,
    _normalize_gradients,
    _to_1d,
    _to_numpy,
)


class TestToNumpy:
    def test_numpy_passthrough(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _to_numpy(arr)
        np.testing.assert_array_equal(result, arr)

    def test_polars_dataframe(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_pandas_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _to_numpy([[1, 2], [3, 4]])


class TestTo1d:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_1d(arr)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)

    def test_polars_series(self):
        s = pl.Series([1.0, 2.0, 3.0])
        result = _to_1d(s)
        assert result.dtype == np.float64

    def test_python_list(self):
        result = _to_1d([1, 2, 3])
        assert result.dtype == np.float64
        assert len(result) == 3

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            _to_1d(np.array([[1, 2], [3, 4]]))


class TestClipHessians:
    def test_clips_below_eps(self):
        h = np.array([1e-10, 1e-5, 1.0])
        result = _clip_hessians(h, eps=1e-4)
        assert result[0] == 1e-4
        assert result[1] == 1e-4
        assert result[2] == 1.0

    def test_does_not_clip_above_eps(self):
        h = np.array([0.1, 0.5, 10.0])
        result = _clip_hessians(h, eps=1e-4)
        np.testing.assert_array_equal(result, h)

    def test_all_positive_after_clip(self):
        h = np.array([-1.0, 0.0, 1e-10, 1.0])
        result = _clip_hessians(h, eps=1e-4)
        assert np.all(result > 0)


class TestNormalizeGradients:
    def test_shape_preserved(self):
        g = np.array([1.0, 2.0, 3.0])
        h = np.array([0.1, 0.2, 0.3])
        g_norm, h_norm = _normalize_gradients(g, h, K=2)
        assert g_norm.shape == g.shape
        assert h_norm.shape == h.shape

    def test_scale_is_Kn(self):
        n = 4
        K = 3
        g = np.ones(n)
        h = np.ones(n)
        g_norm, h_norm = _normalize_gradients(g, h, K=K)
        expected_scale = K * n
        np.testing.assert_allclose(g_norm, 1.0 / expected_scale)
        np.testing.assert_allclose(h_norm, 1.0 / expected_scale)


class TestInputValidation:
    """Test that DistributionalGBM subclasses validate inputs correctly."""

    def test_negative_y_raises(self, tweedie_data):
        """Negative y should raise ValueError before any CatBoost call."""
        from insurance_distributional import TweedieGBM
        model = TweedieGBM()
        X = tweedie_data["X"]
        y_bad = tweedie_data["y"].copy()
        y_bad[0] = -1.0
        with pytest.raises(ValueError, match="Negative y"):
            model.fit(X, y_bad)

    def test_exposure_wrong_length_raises(self, tweedie_data):
        from insurance_distributional import TweedieGBM
        model = TweedieGBM()
        X = tweedie_data["X"]
        y = tweedie_data["y"]
        exposure_bad = np.ones(len(y) + 10)  # wrong length
        with pytest.raises(ValueError, match="exposure length"):
            model.fit(X, y, exposure=exposure_bad)

    def test_zero_exposure_raises(self, tweedie_data):
        from insurance_distributional import TweedieGBM
        model = TweedieGBM()
        X = tweedie_data["X"]
        y = tweedie_data["y"]
        exposure_bad = np.ones(len(y))
        exposure_bad[5] = 0.0
        with pytest.raises(ValueError, match="positive"):
            model.fit(X, y, exposure=exposure_bad)

    def test_predict_before_fit_raises(self):
        from insurance_distributional import TweedieGBM
        model = TweedieGBM()
        X = np.random.randn(10, 3)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_invalid_tweedie_power_raises(self):
        from insurance_distributional import TweedieGBM
        with pytest.raises(ValueError, match="power"):
            TweedieGBM(power=0.5)  # outside (1, 2)

        with pytest.raises(ValueError, match="power"):
            TweedieGBM(power=2.5)
