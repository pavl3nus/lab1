"""Собственные тесты для второй части лабораторной."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput,
    RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, plot_matrix_histograms, sum_prod,
    unique_columns, unique_rows,
)


def test_sum_prod_matches_explicit_product():
    rng = np.random.default_rng(42)
    A = rng.integers(-3, 4, size=(3, 2, 2))
    x = rng.integers(-3, 4, size=(3, 2, 1))
    expected = sum(A[i] @ x[i] for i in range(3))
    np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(A, x)), expected)


def test_sum_prod_output_shape():
    A = np.array([np.eye(3)])
    x = np.array([[[1], [2], [3]]])
    out = sum_prod(MatrixVectorBatchInput(A, x))
    assert out.shape == (3, 1)


def test_binarize_edges_and_no_mutation():
    src = np.array([[-1.0, 0.0, 2.0]])
    copied = src.copy()
    res = binarize(BinarizeInput(src, 0.0))
    np.testing.assert_array_equal(res, [[0, 0, 1]])
    np.testing.assert_array_equal(src, copied)


def test_unique_rows_columns():
    m = np.array([[4, 4, 1, 4], [2, 2, 2, 2]])
    assert unique_rows(MatrixInput(m)) == [[1, 4], [2]]
    assert unique_columns(MatrixInput(m)) == [[2, 4], [2, 4], [1, 2], [2, 4]]


def test_matrix_statistics_shapes_and_seed():
    stats_a = matrix_statistics(RandomMatrixInput(4, 6, mean=0.0, std=1.0, seed=11))
    stats_b = matrix_statistics(RandomMatrixInput(4, 6, mean=0.0, std=1.0, seed=11))
    np.testing.assert_array_equal(stats_a.matrix, stats_b.matrix)
    assert stats_a.row_means.shape == (4,)
    assert stats_a.column_means.shape == (6,)
    assert stats_a.row_variances.shape == (4,)
    assert stats_a.column_variances.shape == (6,)
    np.testing.assert_allclose(stats_a.matrix.mean(axis=0), stats_a.column_means)


def test_plot_returns_figure():
    stats = matrix_statistics(RandomMatrixInput(3, 3, seed=0))
    fig = plot_matrix_histograms(stats, bins=5, show=False)
    assert len(fig.axes) == 2


def test_chess_correctness():
    np.testing.assert_array_equal(
        chess(ChessInput(2, 3, 5, 8)), [[5, 8, 5], [8, 5, 8]])
    np.testing.assert_array_equal(chess(ChessInput(1, 1, 0, 1)), [[0]])


RED = (200, 30, 40)
GREY = (10, 10, 10)


def test_rectangle_smaller_than_canvas():
    img = draw_rectangle(RectangleInput(2, 2, 6, 8, RED, GREY))
    assert img.shape == (6, 8, 3)
    assert img.dtype == np.uint8
    mask = (img == np.array(RED, dtype=np.uint8)).all(axis=2)
    assert mask.sum() == 4


def test_rectangle_bigger_than_canvas_clipped():
    img = draw_rectangle(RectangleInput(99, 99, 5, 5, RED, GREY))
    assert (img == np.array(RED, dtype=np.uint8)).all()


def test_rectangle_zero_size_keeps_background():
    img = draw_rectangle(RectangleInput(0, 5, 4, 4, RED, GREY))
    assert (img == np.array(GREY, dtype=np.uint8)).all()


def test_ellipse_inside_and_outside():
    img = draw_ellipse(EllipseInput(2, 2, 7, 7, RED, GREY))
    mask = (img == np.array(RED, dtype=np.uint8)).all(axis=2)
    assert mask[3, 3]
    assert not mask[0, 0]


def test_ellipse_symmetry():
    img = draw_ellipse(EllipseInput(3, 2, 8, 10, RED, GREY))
    mask = (img == np.array(RED, dtype=np.uint8)).all(axis=2)
    np.testing.assert_array_equal(mask, mask[::-1, :])
    np.testing.assert_array_equal(mask, mask[:, ::-1])


def test_time_series_full_result():
    res = analyze_time_series(TimeSeriesInput([4, 1, 4, 1, 4], 2))
    np.testing.assert_array_equal(res.local_maxima_indices, [0 + 2])
    np.testing.assert_array_equal(res.local_minima_indices, [1, 3])
    np.testing.assert_allclose(res.moving_average, [2.5, 2.5, 2.5, 2.5])


def test_time_series_window_validation():
    with pytest.raises(ValueError):
        analyze_time_series(TimeSeriesInput([1, 2], 3))
    with pytest.raises(ValueError):
        analyze_time_series(TimeSeriesInput([1, 2], 0))


def test_one_hot_auto_and_explicit_count():
    np.testing.assert_array_equal(
        one_hot(OneHotInput([1, 0, 1])),
        [[0, 1], [1, 0], [0, 1]],
    )
    encoded = one_hot(OneHotInput([2], class_count=5))
    assert encoded.shape == (1, 5)
    np.testing.assert_array_equal(encoded, [[0, 0, 1, 0, 0]])


def test_one_hot_errors():
    with pytest.raises(ValueError):
        one_hot(OneHotInput([3], class_count=2))
    with pytest.raises(ValueError):
        one_hot(OneHotInput([-2, 0]))