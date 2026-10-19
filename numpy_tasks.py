"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    
    if vectors.ndim == 2:
        vectors = vectors[:, :, np.newaxis]
        
    return np.matmul(matrices, vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    return np.where(np.asarray(data.matrix) > data.threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(loc=data.mean, scale=data.std, size=(data.rows, data.columns))
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(stats: MatrixStatistics, bins: int = 20, show: bool = True):
    import matplotlib.pyplot as plt

    matrix = np.asarray(stats.matrix)
    fig, (ax_rows, ax_cols) = plt.subplots(1, 2, figsize=(12, 4))

    for i, row in enumerate(matrix):
        ax_rows.hist(row, bins=bins, alpha=0.4, label=f"строка {i}" if matrix.shape[0] <= 5 else None)
    ax_rows.set_title("Гистограммы значений по строкам")
    ax_rows.set_xlabel("Значение")
    ax_rows.set_ylabel("Частота")

    for j, column in enumerate(matrix.T):
        ax_cols.hist(column, bins=bins, alpha=0.4, label=f"столбец {j}" if matrix.shape[1] <= 5 else None)
    ax_cols.set_title("Гистограммы значений по столбцам")
    ax_cols.set_xlabel("Значение")
    ax_cols.set_ylabel("Частота")

    for ax in (ax_rows, ax_cols):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def chess(data: ChessInput) -> np.ndarray:
    i, j = np.indices((data.rows, data.columns))
    return np.where((i + j) % 2 == 0, data.first, data.second)


def _blank_image(height: int, width: int, color) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.array(color, dtype=np.uint8)
    return canvas


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    picture = _blank_image(data.image_height, data.image_width, data.background_color)
    
    if data.height <= 0 or data.width <= 0:
        return picture
        
    top = max((data.image_height - data.height) // 2, 0)
    left = max((data.image_width - data.width) // 2, 0)
    bottom = min(top + data.height, data.image_height)
    right = min(left + data.width, data.image_width)
    
    picture[top:bottom, left:right] = np.array(data.shape_color, dtype=np.uint8)
    return picture


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    picture = _blank_image(data.image_height, data.image_width, data.background_color)
    
    a, b = data.semi_axis_x, data.semi_axis_y
    if a <= 0 or b <= 0:
        return picture
        
    center_x = (data.image_width - 1) / 2
    center_y = (data.image_height - 1) / 2
    
    y, x = np.ogrid[:data.image_height, :data.image_width]
    inside_ellipse = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1
    
    picture[inside_ellipse] = np.array(data.shape_color, dtype=np.uint8)
    return picture


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    series = np.asarray(data.values, dtype=float)
    window = data.window
    
    if series.ndim != 1:
        raise ValueError("Временной ряд должен быть одномерным!")
    if not 1 <= window <= len(series):
        raise ValueError("Размер окна должен быть от 1 до длины ряда!")
    
    maxima = np.flatnonzero((series[1:-1] > series[:-2]) & (series[1:-1] > series[2:])) + 1
    minima = np.flatnonzero((series[1:-1] < series[:-2]) & (series[1:-1] < series[2:])) + 1

    from numpy.lib.stride_tricks import sliding_window_view
    moving_averages = sliding_window_view(series, window).mean(axis=1)

    return TimeSeriesStatistics(
        mean=float(series.mean()),
        variance=float(series.var()),
        std=float(series.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_averages,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    
    if labels.size and labels.min() < 0:
        raise ValueError("Метки классов должны быть неотрицательными!")
        
    class_count = data.class_count
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    elif labels.size and labels.max() >= class_count:
        raise ValueError("Метка класса выходит за пределы class_count!")

    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result