"""Composable plotting helpers for laboratory measurements."""

from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_hex, to_rgba
from matplotlib.figure import Figure

from .electronics import gain_to_db
from .fitting import ModelFit, Model, fit_model, fit_statistics
from .measurement import unpack_measurements
from .methods import Errors, PlotErrors, _data, polynomial_fit
from .styling import PathLike, add_colorbar, finish_plot, get_cmap, style_axis

Size = Tuple[float, float]
Scale = Optional[Literal["linear", "log", "symlog", "logit"]]


def _plot_errors(errors: PlotErrors, size: int, name: str) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    if values.ndim == 0:
        values = np.full(size, values.item())
    if values.shape not in ((size,), (2, size)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar, length-N, or 2-by-N sequence")
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative")
    return values


def _axes(ax: Optional[Axes], figsize: Size) -> Tuple[Figure, Axes]:
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _series_colors(ax: Axes, color: Optional[str], fit_color: Optional[str]) -> Tuple[str, str]:
    """Data colour (next in the axes cycle if unset) and a paler fit colour of the same hue."""
    if color is None:
        color = ax._get_lines.get_next_color()
    if fit_color is None:
        r, g, b, _ = to_rgba(color)
        fit_color = to_hex((1 - (1 - r) * 0.5, 1 - (1 - g) * 0.5, 1 - (1 - b) * 0.5))
    return color, fit_color


def _apply_scales(ax: Axes, xscale: Scale, yscale: Scale) -> None:
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)


def annotate_fit(ax: Axes, fit: Any, color: Optional[str] = None) -> None:
    """Add a text box with the fit parameters (model fits) and χ², reduced χ², p and R²."""
    lines = []
    if isinstance(fit, ModelFit):
        lines += [f"{name} = {parameter.to_string(uncertainty_digits=2)}"
                  for name, parameter in fit.parameters.items()]
    lines += fit_statistics(fit)
    if not lines:
        return
    ax.text(0.02, 0.98 - 0.25 * len(ax.texts), "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color=color,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})


def _finish_if_requested(
    fig: Figure, filename: Optional[PathLike], save_directory: Optional[PathLike], show: bool
) -> Figure:
    if filename is not None or show:
        return finish_plot(fig, filename, save_directory, show)
    return fig


def line_plot(
    y: Sequence[float],
    x: Sequence[float] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[PathLike] = None,
    save_directory: Optional[PathLike] = None,
    show: bool = False,
    figsize: Size = (8, 5),
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    linestyle: str = "-",
    xscale: Scale = None,
    yscale: Scale = None,
) -> Figure:
    """Draw a line on ax (or a new figure), and return the figure."""
    if x is None:
        x = range(1, y.value.size + 1)
    x_values, y_values = _data(x, y)
    fig, ax = _axes(ax, figsize)
    ax.plot(x_values, y_values, label=label, color=color, marker=marker, linestyle=linestyle)
    _apply_scales(ax, xscale, yscale)
    style_axis(ax, title, xlabel, ylabel)
    return _finish_if_requested(fig, filename, save_directory, show)


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    c: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[PathLike] = None,
    save_directory: Optional[PathLike] = None,
    show: bool = False,
    figsize: Size = (8, 5),
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: str = "o",
    cmap: Union[str, Colormap, None] = None,
    colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xscale: Scale = None,
    yscale: Scale = None,
) -> Figure:
    """Scatter points, optionally coloured by ``c`` with a colormap and colourbar."""
    x_values, y_values = _data(x, y)
    if c is None:
        if colorbar or cmap is not None:
            raise ValueError("c is required when cmap or colorbar is used")
        c_values = None
    else:
        c_values = np.asarray(c, dtype=float)
        if c_values.shape != x_values.shape:
            raise ValueError("c must have the same length as x and y")
    fig, ax = _axes(ax, figsize)
    if c_values is None:
        ax.scatter(x_values, y_values, label=label, color=color, marker=marker)
    else:
        mappable = ax.scatter(x_values, y_values, c=c_values, cmap=get_cmap(cmap), marker=marker,
                              vmin=vmin, vmax=vmax, label=label)
        if colorbar:
            add_colorbar(fig, ax, mappable, colorbar_label)
    _apply_scales(ax, xscale, yscale)
    style_axis(ax, title, xlabel, ylabel)
    return _finish_if_requested(fig, filename, save_directory, show)


def scatter_with_fit(
    x: Sequence[float],
    y: Sequence[float],
    degree: int = 1,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[PathLike] = None,
    save_directory: Optional[PathLike] = None,
    show: bool = False,
    label_slope: bool = True,
    label_intercept: bool = True,
    figsize: Size = (8, 5),
    *,
    ax: Optional[Axes] = None,
    yerr: Optional[Errors] = None,
    xerr: Optional[PlotErrors] = None,
    use_errors: bool = True,
    model: Optional[Model] = None,
    p0: Optional[Sequence[float]] = None,
    show_stats: bool = False,
    label: str = "data",
    color: Optional[str] = None,
    fit_color: Optional[str] = None,
    marker: str = "o",
    xscale: Scale = None,
    yscale: Scale = None,
) -> Figure:
    """Plot measurements and a fit (polynomial, or ``model`` when given).

    Uncertainties carried by Measurement inputs are drawn as error bars and weight the fit
    unless ``use_errors`` is False; explicit ``yerr``/``xerr`` always apply. x errors are
    drawn but do not enter the fit. ``show_stats`` adds a box of parameters and χ².
    """
    fig, _ = _scatter_fit(
        x, y, degree, title, xlabel, ylabel, label_slope, label_intercept, figsize, ax, yerr,
        xerr, use_errors, model, p0, show_stats, label, color, fit_color, marker, xscale, yscale,
    )
    return _finish_if_requested(fig, filename, save_directory, show)


def _scatter_fit(
    x, y, degree, title, xlabel, ylabel, label_slope, label_intercept, figsize, ax, yerr, xerr,
    use_errors, model, p0, show_stats, label, color, fit_color, marker, xscale, yscale,
):
    """Draw data and fit on ``ax``; return the figure and the fit result."""
    if use_errors:
        if yerr is None:
            _, yerr = unpack_measurements(y)
        if xerr is None:
            _, xerr = unpack_measurements(x)
    x_values, y_values = _data(x, y)
    if model is None:
        fit = polynomial_fit(x_values, y_values, degree, yerr)
    else:
        fit = fit_model(model, x_values, y_values, yerr, p0)
    plot_yerr = None if yerr is None else _plot_errors(yerr, x_values.size, "yerr")
    plot_xerr = None if xerr is None else _plot_errors(xerr, x_values.size, "xerr")
    fig, ax = _axes(ax, figsize)
    _apply_scales(ax, xscale, yscale)
    color, fit_color = _series_colors(ax, color, fit_color)
    if plot_yerr is None and plot_xerr is None:
        ax.scatter(x_values, y_values, label=label, color=color, marker=marker)
    else:
        ax.errorbar(x_values, y_values, yerr=plot_yerr, xerr=plot_xerr, fmt=marker, color=color,
                    label=label, capsize=3)
    low, high = x_values.min(), x_values.max()
    if ax.get_xscale() == "log" and low > 0:
        fit_x = np.geomspace(low, high, 200)
    else:
        fit_x = np.linspace(low, high, 200)
    if model is not None:
        fit_y, fit_label = fit.predict(fit_x), "fit"
    else:
        fit_y = np.polyval(fit.coefficients, fit_x)
        if label_slope:
            fit_label = f"fit (y = {fit.coefficients[-2]:.3g}x"
            fit_label += f" + {fit.coefficients[-1]:.3g})" if label_intercept else ")"
        else:
            fit_label = f"fit (degree {degree})"
    ax.plot(fit_x, fit_y, color=fit_color, label=fit_label)
    if show_stats:
        annotate_fit(ax, fit, color)
    ax.legend()
    style_axis(ax, title, xlabel, ylabel)
    return fig, fit


def errorbar_plot(
    x: Sequence[float],
    y: Sequence[float],
    yerr: Optional[PlotErrors] = None,
    xerr: Optional[PlotErrors] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[PathLike] = None,
    save_directory: Optional[PathLike] = None,
    show: bool = False,
    figsize: Size = (8, 5),
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: str = "o",
    capsize: float = 3,
    xscale: Scale = None,
    yscale: Scale = None,
) -> Figure:
    """Plot measurements with optional symmetric or asymmetric uncertainties."""
    if yerr is None:
        _, yerr = unpack_measurements(y)
    if xerr is None:
        _, xerr = unpack_measurements(x)
    x_values, y_values = _data(x, y)
    plot_yerr = None if yerr is None else _plot_errors(yerr, x_values.size, "yerr")
    plot_xerr = None if xerr is None else _plot_errors(xerr, x_values.size, "xerr")
    fig, ax = _axes(ax, figsize)
    ax.errorbar(x_values, y_values, yerr=plot_yerr, xerr=plot_xerr, fmt=marker, capsize=capsize,
                label=label, color=color)
    _apply_scales(ax, xscale, yscale)
    style_axis(ax, title, xlabel, ylabel)
    return _finish_if_requested(fig, filename, save_directory, show)


def scatter_with_residuals(
    x: Sequence[float],
    y: Sequence[float],
    degree: int = 1,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[PathLike] = None,
    save_directory: Optional[PathLike] = None,
    show: bool = False,
    label_slope: bool = True,
    figsize: Size = (8, 7),
    *,
    axes: Optional[Tuple[Axes, Axes]] = None,
    yerr: Optional[Errors] = None,
    xerr: Optional[PlotErrors] = None,
    use_errors: bool = True,
    model: Optional[Model] = None,
    p0: Optional[Sequence[float]] = None,
    show_stats: bool = False,
    label: str = "data",
    color: Optional[str] = None,
    fit_color: Optional[str] = None,
    marker: str = "o",
    xscale: Scale = None,
    yscale: Scale = None,
) -> Figure:
    """Plot a fit (polynomial, or ``model``) and its residuals (observed minus predicted).

    Error handling, ``model``, ``show_stats`` and scales are as for
    :func:`scatter_with_fit`; ``yscale`` applies to the data axes only.
    """
    if axes is None:
        fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[3, 1])
    else:
        ax, residual_ax = axes
        if ax is residual_ax or ax.figure is not residual_ax.figure:
            raise ValueError("axes must be distinct and belong to the same figure")
        fig = ax.figure
    if use_errors:
        if yerr is None:
            _, yerr = unpack_measurements(y)
        if xerr is None:
            _, xerr = unpack_measurements(x)
    x_values, y_values = _data(x, y)
    color, fit_color = _series_colors(ax, color, fit_color)
    _, fit = _scatter_fit(
        x_values, y_values, degree, title, None, ylabel, label_slope, True, figsize, ax, yerr,
        xerr, False, model, p0, show_stats, label, color, fit_color, marker, xscale, yscale,
    )
    if xscale is not None:
        residual_ax.set_xscale(xscale)
    if yerr is None:
        residual_ax.scatter(x_values, fit.residuals, color=color, marker=marker)
    else:
        residual_ax.errorbar(x_values, fit.residuals, yerr=_plot_errors(yerr, x_values.size, "yerr"),
                             fmt=marker, color=color, capsize=3)
    residual_ax.axhline(0, color="k", linestyle="--")
    style_axis(residual_ax, "Residuals", xlabel, f"Residual ({ylabel})" if ylabel else "Residual")
    return _finish_if_requested(fig, filename, save_directory, show)


def bode_plot(
    frequency: Sequence[float],
    gain: Optional[Sequence[float]] = None,
    phase: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    filename: Optional[PathLike] = None,
    save_directory: Optional[PathLike] = None,
    show: bool = False,
    figsize: Size = (8, 5),
    *,
    gain_db: Optional[Sequence[float]] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Frequency (Hz)",
    gain_label: str = "Gain (dB)",
    phase_label: str = "Phase (°)",
    label: Optional[str] = None,
    color: Optional[str] = None,
    phase_color: Optional[str] = None,
    marker: str = "o",
    linestyle: str = "-",
    xscale: Scale = "log",
) -> Figure:
    """Bode plot: gain in dB against frequency, with phase in degrees on a twin axis.

    Give the gain as a linear amplitude ratio (``gain``, converted with 20·log10) or
    already in dB (``gain_db``). Measurement uncertainties are propagated and drawn as
    error bars. The frequency axis is logarithmic by default; the phase axis is
    ``fig.axes[1]`` when ``phase`` is supplied.
    """
    if (gain is None) == (gain_db is None):
        raise ValueError("give exactly one of gain (linear ratio) or gain_db")
    if gain is not None:
        converted = gain_to_db(gain)
        magnitude, magnitude_err = converted.value, converted.uncertainty
    else:
        magnitude, magnitude_err = unpack_measurements(gain_db)
    f_values, magnitude_values = _data(frequency, magnitude)
    fig, ax = _axes(ax, figsize)
    color = color or ax._get_lines.get_next_color()
    ax.errorbar(f_values, magnitude_values, yerr=magnitude_err, fmt=marker, linestyle=linestyle,
                color=color, label=label, capsize=3)
    _apply_scales(ax, xscale, None)
    style_axis(ax, title, xlabel, gain_label)
    if phase is not None:
        phase_values, phase_err = unpack_measurements(phase)
        _, phase_values = _data(f_values, phase_values)
        phase_color = phase_color or ax._get_lines.get_next_color()
        twin = ax.twinx()
        twin.grid(False)
        twin.errorbar(f_values, phase_values, yerr=phase_err, fmt=marker, linestyle=linestyle,
                      color=phase_color, capsize=3)
        twin.set_ylabel(phase_label, color=phase_color)
        ax.set_ylabel(gain_label, color=color)
        twin.tick_params(axis="y", colors=phase_color)
        ax.tick_params(axis="y", colors=color)
    if label is not None:
        ax.legend()
    return _finish_if_requested(fig, filename, save_directory, show)