from __future__ import annotations

import matplotlib.pyplot as plt

from ramappy.core.pipeline import ParamsOutputFile, SingleModelParamsValidator, StepClass, StepParams, pipeline_step
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum


class StepPlotSpectraPixelsParams(StepParams, ParamsOutputFile):
    """Parameters for `plot_spectra`."""

    idx: int | list[int] | tuple[int] = 0
    """Index or list of indices of the pixels to plot."""

    figsize: tuple[int, int] = (16, 8)
    """Figure size for the plot (default: ``(16, 8)``)."""


@pipeline_step(
    step_name="plot_spectra_pixels",
    params_validator=SingleModelParamsValidator(StepPlotSpectraPixelsParams),
    friendly_name="Plot Pixel Spectra",
    step_category=StepClass.VISUALIZATION,
)
def plot_spectra(
    spectral_map: SpectralMap | Spectrum,
    idx: int | list[int] | tuple[int] = 0,
    out_file: str | None = None,
    figsize: tuple[int, int] = (16, 8),
):
    """Plot the spectra of one or more pixels and save the figure.

    Parameters
    ----------
    spectral_map : SpectralMap or Spectrum
        Data container providing the spectra.
    idx : int or list of int or tuple of int, optional
        Flat pixel index or indices to plot. Defaults to ``0``.
    out_file : str or None, optional
        Output file path. Defaults to ``<map_name>_plot.pdf``.
    figsize : tuple of int, optional
        Figure dimensions ``(width, height)`` in inches.
    """
    plt.figure(figsize=figsize)
    plt.plot(spectral_map.x, spectral_map.data[idx, :].T)
    plt.savefig(out_file or f"{spectral_map.name}_plot.pdf")
    plt.close()


class StepPlotGroupSpectraParams(StepParams, ParamsOutputFile):
    """Parameters for `plot_group_spectra`."""

    figsize: tuple[int, int] = (16, 8)
    """Figure size for the plot (default: ``(16, 8)``)."""

    image_group_id: str | None = None
    """Identifier of the image group to plot spectra from."""

    mask_group_id: str | None = None
    """Identifier of the mask group to plot spectra from."""

    x_label: str | None = None
    """Label for the x-axis (default: ``None``, which uses the x-axis unit of the last spectrum plotted)."""

    y_label: str | None = None
    """Label for the y-axis (default: ``None``, which uses the data unit of the last spectrum plotted)."""

    title: str | None = None
    """Title for the plot (default: ``None``, which uses an empty title)."""


@pipeline_step(
    step_name="plot_group_spectra",
    params_validator=SingleModelParamsValidator(StepPlotGroupSpectraParams),
    friendly_name="Plot Group Spectra",
    step_category=StepClass.VISUALIZATION,
)
def plot_group_spectra(
    spectral_map: SpectralMap,
    *,
    image_group_id: str | None = None,
    mask_group_id: str | None = None,
    out_file: str | None = None,
    figsize: tuple[int, int] = (16, 8),
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
):
    """Plot all spectra in an image or mask group and save the figure.

    Exactly one of *image_group_id* or *mask_group_id* must be provided.

    Parameters
    ----------
    spectral_map : SpectralMap
        Data container providing the spectra.
    image_group_id : str or None, optional
        Identifier of the image group to plot.
    mask_group_id : str or None, optional
        Identifier of the mask group to plot.
    out_file : str or None, optional
        Output file path. Defaults to ``<map_name>_group_plot.svg``.
    figsize : tuple of int, optional
        Figure dimensions ``(width, height)`` in inches.
    x_label : str or None, optional
        X-axis label. Defaults to the spectral unit of the last plotted spectrum.
    y_label : str or None, optional
        Y-axis label. Defaults to the intensity unit of the last plotted spectrum.
    title : str or None, optional
        Plot title.

    Raises
    ------
    ValueError
        If both or neither of *image_group_id* and *mask_group_id* are provided.
    KeyError
        If the specified group is not found in the map.
    """
    plt.figure(figsize=figsize)
    if not ((image_group_id is not None) ^ (mask_group_id is not None)):
        raise ValueError("Image_group_id or mask_group_id must be provided, not both.")

    last_spectrum = None

    if image_group_id is not None:
        if image_group_id not in spectral_map.images_group:
            raise KeyError(f"Group {image_group_id} not found in images group.")
        for img_key in reversed(spectral_map.images_group[image_group_id].elem_ids or []):
            img = spectral_map.images[img_key]
            s = img.spectrum
            if s is None:
                raise ValueError(f"Image {img_key} does not have a spectrum.")
            last_spectrum = s
            plt.plot(s.x, s.data.T, label=img.name or img_key)
    else:
        assert mask_group_id is not None
        if mask_group_id not in spectral_map.masks_group:
            raise KeyError(f"Group {mask_group_id} not found in masks group.")
        for mask_key in reversed(spectral_map.masks_group[mask_group_id].elem_ids or []):
            mask = spectral_map.masks[mask_key]
            s = spectral_map.get_spectrum(
                mask=mask_key,
                agg="mean",
                prefer_stored_property=mask.spectrum_agg == "centroid",
            )
            if s is None:
                raise ValueError(f"Mask {mask_key} does not have a spectrum.")
            last_spectrum = s
            plt.plot(s.x, s.data.T, label=s.name or mask_key)

    plt.legend()

    plt.ylabel(
        y_label
        or (str(last_spectrum.data_unit) if last_spectrum is not None and last_spectrum.data_unit is not None else "")
    )
    plt.xlabel(
        x_label
        or (
            str(last_spectrum.x_axis_unit)
            if last_spectrum is not None and last_spectrum.x_axis_unit is not None
            else ""
        )
    )
    plt.title(title or "")

    plt.savefig(out_file or f"{spectral_map.name}_group_plot.svg")
    plt.close()


__all__ = ["StepPlotGroupSpectraParams", "StepPlotSpectraPixelsParams", "plot_group_spectra", "plot_spectra"]
