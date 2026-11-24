from __future__ import annotations

import os

import numpy as np

from pyhfo.utils.utils_feature import compute_spectrum


def _uint8_image(values):
    from PIL import Image

    array = np.asarray(values, dtype=float).squeeze()
    if array.ndim != 2:
        array = np.atleast_2d(array)
    finite = np.isfinite(array)
    if not finite.any():
        scaled = np.zeros(array.shape, dtype=np.uint8)
    else:
        low = float(np.nanmin(array[finite]))
        high = float(np.nanmax(array[finite]))
        normalized = np.zeros(array.shape, dtype=float)
        if high > low:
            normalized[finite] = (array[finite] - low) / (high - low)
        scaled = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _draw_waveform(draw, bounds, values, *, highlight_start=None, highlight_end=None):
    x0, y0, x1, y1 = bounds
    trace = np.asarray(values, dtype=float).squeeze()
    if trace.ndim != 1:
        trace = trace.reshape(-1)
    if trace.size < 2:
        return
    trace = np.nan_to_num(trace, copy=False)
    low = float(trace.min())
    high = float(trace.max())
    span = high - low or 1.0
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    def point(index):
        x = x0 + (index / max(1, trace.size - 1)) * width
        y = y1 - ((float(trace[index]) - low) / span) * height
        return (round(x), round(y))

    draw.line([point(index) for index in range(trace.size)], fill=(25, 90, 190), width=2)
    if highlight_start is None or highlight_end is None:
        return
    first = max(0, min(trace.size - 1, int(highlight_start)))
    last = max(first + 1, min(trace.size, int(highlight_end)))
    if last - first >= 2:
        draw.line([point(index) for index in range(first, last)], fill=(210, 45, 45), width=3)


def _plot_feature_with_pillow(
    folder,
    feature,
    start,
    end,
    data,
    data_filtered,
    channel_name,
    biomarker_start,
    biomarker_end,
):
    from PIL import Image, ImageDraw

    width, height = 1800, 500
    margin, title_height, gap = 24, 54, 28
    panel_width = (width - margin * 2 - gap * 2) // 3
    panel_top = title_height + margin
    panel_bottom = height - margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{channel_name}_{start}_{end} with length: {(end - start) * 0.5} ms"
    draw.text((margin, 18), title, fill=(20, 20, 20))

    feature_image = _uint8_image(np.asarray(feature)[0])
    feature_image = feature_image.resize((panel_width, panel_bottom - panel_top))
    canvas.paste(feature_image, (margin, panel_top))

    for panel_index, values in enumerate((data, data_filtered), start=1):
        left = margin + panel_index * (panel_width + gap)
        bounds = (left, panel_top, left + panel_width, panel_bottom)
        draw.rectangle(bounds, outline=(180, 180, 180), width=1)
        _draw_waveform(
            draw,
            bounds,
            values,
            highlight_start=biomarker_start,
            highlight_end=biomarker_end,
        )

    output_path = os.path.join(folder, f"{channel_name}_{start}_{end}.jpg")
    canvas.save(output_path, format="JPEG", quality=90)


def plot_feature(
    folder,
    feature,
    start,
    end,
    data,
    data_filtered,
    channel_name,
    biomarker_start=None,
    biomarker_end=None,
    hfo_start=None,
    hfo_end=None,
):
    """Export the legacy three-panel feature preview.

    Source and full runtime installations keep the historical Matplotlib
    rendering. Minimal headless environments can still export with Pillow.
    """
    if biomarker_start is None:
        biomarker_start = hfo_start
    if biomarker_end is None:
        biomarker_end = hfo_end
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _plot_feature_with_pillow(
            folder,
            feature,
            start,
            end,
            data,
            data_filtered,
            channel_name,
            biomarker_start,
            biomarker_end,
        )
        return

    channel_data = np.squeeze(data)
    channel_data_f = np.squeeze(data_filtered)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.imshow(np.asarray(feature)[0])
    ax2.plot(channel_data, color="blue")
    ax2.plot(
        np.arange(biomarker_start, biomarker_end),
        channel_data[biomarker_start:biomarker_end],
        color="red",
    )
    ax3.plot(channel_data_f, color="blue")
    ax3.plot(
        np.arange(biomarker_start, biomarker_end),
        channel_data_f[biomarker_start:biomarker_end],
        color="red",
    )
    plt.suptitle(f"{channel_name}_{start}_{end} with length: {(end - start) * 0.5} ms")
    plt.savefig(os.path.join(folder, f"{channel_name}_{start}_{end}.jpg"))
    plt.close(fig)


def calculate_time_frequency(data, fs, freq_min=10, freq_max=500):
    return compute_spectrum(
        data,
        ps_SampleRate=fs,
        ps_MinFreqHz=freq_min,
        ps_MaxFreqHz=freq_max,
    )
