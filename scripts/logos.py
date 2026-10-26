"""Maker logos for the charts: docs/logos/<maker>.png, placed at a fixed height in points.

Sources are listed in docs/logos/SOURCES.md.
Logos are trademarks of their owners and are used only to say who made each model.
"""
from pathlib import Path

import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea, VPacker

ROOT = Path(__file__).resolve().parents[1]
MAKER = {
    "whisper-large-v3-turbo": "openai", "whisper-large-v3": "openai", "whisper-1": "openai",
    "gpt-transcribe": "openai", "gpt-4o-transcribe": "openai", "gpt-4o-mini-transcribe-nochunk": "openai",
    "gpt-4o-transcribe-diarize": "openai", "distil-large-v3.5": "huggingface",
    "parakeet-tdt-0.6b-v3": "nvidia", "canary-1b-v2": "nvidia", "Voxtral-Mini-3B": "mistral",
    "granite-speech-3.3-2b": "ibm", "SenseVoice-Small": "alibaba", "Qwen2.5-Omni-3B": "qwen",
    "MERaLiON-2-3B": "astar", "MERaLiON-2-10B-ASR": "astar",
    "mjwong-turbo-singlish": "user",  # a personal Hugging Face account, not a company
}
_cache = {}


def logo(model, height):
    """OffsetImage of the model's maker logo, `height` points tall; None for a model with no company behind it."""
    maker = "lepaq" if model.startswith("whisperlah") else MAKER.get(model)  # WhisperLah is built for Lepaq
    if maker is None:
        return None
    if maker not in _cache:
        _cache[maker] = mpimg.imread(ROOT / "docs" / "logos" / f"{maker}.png")
    img = _cache[maker]
    return OffsetImage(img, zoom=height / img.shape[0], dpi_cor=True)


def logo_label(ax, model, text, xy, height=11, fontsize=10, color="#5F5B54", weight="normal", stacked=False,
               xycoords=("data", "axes fraction"), box_alignment=(0.5, 1.0)):
    """Logo and name as one centred label below an axis: side by side, or the logo above the name."""
    name = TextArea(text, textprops=dict(fontsize=fontsize, color=color, fontweight=weight))
    img = logo(model, height)
    box = name if img is None else (VPacker if stacked else HPacker)(children=[img, name], align="center", pad=0,
                                                                       sep=3 if stacked else 4)
    ax.add_artist(AnnotationBbox(box, xy, xycoords=xycoords, box_alignment=box_alignment, frameon=False,
                                 annotation_clip=False))
