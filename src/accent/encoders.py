"""Frozen pretrained speech encoders. Every encoder returns all hidden states:
    hs   : [B, L, T, D]  (L = number of layers incl. the input embedding layer)
    mask : [B, T] bool   (valid frames)

Names (``--encoder``):
    xlsr300m, wavlm_large, hubert_large, wav2vec2_large, whisper_small, whisper_medium, ecapa
    hf:<hub id or local dir>        any wav2vec2 / HuBERT / WavLM style model (incl. a Wav2Vec2ForCTC dir)
    whisper:<hub id or local dir>   any Whisper model (encoder only)
    phonrec:<checkpoint file|dir>   a checkpoint of our phoneme-recognition repo (XLS-R encoder weights
                                    are taken from the state dict whatever the key prefix is)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger("accent")

ALIASES = {
    "xlsr300m": "hf:facebook/wav2vec2-xls-r-300m",
    "wavlm_large": "hf:microsoft/wavlm-large",
    "hubert_large": "hf:facebook/hubert-large-ll60k",
    "wav2vec2_large": "hf:facebook/wav2vec2-large-lv60",
    "whisper_small": "whisper:openai/whisper-small",
    "whisper_medium": "whisper:openai/whisper-medium",
    "ecapa": "ecapa:speechbrain/spkrec-ecapa-voxceleb",
}


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def normalize_wave(wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero mean / unit variance over the valid samples of every row (what the HF feature extractor
    does for these models); padding stays 0."""
    m = lengths_to_mask(lengths, wav.shape[1]).to(wav.dtype)
    n = m.sum(1, keepdim=True).clamp(min=1)
    mean = (wav * m).sum(1, keepdim=True) / n
    var = (((wav - mean) * m) ** 2).sum(1, keepdim=True) / n
    return (wav - mean) / torch.sqrt(var + 1e-7) * m


class SSLEncoder(nn.Module):
    """wav2vec2 / XLS-R / HuBERT / WavLM via transformers."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers + 1
        self.dim = cfg.hidden_size
        self.frame_rate = 50

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        wav = normalize_wave(wav, lengths)
        att = lengths_to_mask(lengths, wav.shape[1]).long()
        out = self.model(wav, attention_mask=att, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)  # [B, L, T, D]
        frames = self.model._get_feat_extract_output_lengths(lengths).clamp(max=hs.shape[2])
        return hs, lengths_to_mask(frames, hs.shape[2])


class WhisperEncoder(nn.Module):
    """Whisper encoder. Input is padded to 30 s (the model requires it); only frames covering the
    real audio (50 per second) are returned."""

    def __init__(self, name: str):
        super().__init__()
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.model = WhisperModel.from_pretrained(name).encoder
        self.num_layers = self.model.config.encoder_layers + 1
        self.dim = self.model.config.d_model
        self.frame_rate = 50

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        wav = normalize_wave(wav, lengths) * 0.05  # roughly speech-level amplitude
        arrays = [w[: int(n)].float().cpu().numpy() for w, n in zip(wav, lengths)]
        feats = self.fe(arrays, sampling_rate=16000, return_tensors="pt").input_features
        p = next(self.model.parameters())
        out = self.model(feats.to(p.device, p.dtype), output_hidden_states=True)
        frames = torch.div(lengths + 319, 320, rounding_mode="floor").clamp(min=1)
        t = int(frames.max())
        hs = torch.stack(out.hidden_states, dim=1)[:, :, :t]
        return hs, lengths_to_mask(frames.to(hs.device), t)


class EcapaEncoder(nn.Module):
    """SpeechBrain ECAPA-TDNN speaker embedding (192-d), returned as one layer with one frame."""

    def __init__(self, name: str, savedir: str):
        super().__init__()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older speechbrain
            from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source=name, savedir=savedir)
        self.num_layers, self.dim, self.frame_rate = 1, 192, 0

    def to(self, *a, **k):  # speechbrain keeps its own device attribute
        dev = next((x for x in a if isinstance(x, (str, torch.device))), k.get("device"))
        if dev is not None:
            self.model.device = str(dev)
            self.model.mods.to(dev)
        return self

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        rel = lengths.float() / wav.shape[1]
        emb = self.model.encode_batch(wav.float(), rel)  # [B, 1, 192]
        return emb[:, None], torch.ones(wav.shape[0], 1, dtype=torch.bool, device=wav.device)


def _load_phonrec(path: str) -> nn.Module:
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    p = Path(path)
    if p.is_dir() and (p / "config.json").exists():
        from transformers import AutoModel

        return AutoModel.from_pretrained(str(p))
    files = [p] if p.is_file() else sorted([*p.glob("*.safetensors"), *p.glob("*.pt"), *p.glob("*.bin"),
                                            *p.glob("*.pth")])
    if not files:
        raise FileNotFoundError(f"No checkpoint file in {p}")
    sd = {}
    for f in files:
        if f.suffix == ".safetensors":
            from safetensors.torch import load_file

            part = load_file(str(f))
        else:
            part = torch.load(str(f), map_location="cpu", weights_only=False)
        for key in ("model", "state_dict", "model_state_dict", "module"):
            if isinstance(part, dict) and key in part and isinstance(part[key], dict):
                part = part[key]
                break
        sd.update({k: v for k, v in part.items() if torch.is_tensor(v)})
    model = Wav2Vec2Model(Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m"))
    target = model.state_dict()
    mapped = {}
    for k, v in sd.items():
        parts = k.split(".")
        for i in range(len(parts)):
            cand = ".".join(parts[i:])
            if cand in target and target[cand].shape == v.shape:
                mapped[cand] = v
                break
    missing = [k for k in target if k not in mapped]
    if len(mapped) < 0.9 * len(target):
        raise RuntimeError(f"{path}: only {len(mapped)}/{len(target)} encoder tensors found; "
                           f"e.g. missing {missing[:5]}; checkpoint keys e.g. {list(sd)[:5]}")
    model.load_state_dict(mapped, strict=False)
    log.info("phonrec: loaded %d/%d encoder tensors from %s (missing: %s)", len(mapped), len(target), path,
             missing[:10])
    return model


def build_encoder(name: str, cache_dir: str = "pretrained") -> nn.Module:
    spec = ALIASES.get(name, name)
    kind, _, target = spec.partition(":")
    if kind == "hf":
        from transformers import AutoModel

        enc = SSLEncoder(AutoModel.from_pretrained(target))
    elif kind == "phonrec":
        enc = SSLEncoder(_load_phonrec(target))
    elif kind == "whisper":
        enc = WhisperEncoder(target)
    elif kind == "ecapa":
        enc = EcapaEncoder(target, str(Path(cache_dir) / "ecapa"))
    else:
        raise ValueError(f"Unknown encoder {name!r}; use one of {list(ALIASES)} or hf:/whisper:/phonrec:")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").strip("_")
