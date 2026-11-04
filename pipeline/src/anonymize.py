#!/usr/bin/env python
"""
Stage 2 - Voice anonymization with Seed-VC (zero-shot VC, arXiv 2411.09943), timed per file.

Runs inside the `seedvc` conda env, with the Seed-VC git checkout given by
--seedvc-root. Models are loaded ONCE (via seed-vc's own inference.load_models),
the reference (pseudo) voices are pre-encoded once, and every speech segment
is converted with the same chunked DiT/vocoder loop as seed-vc/inference.py.

Modes
  --mode diarized : doctor-patient audio. Reads <id>.rttm from --rttm-dir, converts
                    each speaker's speech to a different pseudo-voice, re-assembles
                    the conversation on the original timeline.
                    --granularity segment : one VC call per (merged) segment
                    --granularity speaker : concatenate all speech of a speaker, one
                                            VC call per speaker, split back afterwards
  --mode whole    : doctor-dictation audio. Whole file -> one pseudo-voice, no RTTM.
  --groups CSV    : (id,group[,num_speakers]) files of one group are the same people recorded several times:
                    every person gets ONE target voice for the whole group. Pass 1 finds the speakers of all the
                    group's files and links the same person across files by CAM++ timbre; the target is then
                    chosen once per person with --ref-strategy; pass 2 converts. num_speakers is that file's
                    speaker cap. Written to <log-dir>/anonymize_<tag>_group_refs.json. (src/anonymize_dataset.py
                    builds the CSV from a dataset/<group>/*.wav folder tree.)

Per file a JSON record with sub-stage timings (whisper semantic encoder, length
regulator, DiT flow-matching, BigVGAN vocoder, resampling/assembly, I/O), GPU peak
memory and real-time factors is appended to <log-dir>/anonymize_<tag>_records.jsonl.
Per-segment timings go to anonymize_<tag>_segments.jsonl.
"""
import os
import sys
import json
import time
import math
import hashlib
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (Timer, JsonlWriter, env_info, gpu_peak_mem_mb, gpu_reset_peak, list_wavs,
                    wav_duration_s, log, write_json, now_iso, cuda_sync, read_rttm, find_rttm,
                    merge_segments, union_seconds)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default=None,
                   help="pipeline settings file (configs/pipelines/<name>.env: checkpoint, config, F0, target rule, speaker cap, "
                        "steps, guidance, prompt, context). Flags given explicitly on the command line override it. "
                        "${PIPE}, ${BENCH_ROOT}, ${MODELS_DIR} inside the file are expanded (defaults: project root, project root, "
                        "$BENCH_ROOT/models; set the environment variables or --models-dir to point elsewhere)")
    p.add_argument("--models-dir", default=None, help="value of ${MODELS_DIR} for --profile (default $BENCH_ROOT/models)")
    p.add_argument("--seedvc-root", required=True, help="path of the Plachtaa/seed-vc checkout")
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--out-dir", required=True, help="anonymized wavs are written here (16 kHz mono PCM16)")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--tag", required=True, help="run tag, e.g. diarized_segment_steps25")
    p.add_argument("--mode", choices=["diarized", "whole"], default="diarized")
    p.add_argument("--rttm-dir", default=None, help="RTTMs for --mode diarized (stage-1 output or reference RTTMs)")
    p.add_argument("--granularity", choices=["segment", "speaker"], default="segment")
    p.add_argument("--ref-dir", default=None,
                   help="dir of pseudo-speaker reference wavs (default <seedvc-root>/examples/reference). May hold one "
                        "sub-folder per language code (hi/, ta/, ..., neutral/): with --lang the pool of the file's "
                        "language is used, else neutral/, else every wav under the dir")
    p.add_argument("--lang", default="none",
                   help="none: ignore language (flat pool). auto: Whisper language-ID on the first 30 s of speech of each "
                        "file. Or a fixed code (hi, ta, te, bn, mr, gu, kn, ml, pa, ur, ...) e.g. from the hospital's registration")
    p.add_argument("--lid-model", default="openai/whisper-small", help="Whisper checkpoint for --lang auto (already staged for V1)")
    p.add_argument("--ref-strategy", choices=["rotate", "fixed", "pitch", "pitch_far", "cross", "timbre_far"], default="rotate",
                   help="rotate: deterministic per-file choice of pseudo-voices; fixed: spk-rank k -> ref k; "
                        "pitch: measure each speaker's median F0 -> pitch class (low <165 Hz = male, high = female) and pick "
                        "the unused reference of the same GENDER (reference gender from the pool's manifest.json, else the "
                        "m_/f_ filename prefix, else its own pitch class) with the nearest F0; never a clip cut from this file. "
                        "pitch_far: same gender, FARTHEST F0 (pushes the voice away from the source while keeping gender); "
                        "cross: OPPOSITE gender, farthest F0 (maximum distance; gender is not preserved). "
                        "timbre_far: same pitch class on BOTH sides (speaker and reference on the same side of 165 Hz, so the "
                        "perceived gender is kept) and, among those, the reference whose CAM++ speaker embedding is LEAST similar "
                        "to the speaker's own (pushes the identity, not the pitch)")
    p.add_argument("--ref-exclude", default="azuma,teio,dingzhen,trump",
                   help="comma-separated substrings; reference files whose name contains one are not used "
                        "(default drops the anime/celebrity clips bundled with seed-vc)")
    p.add_argument("--loudness", choices=["none", "match"], default="none",
                   help="match: scale every converted segment to the RMS of the source segment it replaces")
    p.add_argument("--attenuate-db", type=float, default=20.0, help="for --non-speech attenuate: background level drop")
    p.add_argument("--f0-condition", type=lambda s: s.lower() in ("1", "true", "yes"), default=False,
                   help="use seed-vc's F0-conditioned 44.1 kHz model (explicit pitch contour from RMVPE); slower, "
                        "preserves intonation, and with --auto-f0-adjust moves the pitch level to the reference voice")
    p.add_argument("--auto-f0-adjust", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--f0-shift", type=int, default=0, help="extra pitch shift in semitones (F0 mode only)")
    p.add_argument("--ref-rank", type=int, default=0,
                   help="take the k-th choice of the reference strategy instead of the first (0 = the usual pick); "
                        "runs with 0, 1, 2, ... give the same speaker different target voices of the same gender")
    p.add_argument("--ref-max-sec", type=float, default=25.0,
                   help="seconds of reference used as timbre prompt (upstream default 25, the maximum). A shorter "
                        "prompt speeds up long inputs but the README warns it 'may reduce similarity to prompt speech'.")
    p.add_argument("--diffusion-steps", type=int, default=25)
    p.add_argument("--cfg-rate", type=float, default=0.7)
    p.add_argument("--whole-chunk-sec", type=float, default=30.0,
                   help="--mode whole: convert the file in chunks of this length (each with --context-sec on both sides) "
                        "instead of one call; RMVPE/cuDNN fail on multi-minute inputs and the DiT windows are 30 s anyway")
    p.add_argument("--max-speakers", type=int, default=0,
                   help="diarized mode: if the RTTM has more speakers than this, the surplus clusters (least speech first) are "
                        "merged into the remaining speaker whose CAM++ embedding is closest (F0 if no embedding), so one real "
                        "person never gets two target voices. 0 = no cap")
    p.add_argument("--min-speaker-sec", type=float, default=2.0,
                   help="diarized mode: a speaker with less total speech than this (a clustering blip) is merged into the "
                        "speaker with the most speech instead of getting its own target voice")
    p.add_argument("--context-sec", type=float, default=0.0,
                   help="per-segment mode: convert each turn together with this many seconds of surrounding audio on "
                        "each side, then keep only the turn. Mirrors the 'extra context' the repo's real-time mode uses "
                        "for stability; 1.7 s turns alone are at the bottom of the model's 1-30 s range.")
    p.add_argument("--model", choices=["v1", "v2"], default="v1",
                   help="v1: seed-uvit-whisper-small-wavenet (voice cloning, the paper). "
                        "v2: hubert-bsqvae-small, 'best in suppressing source speaker traits', with anonymization-only mode")
    p.add_argument("--v2-anonymization-only", type=lambda s: s.lower() in ("1", "true", "yes"), default=True,
                   help="v2: ignore the reference identity and convert to the model's speaker-unconditional 'average voice' "
                        "(reference mel and speaker embedding are zeroed inside the CFM solver). All speakers of a file "
                        "then share one voice; set false for reference-guided conversion with distinct pseudo-voices.")
    p.add_argument("--v2-convert-style", type=lambda s: s.lower() in ("1", "true", "yes"), default=False,
                   help="v2: also run the AR model that converts accent/emotion (slower, changes delivery). Off keeps "
                        "the speaker's own prosody, which is what medical audio needs.")
    p.add_argument("--v2-intelligibility-cfg", type=float, default=0.7)
    p.add_argument("--v2-similarity-cfg", type=float, default=0.7)
    p.add_argument("--v2-top-p", type=float, default=0.9)
    p.add_argument("--v2-temperature", type=float, default=1.0)
    p.add_argument("--v2-repetition-penalty", type=float, default=1.0)
    p.add_argument("--v2-cfm-checkpoint", default=None, help="v2: custom/fine-tuned CFM checkpoint (default: HF v2/cfm_small.pth)")
    p.add_argument("--v2-ar-checkpoint", default=None, help="v2: custom AR checkpoint (default: HF v2/ar_base.pth)")
    p.add_argument("--compile", type=lambda s: s.lower() in ("1", "true", "yes"), default=False,
                   help="v2 with convert-style: torch.compile the AR decoder (~6x faster AR per README; needs triton)")
    p.add_argument("--length-adjust", type=float, default=1.0)
    p.add_argument("--fp16", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--checkpoint", default=None, help="custom DiT checkpoint (default: seed-uvit-whisper-small-wavenet)")
    p.add_argument("--config", default=None)
    p.add_argument("--merge-gap", type=float, default=0.3, help="merge same-speaker segments closer than this (s)")
    p.add_argument("--pad", type=float, default=0.1, help="pad each merged segment by this many seconds")
    p.add_argument("--min-seg", type=float, default=0.5, help="grow segments shorter than this (s)")
    p.add_argument("--non-speech", choices=["mute", "keep", "attenuate"], default="mute",
                   help="what to put where nobody speaks: silence (safest), original audio, or original audio "
                        "lowered by --attenuate-db (sounds less chopped, still hides background speech)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, help="i/N")
    p.add_argument("--files", default=None)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--groups", default=None,
                   help="CSV id,group[,num_speakers]: one target voice per person per group (see the module docstring)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Seed-VC bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_seedvc(root):
    """Must run before importing torch/transformers: seed-vc resolves its checkpoint
    cache relative to CWD and sets HF_HUB_CACHE at import time."""
    root = Path(root).resolve()
    if not (root / "inference.py").exists():
        sys.exit(f"--seedvc-root {root} does not contain inference.py")
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["HF_HUB_CACHE"] = str(root / "checkpoints" / "hf_cache")
    import torch
    _orig = torch.load

    def _load(*a, **k):  # trusted checkpoints; torch>=2.6 defaults to weights_only=True
        k["weights_only"] = False
        return _orig(*a, **k)

    torch.load = _load
    os.environ.setdefault("TQDM_DISABLE", "1")   # seed-vc wraps the diffusion steps in tqdm: 1 bar per segment
    import inference as svc  # seed-vc/inference.py
    try:  # silence the per-call "pass the sampling_rate argument" reminder from the whisper feature extractor
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    return svc


class SeedVCEngine:
    def __init__(self, svc, args):
        import torch
        self.torch = torch
        self.svc = svc
        ns = argparse.Namespace(f0_condition=args.f0_condition, checkpoint=args.checkpoint, config=args.config, fp16=args.fp16)
        (self.model, self.semantic_fn, self.f0_fn, self.vocoder_fn, self.campplus,
         self.to_mel, self.mel_args) = svc.load_models(ns)
        self.f0_condition = bool(args.f0_condition)
        self.auto_f0 = bool(args.auto_f0_adjust)
        self.f0_shift = int(args.f0_shift)
        self.device = svc.device
        self.sr = int(self.mel_args["sampling_rate"])
        self.hop = int(self.mel_args["hop_size"])
        self.max_context_window = self.sr // self.hop * 30
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop
        self.fp16 = args.fp16
        self.steps = args.diffusion_steps
        self.cfg = args.cfg_rate
        self.length_adjust = args.length_adjust
        self.ref_max_sec = args.ref_max_sec
        self.refs = {}

    # -- semantic features (whisper-small encoder), same chunking as inference.py
    def semantic(self, w16):
        torch = self.torch
        if w16.size(-1) <= 16000 * 30:
            return self.semantic_fn(w16)
        ovl = 5
        out, buf, trav = [], None, 0
        while trav < w16.size(-1):
            if buf is None:
                chunk = w16[:, trav:trav + 16000 * 30]
            else:
                chunk = torch.cat([buf, w16[:, trav:trav + 16000 * (30 - ovl)]], dim=-1)
            s = self.semantic_fn(chunk)
            out.append(s if trav == 0 else s[:, 50 * ovl:])
            buf = chunk[:, -16000 * ovl:]
            trav += 30 * 16000 if trav == 0 else chunk.size(-1) - 16000 * ovl
        return torch.cat(out, dim=1)

    def prepare_ref(self, path):
        """Encode a reference voice once: mel prompt, CAM++ timbre vector, length-regulated semantic prompt."""
        if path in self.refs:
            return self.refs[path]
        torch = self.torch
        import torchaudio
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))[None].to(self.device)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        f0_full = estimate_f0_hz(torchaudio.functional.resample(wav, self.sr, 16000))   # pitch class from the whole clip
        wav = wav[:, : int(self.sr * min(self.ref_max_sec, 25))]
        w16 = torchaudio.functional.resample(wav, self.sr, 16000)
        S_ori = self.semantic(w16)
        mel2 = self.to_mel(wav.float())
        feat2 = torchaudio.compliance.kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus(feat2.unsqueeze(0))
        F0_ori = None
        if self.f0_condition:
            F0_ori = torch.from_numpy(self.f0_fn(w16[0], thred=0.03)).to(self.device)[None]
        prompt_condition = self.model.length_regulator(
            S_ori, ylens=torch.LongTensor([mel2.size(2)]).to(self.device), n_quantizers=3, f0=F0_ori)[0]
        ref = {"path": str(path), "name": Path(path).stem, "mel2": mel2, "style2": style2,
               "prompt_condition": prompt_condition, "prompt_frames": int(mel2.size(2)),
               "prompt_s": round(wav.size(-1) / self.sr, 2), "F0_ori": F0_ori,
               "f0_hz": f0_full}
        self.refs[path] = ref
        return ref

    def _source_f0(self, w16, ref):
        """RMVPE F0 of the source, shifted to the reference's pitch level like seed-vc/inference.py."""
        torch = self.torch
        F0_alt = torch.from_numpy(self.f0_fn(w16[0], thred=0.03)).to(self.device)[None]
        F0_ori = ref["F0_ori"]
        voiced_alt, voiced_ori = F0_alt[F0_alt > 1], F0_ori[F0_ori > 1]
        log_alt = torch.log(F0_alt + 1e-5)
        shifted = log_alt.clone()
        if self.auto_f0 and voiced_alt.numel() and voiced_ori.numel():
            shifted[F0_alt > 1] = log_alt[F0_alt > 1] - torch.median(torch.log(voiced_alt + 1e-5)) + torch.median(torch.log(voiced_ori + 1e-5))
        shifted = torch.exp(shifted)
        if self.f0_shift:
            shifted[F0_alt > 1] = shifted[F0_alt > 1] * (2 ** (self.f0_shift / 12))
        return shifted

    def convert(self, w16, ref, t):
        """w16: (1, N) float tensor at 16 kHz on device -> numpy float32 at self.sr.
        Mirrors seed-vc/inference.py main() with per-stage timers accumulated into t."""
        torch, np = self.torch, __import__("numpy")
        import torchaudio
        with Timer(t, "resample_s"):
            src = torchaudio.functional.resample(w16, 16000, self.sr)
        with Timer(t, "semantic_s"):
            S_alt = self.semantic(w16)
        with Timer(t, "mel_s"):
            mel = self.to_mel(src.float())
        target_lengths = torch.LongTensor([int(mel.size(2) * self.length_adjust)]).to(self.device)
        f0_src = None
        if self.f0_condition:
            with Timer(t, "f0_s"):
                f0_src = self._source_f0(w16, ref)
        with Timer(t, "lenreg_s"):
            cond = self.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=f0_src)[0]
        mel2, style2, prompt_condition = ref["mel2"], ref["style2"], ref["prompt_condition"]
        max_source_window = self.max_context_window - mel2.size(2)
        processed, chunks, previous = 0, [], None
        n_windows = 0
        while processed < cond.size(1):
            chunk_cond = cond[:, processed:processed + max_source_window]
            is_last = processed + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with Timer(t, "dit_s"):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
                    vc_target = self.model.cfm.inference(
                        cat_condition, torch.LongTensor([cat_condition.size(1)]).to(self.device),
                        mel2, style2, None, self.steps, inference_cfg_rate=self.cfg)
                    vc_target = vc_target[:, :, mel2.size(-1):]
            with Timer(t, "vocoder_s"):
                vc_wave = self.vocoder_fn(vc_target.float()).squeeze()[None, :]
            n_windows += 1
            ovl = self.overlap_wave_len
            if processed == 0:
                if is_last:
                    chunks.append(vc_wave[0].cpu().numpy())
                    break
                chunks.append(vc_wave[0, :-ovl].cpu().numpy())
                previous = vc_wave[0, -ovl:]
                processed += vc_target.size(2) - self.overlap_frame_len
            elif is_last:
                chunks.append(self.svc.crossfade(previous.cpu().numpy(), vc_wave[0].cpu().numpy(), ovl))
                processed += vc_target.size(2) - self.overlap_frame_len
                break
            else:
                chunks.append(self.svc.crossfade(previous.cpu().numpy(), vc_wave[0, :-ovl].cpu().numpy(), ovl))
                previous = vc_wave[0, -ovl:]
                processed += vc_target.size(2) - self.overlap_frame_len
        t["n_dit_windows"] = t.get("n_dit_windows", 0) + n_windows
        t["n_vc_calls"] = t.get("n_vc_calls", 0) + 1
        return np.concatenate(chunks).astype("float32")

    def convert_to_16k(self, w16, ref, t):
        """Convert and bring the result back to 16 kHz numpy, length-fitted to the input."""
        torch = self.torch
        import torchaudio
        out = self.convert(w16, ref, t)
        with Timer(t, "resample_s"):
            out16 = torchaudio.functional.resample(torch.from_numpy(out)[None].to(self.device), self.sr, 16000)
            out16 = out16[0].cpu().numpy()
        return fit_length(out16, w16.size(-1))


class SeedVC2Engine:
    """Seed-VC V2 (hubert-bsqvae-small): ASTRAL speaker-disentangled content tokens + CFM DiT, optional AR
    style model. Mirrors modules/v2/vc_wrapper.VoiceConversionWrapper.convert_voice_with_streaming step by
    step, but on in-memory segments with the reference encoded once. Exposes the same interface as SeedVCEngine."""

    def __init__(self, args):
        import torch
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig
        self.torch = torch
        cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
        self.w = instantiate(cfg)
        self.w.load_checkpoints(ar_checkpoint_path=args.v2_ar_checkpoint, cfm_checkpoint_path=args.v2_cfm_checkpoint)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.float16 if (args.fp16 and self.device.type == "cuda") else torch.float32
        self.w.to(self.device)
        self.w.eval()
        self.w.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=self.dtype, device=self.device)
        self.convert_style = bool(args.v2_convert_style)
        if args.compile and self.convert_style:
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            if hasattr(torch._inductor.config, "fx_graph_cache"):
                torch._inductor.config.fx_graph_cache = True
            self.w.compile_ar()
        self.sr = int(self.w.sr)
        self.hop = int(self.w.hop_size)
        self.max_context_window = self.sr // self.hop * self.w.dit_max_context_len
        self.overlap_frame_len = self.w.overlap_frame_len
        self.overlap_wave_len = self.overlap_frame_len * self.hop
        self.fp16 = args.fp16
        self.steps = args.diffusion_steps
        self.cfg = [args.v2_intelligibility_cfg, args.v2_similarity_cfg]
        self.anon = bool(args.v2_anonymization_only)
        self.top_p, self.temperature, self.rep = args.v2_top_p, args.v2_temperature, args.v2_repetition_penalty
        self.length_adjust = args.length_adjust
        self.ref_max_sec = min(args.ref_max_sec, self.w.dit_max_context_len - 5)   # upstream hard limit: 25 s
        self.f0_condition = False
        self.refs = {}

    def prepare_ref(self, path):
        if path in self.refs:
            return self.refs[path]
        torch = self.torch
        import torchaudio
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))[None].to(self.device)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        f0_full = estimate_f0_hz(torchaudio.functional.resample(wav.float(), self.sr, 16000))
        wav = wav[:, : int(self.sr * self.ref_max_sec)].float()
        w16 = torchaudio.functional.resample(wav, self.sr, 16000)
        target_mel = self.w.mel_fn(wav)
        tlen = target_mel.size(2)
        narrow = None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            tci = self.w._process_content_features(w16, is_narrow=False)
            style = self.w.compute_style(w16)
            prompt_condition, _ = self.w.cfm_length_regulator(tci, ylens=torch.LongTensor([tlen]).to(self.device))
            if self.convert_style:
                tni = self.w._process_content_features(w16, is_narrow=True)
                narrow, _ = self.w.duration_reduction_func(tni[0], 1)
        # the CFM solver runs in float32 (as upstream forces); tensors produced under fp16 autocast must be cast back
        ref = {"path": str(path), "name": Path(path).stem, "mel2": target_mel.float(), "style2": style.float(),
               "prompt_condition": prompt_condition.float(), "prompt_frames": int(tlen), "content_indices": tci,
               "narrow_reduced": narrow, "prompt_s": round(wav.size(-1) / self.sr, 2), "f0_hz": f0_full}
        self.refs[path] = ref
        return ref

    def convert(self, w16, ref, t):
        torch, np = self.torch, __import__("numpy")
        import torchaudio
        dev = self.device
        with Timer(t, "resample_s"):
            src = torchaudio.functional.resample(w16, 16000, self.sr).float()
        with Timer(t, "mel_s"):
            source_mel_len = self.w.mel_fn(src).size(2)
        with Timer(t, "semantic_s"):
            with torch.autocast(device_type=dev.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                sci = self.w._process_content_features(w16, is_narrow=False)
                sni = self.w._process_content_features(w16, is_narrow=True) if self.convert_style else None
        mel2, style2, prompt_condition = ref["mel2"], ref["style2"], ref["prompt_condition"]
        tlen = mel2.size(2)
        state = {"processed": 0, "previous": None, "chunks": [], "n_windows": 0}

        def run_window(cat_condition, ac_dtype):
            cat_condition = cat_condition.float()
            original_len = cat_condition.size(1)
            with Timer(t, "dit_s"):
                with torch.autocast(device_type=dev.type, dtype=ac_dtype, enabled=ac_dtype != torch.float32):
                    vc_mel = self.w.cfm.inference(cat_condition, torch.LongTensor([original_len]).to(dev), mel2, style2,
                                                  self.steps, inference_cfg_rate=self.cfg, random_voice=self.anon)
            vc_mel = vc_mel[:, :, tlen:original_len]
            with Timer(t, "vocoder_s"):
                vc_wave = self.w.vocoder(vc_mel.float()).squeeze()[None]
            state["n_windows"] += 1
            return vc_mel, vc_wave

        def stitch(vc_wave, vc_mel, is_last):   # = vc_wrapper._stream_wave_chunks without the mp3 encoding
            ovl = self.overlap_wave_len
            if state["processed"] == 0:
                if is_last:
                    state["chunks"].append(vc_wave[0].cpu().numpy())
                    return True
                state["chunks"].append(vc_wave[0, :-ovl].cpu().numpy())
                state["previous"] = vc_wave[0, -ovl:]
            elif is_last:
                state["chunks"].append(self.w.crossfade(state["previous"].cpu().numpy(), vc_wave[0].cpu().numpy(), ovl))
                state["processed"] += vc_mel.size(2) - self.overlap_frame_len
                return True
            else:
                state["chunks"].append(self.w.crossfade(state["previous"].cpu().numpy(), vc_wave[0, :-ovl].cpu().numpy(), ovl))
                state["previous"] = vc_wave[0, -ovl:]
            state["processed"] += vc_mel.size(2) - self.overlap_frame_len
            return False

        if not self.convert_style:
            with Timer(t, "lenreg_s"):
                cond, _ = self.w.cfm_length_regulator(
                    sci, ylens=torch.LongTensor([int(source_mel_len * self.length_adjust)]).to(dev))
                cond = cond.float()
            max_source_window = self.max_context_window - tlen
            while state["processed"] < cond.size(1):
                p0 = state["processed"]
                chunk_cond = cond[:, p0:p0 + max_source_window]
                is_last = p0 + max_source_window >= cond.size(1)
                vc_mel, vc_wave = run_window(torch.cat([prompt_condition, chunk_cond], dim=1), torch.float32)  # upstream forces fp32 here
                if stitch(vc_wave, vc_mel, is_last):
                    break
        else:
            src_reduced, _ = self.w.duration_reduction_func(sni[0], 1)
            tgt_reduced = ref["narrow_reduced"]
            max_chunk = self.w.ar_max_content_len - len(tgt_reduced)
            for i in range(0, len(src_reduced), max_chunk):
                is_last = i + max_chunk >= len(src_reduced)
                chunk = src_reduced[i:i + max_chunk]
                with Timer(t, "ar_s"):
                    with torch.autocast(device_type=dev.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                        if self.anon:
                            ar_cond = self.w.ar_length_regulator(chunk[None])[0]
                            ar_out = self.w.ar.generate(ar_cond, torch.zeros([1, 0]).long().to(dev),
                                                        compiled_decode_fn=self.w.compiled_decode_fn, top_p=self.top_p,
                                                        temperature=self.temperature, repetition_penalty=self.rep)
                        else:
                            ar_cond = self.w.ar_length_regulator(torch.cat([tgt_reduced, chunk], dim=0)[None])[0]
                            ar_out = self.w.ar.generate(ar_cond, ref["content_indices"], compiled_decode_fn=self.w.compiled_decode_fn,
                                                        top_p=self.top_p, temperature=self.temperature, repetition_penalty=self.rep)
                        mel_len = int(source_mel_len / sci.size(-1) * ar_out.size(-1) * self.length_adjust)
                        chunk_cond, _ = self.w.cfm_length_regulator(ar_out, ylens=torch.LongTensor([mel_len]).to(dev))
                vc_mel, vc_wave = run_window(torch.cat([prompt_condition, chunk_cond], dim=1), self.dtype)
                if stitch(vc_wave, vc_mel, is_last):
                    break
        t["n_dit_windows"] = t.get("n_dit_windows", 0) + state["n_windows"]
        t["n_vc_calls"] = t.get("n_vc_calls", 0) + 1
        return np.concatenate(state["chunks"]).astype("float32")

    def convert_to_16k(self, w16, ref, t):
        torch = self.torch
        import torchaudio
        out = self.convert(w16, ref, t)
        with Timer(t, "resample_s"):
            out16 = torchaudio.functional.resample(torch.from_numpy(out)[None].to(self.device), self.sr, 16000)
            out16 = out16[0].cpu().numpy()
        return fit_length(out16, w16.size(-1))


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def fit_length(x, n):
    import numpy as np
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def place(canvas, seg, start, fade=160):
    """Add seg into canvas at sample offset start with short fades (10 ms at 16 kHz)."""
    import numpy as np
    n = len(seg)
    if n == 0:
        return
    f = min(fade, n // 4)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype="float32")
        seg = seg.copy()
        seg[:f] *= ramp
        seg[-f:] *= ramp[::-1]
    end = min(start + n, len(canvas))
    canvas[start:end] += seg[: end - start]


def estimate_f0_hz(w16, max_sec=60):
    """Median voiced F0 (Hz) of a (1, N) 16 kHz tensor; energy-gated autocorrelation pitch tracker. None if unvoiced.
    The clip is scaled to its peak first, so the voiced-frame gate is relative to the clip's own level (a quiet
    recording, RMS ~0.01, would otherwise have no frame above the gate and no pitch at all)."""
    import torch
    import torchaudio
    x = w16[:, : 16000 * max_sec].detach().float().cpu()
    if x.size(-1) < 16000:
        return None
    x = x / (x.abs().max() + 1e-8) * 0.9
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(x, 16000, frame_time=0.02, freq_low=60, freq_high=500)[0]
    except Exception:
        return None
    n = f0.numel()
    if n == 0:
        return None
    frames = x[0, : (x.size(-1) // n) * n].reshape(n, -1)
    energy = frames.pow(2).mean(dim=1).sqrt()
    voiced = f0[(energy > 0.02) & (f0 > 60) & (f0 < 500)]
    if voiced.numel() < 10:
        return None
    return float(voiced.median())


PITCH_CLASS_HZ = 165.0  # below: low (typically male) voice; above: high (typically female / child)


class LangID:
    """Zero-shot spoken language ID with Whisper (99 languages incl. all scheduled Indic ones): one decoder step
    from <|startoftranscript|>, softmax over the language tokens. ~1 GB fp16 for whisper-small, <0.1 s per file."""

    def __init__(self, name, device):
        import re
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.torch = torch
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        tok = self.proc.tokenizer
        self.lang_tokens = [t for t in tok.additional_special_tokens if re.fullmatch(r"<\|[a-z]{2,3}\|>", t)]
        self.lang_ids = torch.tensor(tok.convert_tokens_to_ids(self.lang_tokens), device=device)
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.device = device

    def __call__(self, x16):
        """x16: 1-D numpy float32 at 16 kHz (<= 30 s used). Returns (code, prob)."""
        torch = self.torch
        feats = self.proc(x16[: 16000 * 30], sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=torch.float16)
        with torch.no_grad():
            logits = self.model(input_features=feats, decoder_input_ids=torch.tensor([[self.sot]], device=self.device)).logits[0, -1]
        probs = logits[self.lang_ids].float().softmax(-1)
        k = int(probs.argmax())
        return self.lang_tokens[k].strip("<|>"), float(probs[k])


def load_ref_pools(ref_dir, excl):
    """{pool_name: [wav paths]}; sub-folders are language pools, loose wavs form the 'neutral' pool."""
    ref_dir = Path(ref_dir)
    ok = lambda p: p.suffix.lower() == ".wav" and not any(s in p.stem.lower() for s in excl)
    pools = {}
    loose = sorted(str(p) for p in ref_dir.iterdir() if p.is_file() and ok(p))
    if loose:
        pools["neutral"] = loose
    for sub in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        wavs = sorted(str(p) for p in sub.iterdir() if p.is_file() and ok(p))
        if wavs:
            pools[sub.name.lower()] = wavs
    return pools


def pool_for(pools, lang):
    """Language pool if present, else the neutral pool, else everything."""
    if lang and lang in pools:
        return pools[lang], lang
    if "neutral" in pools:
        return pools["neutral"], "neutral"
    return sorted({r for v in pools.values() for r in v}), "all"


def gender_of(ref, ref_gender, ref_f0):
    g = ref_gender.get(ref)
    if g:
        return g
    f = ref_f0.get(ref)
    return None if not f else ("male" if f < PITCH_CLASS_HZ else "female")


PROFILE_KEYS = {   # profile variable -> (argparse destination, converter)
    "MODEL": ("model", str),
    "CKPT": ("checkpoint", lambda v: v or None),
    "CFG_YML": ("config", lambda v: v or None),
    "F0": ("f0_condition", lambda v: str(v).lower() in ("1", "true", "yes")),
    "REF_STRATEGY": ("ref_strategy", str),
    "MAX_SPEAKERS": ("max_speakers", int),
    "STEPS": ("diffusion_steps", int),
    "CFG": ("cfg_rate", float),
    "REF_SEC": ("ref_max_sec", float),
    "CTX": ("context_sec", float),
    "MERGE_GAP": ("merge_gap", float),
    "PAD": ("pad", float),
    "MIN_SEG": ("min_seg", float),
}


def read_profile(path, project_root=None, models_dir=None):
    """Parse a pipeline settings file (KEY=VALUE lines, # comments, bash-sourceable) and expand ${PIPE}, ${BENCH_ROOT},
    ${MODELS_DIR} and any environment variable. Returns {KEY: value} with values as strings."""
    import re
    root = Path(project_root or os.environ.get("BENCH_ROOT") or Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env.setdefault("PIPE", str(root))
    env.setdefault("BENCH_ROOT", str(root))
    env["MODELS_DIR"] = str(models_dir or env.get("MODELS_DIR") or (Path(env["BENCH_ROOT"]) / "models"))
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip('"').strip("'")
        v = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: env.get(m.group(1) or m.group(2), ""), v)
        out[k.strip()] = v
    return out


def apply_profile(args, argv=None):
    """Fill args from --profile for every setting NOT given explicitly on the command line."""
    if not getattr(args, "profile", None):
        return {}
    prof = read_profile(args.profile, models_dir=getattr(args, "models_dir", None))
    argv = argv if argv is not None else sys.argv[1:]
    given = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    applied = {}
    for key, (dest, conv) in PROFILE_KEYS.items():
        if key in prof and "--" + dest.replace("_", "-") not in given:
            setattr(args, dest, conv(prof[key]))
            applied[dest] = getattr(args, dest)
    return applied


def cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def read_groups(path):
    """--groups CSV (header id,group[,num_speakers]) -> {id: {"group": name, "n": speaker cap or 0}}."""
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fid = (row.get("id") or "").strip()
            if fid:
                n = (row.get("num_speakers") or "").strip()
                out[fid] = {"group": (row.get("group") or "").strip() or fid, "n": int(n) if n else 0}
    return out


LINK_MIN_COS = 0.35   # CAM++ cosine below which a cross-file match is flagged for a listen
PITCH_MARGIN_HZ = 15  # within this distance of the pitch line a voice is ambiguous: its gender is not read per file


def clearly_apart(f1, f2, split_hz, margin=PITCH_MARGIN_HZ):
    """True when two median pitches sit clearly on opposite sides of split_hz (beyond the margin on both sides)."""
    if not (split_hz and f1 and f2):
        return False
    lo, hi = split_hz - margin, split_hz + margin
    return (f1 < lo and f2 > hi) or (f1 > hi and f2 < lo)


def link_speakers(files, split_hz=None):
    """Label the speakers of the files of ONE group (the same people recorded several times) with group-wide ids
    G1, G2, ... so that one person carries the same id, and so one target voice, in every file.
    files: [{"file_id", "speakers": [spk, ...] (most speech first), "speech": {spk: s}, "emb": {spk: vector|None},
    "f0": {spk: Hz|None}}]. The file with the most speakers (then the most speech) sets the ids; every other file's
    speakers are matched one-to-one to them by the highest total cosine to the running speech-weighted CAM++
    centroids (pitch closeness where an embedding is missing).
    Gender is never traded away: a speaker whose pitch is clearly on the other side of split_hz from a person's
    (beyond PITCH_MARGIN_HZ on both sides) is never linked to that person, however similar the voices (two people in
    one room can score high); the speaker becomes a new person with a target of their own gender. Within the margin
    pitch says little about gender, so the voice decides. A person's gender is then the speech-weighted pitch over
    all their files.
    Every link is checked: opposite pitch classes, or a leave-one-out cosine under LINK_MIN_COS, is reported in the
    flags (the matching may be wrong, or the folder may not hold the same people).
    Returns ({file_id: {spk: gid}}, {gid: {"speech": s, "f0": Hz|None, "emb": unit vector|None}},
             [{"file", "spk", "person", "cos", "spk_f0", "person_f0", "reason"}])."""
    import itertools
    import numpy as np

    def unit(v):
        v = np.asarray(v, dtype=float).ravel()
        return v / (np.linalg.norm(v) + 1e-9)

    people, out, parts = {}, {}, []
    for f in sorted(files, key=lambda f: (-len(f["speakers"]), -sum(f["speech"].values()))):
        def score(s, g):
            if g is None:                         # a new person: worse than any allowed match
                return -2.0
            fs = f["f0"].get(s)
            fg = people[g]["f0"] / people[g]["f0_w"] if people[g]["f0_w"] else None
            e, c = f["emb"].get(s), people[g]["emb"]
            cos = float(np.dot(unit(e), unit(c))) if e is not None and c is not None else None
            if clearly_apart(fs, fg, split_hz):
                return None                       # not allowed: the other gender
            if cos is not None:
                return cos
            return -abs(fs - fg) / 100.0 if fs and fg else 0.0
        spks = f["speakers"]
        best, assign = None, {}
        # every one-to-one assignment of this file's speakers to known people or to new ones (None)
        for perm in itertools.permutations(list(people) + [None] * len(spks), len(spks)):
            scores = [score(s, g) for s, g in zip(spks, perm)]
            if any(v is None for v in scores):
                continue
            if best is None or sum(scores) > best:
                best, assign = sum(scores), dict(zip(spks, perm))
        for s in spks:
            if assign.get(s) is None:
                g = f"G{len(people) + 1}"
                people[g] = {"emb": None, "speech": 0.0, "f0": 0.0, "f0_w": 0.0}
                assign[s] = g
        for s, g in assign.items():
            w, P = float(f["speech"].get(s, 0.0)), people[g]
            v = None
            if f["emb"].get(s) is not None:
                v = unit(f["emb"][s]) * max(w, 1e-3)
                P["emb"] = v if P["emb"] is None else P["emb"] + v
            if f["f0"].get(s):
                P["f0"] += f["f0"][s] * w
                P["f0_w"] += w
            P["speech"] += w
            parts.append((f["file_id"], s, g, v, f["f0"].get(s)))
        out[f["file_id"]] = assign
    summary = {g: {"speech": P["speech"], "f0": P["f0"] / P["f0_w"] if P["f0_w"] else None,
                   "emb": None if P["emb"] is None else unit(P["emb"])} for g, P in people.items()}
    flags = []
    n_files = {g: sum(1 for x in parts if x[2] == g) for g in people}
    for fid, s, g, v, f0 in parts:
        reasons, cos = [], None
        if n_files[g] < 2:
            if len(files) > 1:
                reasons.append("heard in this file only (no match in the other files)")
        else:
            if v is not None and people[g]["emb"] is not None:
                rest = people[g]["emb"] - v       # the person's voice without this file
                cos = float(np.dot(unit(v), unit(rest))) if np.linalg.norm(rest) > 1e-6 else None
            pf0 = summary[g]["f0"]
            if clearly_apart(f0, pf0, split_hz):
                reasons.append("pitch clearly on the other side of the line from the person's (voice matched)")
            if cos is not None and cos < LINK_MIN_COS:
                reasons.append(f"low voice similarity ({cos:.2f})")
        if reasons:
            pf0 = summary[g]["f0"]
            flags.append({"file": fid, "spk": s, "person": g, "cos": None if cos is None else round(cos, 3),
                          "spk_f0": None if not f0 else round(f0, 1), "person_f0": None if not pf0 else round(pf0, 1),
                          "reason": "; ".join(reasons)})
    return out, summary, flags


def speaker_embedding(eng, w16):
    """CAM++ timbre vector (numpy, 192-d) of a 16 kHz speech tensor [1, T], computed the same way as for the references.
    None when the engine has no CAM++ or the audio is under one second."""
    camp = getattr(eng, "campplus", None)
    if camp is None or w16 is None or w16.size(-1) < 16000:
        return None
    import torchaudio
    feat = torchaudio.compliance.kaldi.fbank(w16.float(), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return camp(feat.unsqueeze(0)).detach().float().squeeze(0).cpu().numpy()


def choose_refs(refs, speakers, file_id, strategy, spk_f0=None, ref_f0=None, spk_gender=None, ref_gender=None,
                ref_source=None, spk_emb=None, ref_emb=None, rank=0):
    """speakers: list ordered by speech amount (desc). Returns dict spk -> ref path.
    pitch strategy: same GENDER as the speaker (classifier labels; pitch class when a label is missing), nearest
    median F0, each reference used once per file, never a reference cut from this very file (ref_source).
    pitch_far: same gender, farthest median F0. cross: opposite gender, farthest median F0.
    timbre_far: reference on the speaker's side of PITCH_CLASS_HZ with the lowest cosine similarity of CAM++ embeddings
    (spk_emb / ref_emb: 1-D numpy or torch vectors); falls back to pitch_far when an embedding is missing."""
    if ref_source:
        own = [r for r in refs if ref_source.get(r) == file_id]
        if own and len(own) < len(refs):
            refs = [r for r in refs if r not in own]
    n = len(refs)
    spk_gender, ref_gender, ref_f0 = spk_gender or {}, ref_gender or {}, ref_f0 or {}
    if strategy in ("pitch", "pitch_far", "cross", "timbre_far") and spk_f0:
        spk_emb, ref_emb = spk_emb or {}, ref_emb or {}

        def cos(a, b):
            import numpy as np
            a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        used, out = set(), {}
        for spk in speakers:
            f = spk_f0.get(spk)
            cands = [r for r in refs if r not in used] or refs
            g = spk_gender.get(spk) or (None if f is None else ("male" if f < PITCH_CLASS_HZ else "female"))
            if g is not None:
                want = g if strategy != "cross" else ("female" if g == "male" else "male")
                same = [r for r in cands if gender_of(r, ref_gender, ref_f0) == want]
                pool = same or cands
                dist = (lambda r: abs((ref_f0.get(r) or 0) - f))
                nth = lambda seq: seq[rank % len(seq)]           # the rank-th choice (rank 0 = the best)
                if f is None:
                    pick = nth(pool)
                elif strategy == "pitch":
                    pick = nth(sorted(pool, key=dist))
                elif strategy == "timbre_far" and spk_emb.get(spk) is not None and any(ref_emb.get(r) is not None for r in pool):
                    # same side of the pitch line as the speaker (keeps the perceived gender), then least similar timbre
                    side = [r for r in pool if ref_f0.get(r) is not None and ref_emb.get(r) is not None
                            and ((ref_f0[r] < PITCH_CLASS_HZ) == (f < PITCH_CLASS_HZ))]
                    side = side or [r for r in pool if ref_emb.get(r) is not None]
                    pick = nth(sorted(side, key=lambda r: cos(spk_emb[spk], ref_emb[r])))
                else:                       # pitch_far / cross: farthest F0 within the wanted gender, among references
                    known = [r for r in pool if ref_f0.get(r) is not None]   # whose F0 is known (unknown would read as 0 Hz)
                    pick = nth(sorted(known or pool, key=dist, reverse=True))
            else:
                pick = cands[(int(hashlib.md5((file_id + spk).encode()).hexdigest(), 16) + rank) % len(cands)]
            out[spk] = pick
            used.add(pick)
        return out
    if strategy == "fixed":
        base = 0
    else:
        base = int(hashlib.md5(file_id.encode()).hexdigest(), 16) % n
    return {spk: refs[(base + rank + k) % n] for k, spk in enumerate(speakers)}


def match_loudness(y, src, max_gain=10.0):
    """Scale converted segment y to the RMS of the source segment src (both numpy)."""
    import numpy as np
    if len(y) == 0 or len(src) == 0:
        return y
    ys = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    ss = float(np.sqrt(np.mean(src ** 2)) + 1e-9)
    if ss < 1e-4 or ys < 1e-4:
        return y
    return y * float(min(ss / ys, max_gain))


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    applied = apply_profile(args)
    if applied:
        print(f"[profile] {args.profile}: " + ", ".join(f"{k}={v}" for k, v in applied.items()), flush=True)
    # absolute paths BEFORE bootstrap chdirs into the seed-vc checkout (relative --ref-dir etc. would break)
    for k in ("wav_dir", "rttm_dir", "out_dir", "log_dir", "ref_dir", "files", "checkpoint", "config",
              "v2_cfm_checkpoint", "v2_ar_checkpoint"):
        if getattr(args, k, None):
            setattr(args, k, str(Path(getattr(args, k)).resolve()))
    svc = bootstrap_seedvc(args.seedvc_root)
    import torch
    import numpy as np
    import soundfile as sf
    import torchaudio

    log_dir, out_dir = Path(args.log_dir), Path(args.out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_w = JsonlWriter(log_dir / f"anonymize_{args.tag}_records.jsonl")
    seg_w = JsonlWriter(log_dir / f"anonymize_{args.tag}_segments.jsonl")

    ref_dir = Path(args.ref_dir) if args.ref_dir else Path(args.seedvc_root) / "examples" / "reference"
    excl = [s.strip().lower() for s in (args.ref_exclude or "").split(",") if s.strip()]
    pools = load_ref_pools(ref_dir, excl)
    refs = sorted({r for v in pools.values() for r in v})       # every reference gets encoded once
    if not refs:
        sys.exit(f"no reference wavs in {ref_dir} (after excluding {excl})")
    lang_mode = (args.lang or "none").lower()
    log(f"reference pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()) + f"; language mode={lang_mode}")
    # gender + source file of every reference: manifest.json next to the wavs, else the m_/f_ filename prefix
    ref_gender, ref_source = {}, {}
    for r in refs:
        man = Path(r).parent / "manifest.json"
        if man.exists():
            try:
                for m in json.loads(man.read_text(encoding="utf-8")):
                    if m.get("file") == Path(r).name:
                        ref_gender[r] = m.get("gender")
                        ref_source[r] = m.get("source_file")
            except Exception as e:
                log(f"could not read {man}: {e!r}")
        stem = Path(r).stem.lower()
        if not ref_gender.get(r) and len(stem) > 1 and stem[0] in "mf" and (stem[1] == "_" or stem[1].isdigit()):
            ref_gender[r] = "male" if stem[0] == "m" else "female"
    log("reference genders: " + ", ".join(f"{Path(r).stem}={ref_gender.get(r) or '?'}" for r in refs))
    if args.mode == "diarized" and not args.rttm_dir:
        sys.exit("--rttm-dir is required for --mode diarized")

    info = env_info()
    info["stage"] = "anonymization"
    info["args"] = vars(args)
    log(f"env: {json.dumps(info)}")

    t_load = {}
    with Timer(t_load, "model_load_s"):
        eng = SeedVC2Engine(args) if args.model == "v2" else SeedVCEngine(svc, args)
    log(f"Seed-VC {args.model} loaded in {t_load['model_load_s']:.1f}s on {info.get('gpu_name', 'cpu')} "
        f"(sr={eng.sr}, hop={eng.hop}, steps={eng.steps}, cfg={eng.cfg}, fp16={eng.fp16}"
        + (f", anonymization_only={eng.anon}, convert_style={eng.convert_style}" if args.model == "v2" else "") + ")")
    load_mem = gpu_peak_mem_mb()
    lid = None
    if lang_mode == "auto":
        t_lid = {}
        with Timer(t_lid, "lid_load_s"):
            lid = LangID(args.lid_model, eng.device)
        log(f"language-ID model {args.lid_model} loaded in {t_lid['lid_load_s']:.1f}s ({len(lid.lang_tokens)} languages)")

    t_ref = {}
    with Timer(t_ref, "ref_prep_s"):
        for r in refs:
            eng.prepare_ref(r)
    log("reference voices (median F0): " + ", ".join(
        f"{eng.refs[r]['name']}={eng.refs[r]['f0_hz']:.0f}Hz" if eng.refs[r]["f0_hz"] else f"{eng.refs[r]['name']}=?" for r in refs))
    log(f"{len(refs)} reference voices encoded in {t_ref['ref_prep_s']:.1f}s "
        f"(prompt {eng.ref_max_sec}s -> {eng.refs[refs[0]]['prompt_frames']} mel frames; "
        f"source window per DiT call = {(eng.max_context_window - eng.refs[refs[0]]['prompt_frames']) * eng.hop / eng.sr:.1f}s)")

    wavs = list_wavs(args.wav_dir, None if args.groups else args.limit, None if args.groups else args.shard, args.files)
    groups = None
    if args.groups:
        # grouped run: the files of one group are the same people recorded several times; each person gets ONE target
        # voice for the whole group. --shard and --limit then count whole groups, never split one.
        groups = read_groups(args.groups)
        wavs = [w for w in wavs if w.stem in groups]
        names = sorted({groups[w.stem]["group"] for w in wavs})
        if args.shard:
            i, k = [int(v) for v in args.shard.split("/")]
            names = [g for j, g in enumerate(names) if j % k == i]
        if args.limit:
            names = names[:args.limit]
        wavs = [w for w in wavs if groups[w.stem]["group"] in names]
    if not wavs:
        sys.exit(f"no wav files in {args.wav_dir}")
    log(f"{len(wavs)} files, mode={args.mode}, granularity={args.granularity}"
        + (f", {len({groups[w.stem]['group'] for w in wavs})} groups (one target per person per group)" if groups else ""))

    warm = {}
    if not args.no_warmup:
        with torch.inference_mode():
            x, sr = sf.read(str(wavs[0]), dtype="float32", frames=16000 * 5, always_2d=True)
            x = torch.from_numpy(x.mean(axis=1))[None].to(eng.device)
            if sr != 16000:
                x = torchaudio.functional.resample(x, sr, 16000)
            tw = {}
            with Timer(tw, "warmup_s"):
                eng.convert_to_16k(x, eng.refs[refs[0]], tw)
            warm = {k: round(v, 3) if isinstance(v, float) else v for k, v in tw.items()}
            log(f"warm-up (5 s clip): {warm}")

    def analyse(wav, rec, t):
        """Read one file, find its speakers (RTTM, or fixed chunks for --mode whole), apply the speaker cap, measure
        each speaker's pitch and CAM++ timbre, identify the language. Returns the per-file state for choosing the
        target voices and converting."""
        file_id = wav.stem
        cap = args.max_speakers
        if groups is not None and groups[file_id].get("n"):
            cap = groups[file_id]["n"]
        with Timer(t, "read_s"):
            x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != 16000:
                x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy()
            n = len(x)
            dur = n / 16000.0
            xg = torch.from_numpy(x)[None].to(eng.device)
        rec["duration_s"] = round(dur, 3)
        if args.mode == "whole":
            step = max(5.0, float(args.whole_chunk_sec))
            spk_segs = {"spk0": [(a, min(dur, a + step)) for a in [i * step for i in range(int(math.ceil(dur / step)))]]}
            raw_n = len(spk_segs["spk0"])
        else:
            rttm = find_rttm(args.rttm_dir, file_id)
            if rttm is None:
                raise FileNotFoundError(f"no RTTM for {file_id} in {args.rttm_dir}")
            raw = read_rttm(rttm)
            raw_n = len(raw)
            spk_segs = merge_segments(raw, args.merge_gap, args.pad, args.min_seg, dur)
            rec["rttm"] = str(rttm)
            # clustering blips: a "speaker" with almost no speech gets no voice of its own
            if len(spk_segs) > 1 and args.min_speaker_sec > 0:
                tot = {s: sum(b - a for a, b in v) for s, v in spk_segs.items()}
                main_spk = max(tot, key=tot.get)
                for s in [s for s in spk_segs if s != main_spk and tot[s] < args.min_speaker_sec]:
                    spk_segs[main_spk] = sorted(spk_segs[main_spk] + spk_segs.pop(s))
                    rec.setdefault("merged_blip_speakers", []).append({"spk": s, "speech_s": round(tot[s], 2)})

        speakers = sorted(spk_segs, key=lambda s: -sum(b - a for a, b in spk_segs[s]))
        spk_f0, spk_gender, spk_emb = {}, {}, {}
        # a grouped run always needs the timbre: it is how one person is recognised across the files of a group
        need_emb = args.ref_strategy == "timbre_far" or groups is not None or (cap > 0 and len(spk_segs) > cap)

        def speaker_audio(spk, max_s=60):
            pieces, tot = [], 0
            for a, b in spk_segs[spk]:
                pieces.append(xg[:, int(a * 16000):int(b * 16000)])
                tot += int(b * 16000) - int(a * 16000)
                if tot > 16000 * max_s:
                    break
            return torch.cat(pieces, dim=-1) if pieces else None

        if args.ref_strategy in ("pitch", "pitch_far", "cross", "timbre_far") or need_emb:
            with Timer(t, "f0_select_s"):
                for spk in speakers:
                    w = speaker_audio(spk)
                    spk_f0[spk] = estimate_f0_hz(w) if w is not None else None
                    if spk_f0[spk] is not None:
                        spk_gender[spk] = "male" if spk_f0[spk] < PITCH_CLASS_HZ else "female"
                    if need_emb:
                        spk_emb[spk] = speaker_embedding(eng, w)
            # speaker cap: surplus clusters (least speech first) join the closest remaining voice
            if cap > 0 and len(speakers) > cap:
                keep = speakers[:cap]
                for s_ in speakers[cap:]:
                    def closeness(k):
                        if spk_emb.get(s_) is not None and spk_emb.get(k) is not None:
                            return cosine(spk_emb[s_], spk_emb[k])
                        return -abs((spk_f0.get(s_) or 0) - (spk_f0.get(k) or 0))
                    target = max(keep, key=closeness)
                    spk_segs[target] = sorted(spk_segs[target] + spk_segs.pop(s_))
                    rec.setdefault("merged_surplus_speakers", []).append(
                        {"spk": s_, "into": target, "f0_hz": (round(spk_f0[s_], 1) if spk_f0.get(s_) else None),
                         "by": "embedding" if spk_emb.get(s_) is not None else "f0"})
                    for d in (spk_f0, spk_gender, spk_emb):
                        d.pop(s_, None)
                speakers = keep
                # the merged speaker's pitch and embedding are re-measured on the joined audio
                for spk in keep:
                    w = speaker_audio(spk)
                    spk_f0[spk] = estimate_f0_hz(w) if w is not None else None
                    if spk_f0[spk] is not None:
                        spk_gender[spk] = "male" if spk_f0[spk] < PITCH_CLASS_HZ else "female"
                    if need_emb:
                        spk_emb[spk] = speaker_embedding(eng, w)
            rec["speaker_f0_hz"] = {s: (round(v, 1) if v else None) for s, v in spk_f0.items()}
            rec["speaker_gender"] = {s: {"gender": g, "source": "pitch"} for s, g in spk_gender.items()}
        # language -> reference pool (zero-shot: Whisper LID on the first 30 s of speech, or a fixed code)
        lang, lang_prob = None, None
        if lang_mode == "auto":
            with Timer(t, "lid_s"):
                pieces, tot = [], 0
                for a, b in sorted(s for ss in spk_segs.values() for s in ss):
                    pieces.append(x[int(a * 16000):int(b * 16000)])
                    tot += int(b * 16000) - int(a * 16000)
                    if tot >= 16000 * 30:
                        break
                lang, lang_prob = lid(np.concatenate(pieces)) if pieces else (None, None)
        elif lang_mode != "none":
            lang = lang_mode
        rec.update({"lang": lang, "lang_prob": None if lang_prob is None else round(lang_prob, 3)})
        return {"file_id": file_id, "x": x, "n": n, "dur": dur, "spk_segs": spk_segs, "raw_n": raw_n,
                "speakers": speakers, "spk_f0": spk_f0, "spk_gender": spk_gender, "spk_emb": spk_emb, "lang": lang}

    def pick_refs(speakers, key, lang, spk_f0, spk_gender, spk_emb, rec):
        """Target voice per speaker (the profile's --ref-strategy) from the language's reference pool."""
        pool, pool_name = pool_for(pools, lang) if lang_mode != "none" else (refs, "flat")
        rec["ref_pool"] = pool_name
        ref_emb = {r: (eng.refs[r]["style2"].detach().float().squeeze().cpu().numpy() if eng.refs[r].get("style2") is not None else None)
                   for r in pool} if args.ref_strategy == "timbre_far" else None
        for s in speakers:
            if spk_gender.get(s) is None and args.ref_strategy in ("pitch", "pitch_far", "cross", "timbre_far"):
                log(f"CHECK {key}: speaker {s} has no measurable pitch (silent or unvoiced): its gender is unknown and "
                    f"the target voice is not gender-matched")
                rec.setdefault("check", []).append(f"{s}: pitch unknown, target gender not enforced")
        ref_map = choose_refs(pool, speakers, key, args.ref_strategy, spk_f0,
                              {r: eng.refs[r]["f0_hz"] for r in pool}, spk_gender, ref_gender, ref_source,
                              spk_emb=spk_emb if args.ref_strategy == "timbre_far" else None, ref_emb=ref_emb,
                              rank=args.ref_rank)
        if args.ref_strategy == "timbre_far" and ref_emb:
            rec["ref_similarity"] = {s: (round(cosine(spk_emb[s], ref_emb[r]), 3)
                                         if spk_emb.get(s) is not None and ref_emb.get(r) is not None else None)
                                     for s, r in ref_map.items()}
        return ref_map

    def convert(st, ref_map, t):
        """Convert every speaker's speech to its target voice and write <out-dir>/<id>.wav on the original timeline.
        Returns (seconds converted, segments converted)."""
        x, n, spk_segs = st["x"], st["n"], st["spk_segs"]
        file_id = st["file_id"]
        xg = torch.from_numpy(x)[None].to(eng.device)
        if args.non_speech in ("keep", "attenuate"):
            canvas = x.copy()
            if args.non_speech == "attenuate":
                canvas *= float(10 ** (-args.attenuate_db / 20))
            for segs in spk_segs.values():
                for a, b in segs:
                    canvas[int(a * 16000):int(b * 16000)] = 0.0
        else:
            canvas = np.zeros(n, dtype="float32")

        converted_s = 0.0
        n_merged = 0
        for spk in st["speakers"]:
            ref = eng.refs[ref_map[spk]]
            segs = [(int(round(a * 16000)), int(round(b * 16000))) for a, b in spk_segs[spk]]
            segs = [(a, b) for a, b in segs if b - a > 0]
            n_merged += len(segs)
            ctx = int(round(args.context_sec * 16000))
            if args.mode == "whole" or args.granularity == "segment":
                for a, b in segs:
                    ts = {}
                    cuda_sync()
                    t0 = time.perf_counter()
                    a0, b0 = max(0, a - ctx), min(n, b + ctx)
                    y = eng.convert_to_16k(xg[:, a0:b0], ref, ts)
                    if ctx:
                        y = fit_length(y[a - a0:a - a0 + (b - a)], b - a)
                    cuda_sync()
                    el = time.perf_counter() - t0
                    if args.loudness == "match":
                        y = match_loudness(y, x[a:b])
                    place(canvas, y, a)
                    converted_s += (b - a) / 16000.0
                    for k, v in ts.items():
                        t[k] = t.get(k, 0) + v
                    seg_w.write({"file": file_id, "tag": args.tag, "spk": spk, "ref": ref["name"],
                                 "start_s": round(a / 16000, 3), "len_s": round((b - a) / 16000, 3),
                                 "wall_s": round(el, 4), "rtf": round(el / ((b - a) / 16000), 4),
                                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ts.items()}})
            else:  # speaker granularity: one VC call over the concatenated speech of this speaker
                ts = {}
                cuda_sync()
                t0 = time.perf_counter()
                pieces = [xg[:, a:b] for a, b in segs]
                cat = torch.cat(pieces, dim=-1)
                y = eng.convert_to_16k(cat, ref, ts)
                cuda_sync()
                el = time.perf_counter() - t0
                ratio = len(y) / cat.size(-1)
                off = 0
                for a, b in segs:
                    ln = b - a
                    piece = fit_length(y[int(round(off * ratio)):int(round((off + ln) * ratio))], ln)
                    if args.loudness == "match":
                        piece = match_loudness(piece, x[a:b])
                    place(canvas, piece, a)
                    off += ln
                converted_s += cat.size(-1) / 16000.0
                for k, v in ts.items():
                    t[k] = t.get(k, 0) + v
                seg_w.write({"file": file_id, "tag": args.tag, "spk": spk, "ref": ref["name"],
                             "start_s": None, "len_s": round(cat.size(-1) / 16000, 3), "n_pieces": len(segs),
                             "wall_s": round(el, 4), "rtf": round(el / (cat.size(-1) / 16000), 4),
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ts.items()}})

        with Timer(t, "write_s"):
            peak = float(np.max(np.abs(canvas))) if n else 0.0
            if peak > 0.99:
                canvas *= 0.99 / peak
            sf.write(str(out_dir / f"{file_id}.wav"), canvas, 16000, subtype="PCM_16")
        return converted_s, n_merged

    acc = {"audio": 0.0, "speech": 0.0, "wall": 0.0, "ok": 0, "fail": 0}

    def new_record(wav):
        return {"stage": "anonymization", "tag": args.tag, "file": wav.stem, "path": str(wav), "mode": args.mode,
                "model": args.model, "context_sec": args.context_sec,
                "granularity": args.granularity if args.mode == "diarized" else "whole",
                "diffusion_steps": args.diffusion_steps, "cfg_rate": args.cfg_rate, "fp16": args.fp16,
                "ref_max_sec": args.ref_max_sec, "started": now_iso(),
                **({"group": groups[wav.stem]["group"]} if groups is not None else {})}

    def finish(label, st, rec, t, ref_map, wall, converted_s, n_merged):
        spk_segs, dur = st["spk_segs"], st["dur"]
        speech_union = union_seconds([iv for segs in spk_segs.values() for iv in segs])
        rec.update({
            "ok": True, "wall_s": round(wall, 3), "rtf_wall": round(wall / dur, 5),
            "rtf_speech": round(wall / converted_s, 5) if converted_s else None,
            "speech_union_s": round(speech_union, 2), "converted_s": round(converted_s, 2),
            "speech_fraction": round(speech_union / dur, 4),
            "n_speakers": len(st["speakers"]), "n_segments_rttm": st["raw_n"], "n_segments_converted": n_merged,
            "ref_map": {s: Path(r).stem for s, r in ref_map.items()},
            "gpu_peak_mem_mb": gpu_peak_mem_mb(),
            "substage_s": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()},
            "output": str(out_dir / f"{st['file_id']}.wav"),
        })
        acc["audio"] += dur
        acc["speech"] += converted_s
        acc["wall"] += wall
        acc["ok"] += 1
        log(f"[{label}] {st['file_id']} dur={dur:.0f}s speech={converted_s:.0f}s segs={n_merged} "
            f"wall={wall:.1f}s rtf_wall={wall / dur:.3f} rtf_speech={(wall / converted_s if converted_s else 0):.3f} "
            f"dit={t.get('dit_s', 0):.1f}s voc={t.get('vocoder_s', 0):.1f}s sem={t.get('semantic_s', 0):.1f}s"
            + (f" lang={rec.get('lang')}({rec.get('lang_prob')}) pool={rec.get('ref_pool')}" if lang_mode != "none" else "")
            + " refs=" + ",".join(
                f"{s}[{(rec.get('speaker_gender') or {}).get(s, {}).get('gender', '?')}]->{Path(r).stem}"
                for s, r in ref_map.items()))

    def failed(label, wav, rec, e):
        import traceback
        rec.update({"ok": False, "error": repr(e), "traceback": traceback.format_exc(), "finished": now_iso()})
        acc["fail"] += 1
        log(f"[{label}] {wav.stem} FAILED: {e!r}")
        rec_w.write(rec)

    stage_start = now_iso()
    if groups is None:
        for idx, wav in enumerate(wavs):
            label = f"{idx + 1}/{len(wavs)}"
            gpu_reset_peak()
            t, rec = {}, new_record(wav)
            try:
                cuda_sync()
                t_file0 = time.perf_counter()
                with torch.inference_mode():
                    st = analyse(wav, rec, t)
                    ref_map = pick_refs(st["speakers"], st["file_id"], st["lang"], st["spk_f0"], st["spk_gender"],
                                        st["spk_emb"], rec)
                    converted_s, n_merged = convert(st, ref_map, t)
                cuda_sync()
                finish(label, st, rec, t, ref_map, time.perf_counter() - t_file0, converted_s, n_merged)
            except Exception as e:
                failed(label, wav, rec, e)
                continue
            rec["finished"] = now_iso()
            rec_w.write(rec)
    else:
        group_refs = {}
        done = 0
        for gname in sorted({groups[w.stem]["group"] for w in wavs}):
            members = [w for w in wavs if groups[w.stem]["group"] == gname]
            # pass 1: speakers of every file in the group
            states = []
            for wav in members:
                done += 1
                label = f"{done}/{len(wavs)} {gname}"
                gpu_reset_peak()
                t, rec = {}, new_record(wav)
                try:
                    cuda_sync()
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        st = analyse(wav, rec, t)
                    cuda_sync()
                    st["analyse_wall"] = time.perf_counter() - t0
                    states.append((label, wav, st, rec, t))
                except Exception as e:
                    failed(label, wav, rec, e)
            if not states:
                continue
            # the same person across the files gets one group-wide id, then one target voice
            link, people, flags = link_speakers([
                {"file_id": st["file_id"], "speakers": st["speakers"], "emb": st["spk_emb"], "f0": st["spk_f0"],
                 "speech": {s: sum(b - a for a, b in st["spk_segs"][s]) for s in st["speakers"]}}
                for _, _, st, _, _ in states], split_hz=PITCH_CLASS_HZ)
            gids = sorted(people, key=lambda g: -people[g]["speech"])
            g_f0 = {g: people[g]["f0"] for g in gids}
            g_gender = {g: ("male" if f < PITCH_CLASS_HZ else "female") for g, f in g_f0.items() if f}
            g_emb = {g: people[g]["emb"] for g in gids}
            langs = [st["lang"] for _, _, st, _, _ in states if st["lang"]]
            g_lang = max(set(langs), key=langs.count) if langs else None
            g_rec = {}
            g_refs = pick_refs(gids, gname, g_lang, g_f0, g_gender, g_emb, g_rec)
            # gender preservation is compulsory: a target of the other gender stops the group before any conversion
            wrong = [g for g in gids if g_gender.get(g) and gender_of(g_refs[g], ref_gender, {r: eng.refs[r]["f0_hz"] for r in refs})
                     not in (None, g_gender[g])]
            if wrong or any(not g_gender.get(g) for g in gids):
                for _, wav, _, rec, _ in states:
                    failed(f"group {gname}", wav, rec, RuntimeError(
                        "gender check failed: " + ", ".join(f"{g} is {g_gender.get(g) or 'of unknown pitch'} but its target "
                                                            f"{Path(g_refs[g]).stem} is not" for g in (wrong or gids))))
                continue
            # ... and per speaker, file by file: a speaker whose own pitch is clearly male or clearly female
            # (beyond the margin) must get a target of that gender
            bad = []
            for _, wav, st, rec, _ in states:
                for s_ in st["speakers"]:
                    f_ = st["spk_f0"].get(s_)
                    tg = gender_of(g_refs[link[st["file_id"]][s_]], ref_gender, {r: eng.refs[r]["f0_hz"] for r in refs})
                    if f_ and tg and ((f_ < PITCH_CLASS_HZ - PITCH_MARGIN_HZ and tg != "male") or
                                      (f_ > PITCH_CLASS_HZ + PITCH_MARGIN_HZ and tg != "female")):
                        bad.append(f"{st['file_id']} speaker {s_} ({f_:.0f} Hz) would get {Path(g_refs[link[st['file_id']][s_]]).stem}")
            if bad:
                for _, wav, _, rec, _ in states:
                    failed(f"group {gname}", wav, rec, RuntimeError("gender check failed: " + "; ".join(bad)))
                continue
            group_refs[gname] = {
                "people": {g: {"target": Path(g_refs[g]).stem, "speech_s": round(people[g]["speech"], 1),
                               "f0_hz": None if g_f0[g] is None else round(g_f0[g], 1), "gender": g_gender.get(g),
                               "target_similarity": (g_rec.get("ref_similarity") or {}).get(g)} for g in gids},
                "language": g_lang, "ref_pool": g_rec.get("ref_pool"), "flags": flags,
                "files": {fid: m for fid, m in link.items()}}
            log(f"group {gname}: " + ", ".join(
                f"{g}[{g_gender.get(g, '?')}, {people[g]['speech']:.0f}s]->{Path(g_refs[g]).stem}" for g in gids))
            for fl in flags:
                log(f"CHECK group {gname}: {fl['file']} speaker {fl['spk']} -> {fl['person']}: {fl['reason']} "
                    f"(speaker {fl['spk_f0']} Hz, person {fl['person_f0']} Hz)")
            # pass 2: convert every file with the group's targets
            for label, wav, st, rec, t in states:
                try:
                    cuda_sync()
                    t0 = time.perf_counter()
                    ref_map = {s: g_refs[link[st["file_id"]][s]] for s in st["speakers"]}
                    rec.update({"ref_pool": g_rec.get("ref_pool"), "group_speaker": link[st["file_id"]]})
                    with torch.inference_mode():
                        converted_s, n_merged = convert(st, ref_map, t)
                    cuda_sync()
                    finish(label, st, rec, t, ref_map, st["analyse_wall"] + time.perf_counter() - t0, converted_s, n_merged)
                except Exception as e:
                    failed(label, wav, rec, e)
                    continue
                rec["finished"] = now_iso()
                rec_w.write(rec)
            states = None
        write_json(log_dir / f"anonymize_{args.tag}_group_refs.json", group_refs)
    tot_audio, tot_speech, tot_wall, n_ok, n_fail = acc["audio"], acc["speech"], acc["wall"], acc["ok"], acc["fail"]

    summary = {
        "stage": "anonymization", "tag": args.tag, "mode": args.mode,
        "granularity": args.granularity if args.mode == "diarized" else "whole",
        "env": info, "seedvc_sr": eng.sr, "diffusion_steps": args.diffusion_steps, "cfg_rate": args.cfg_rate,
        "fp16": args.fp16, "ref_max_sec": args.ref_max_sec, "n_refs": len(refs),
        "ref_strategy": args.ref_strategy, "loudness": args.loudness, "non_speech": args.non_speech,
        "lang_mode": lang_mode, "ref_pools": {k: [Path(r).stem for r in v] for k, v in pools.items()},
        "ref_genders": {Path(r).stem: ref_gender.get(r) for r in refs},
        "f0_condition": args.f0_condition, "refs_f0_hz": {eng.refs[r]["name"]: eng.refs[r]["f0_hz"] for r in refs},
        "model": args.model, "context_sec": args.context_sec,
        "v2": ({"anonymization_only": args.v2_anonymization_only, "convert_style": args.v2_convert_style,
                "intelligibility_cfg": args.v2_intelligibility_cfg, "similarity_cfg": args.v2_similarity_cfg}
               if args.model == "v2" else None),
        "model_load_s": round(t_load["model_load_s"], 3), "ref_prep_s": round(t_ref["ref_prep_s"], 3),
        "gpu_mem_after_load_mb": load_mem, "warmup": warm,
        "started": stage_start, "finished": now_iso(),
        "n_files_ok": n_ok, "n_files_failed": n_fail,
        "audio_hours": round(tot_audio / 3600, 4), "speech_hours": round(tot_speech / 3600, 4),
        "wall_hours": round(tot_wall / 3600, 4),
        "aggregate_rtf_wall": round(tot_wall / tot_audio, 5) if tot_audio else None,
        "aggregate_rtf_speech": round(tot_wall / tot_speech, 5) if tot_speech else None,
        "audio_hours_per_wall_hour": round(tot_audio / tot_wall, 3) if tot_wall else None,
    }
    write_json(log_dir / f"anonymize_{args.tag}_summary.json", summary)
    log(f"SUMMARY {json.dumps(summary)}")
    rec_w.close()
    seg_w.close()


if __name__ == "__main__":
    main()
