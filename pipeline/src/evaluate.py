#!/usr/bin/env python
"""
Objective evaluation of anonymized audio, following the metrics Seed-VC's EVAL.md uses
(speaker-embedding similarity, WER/CER through a fixed ASR, DNSMOS) but pointed at anonymization:

  privacy      cosine similarity between the ORIGINAL speaker and the SAME speaker in the anonymized file
               (WavLM-SV x-vectors) - lower is better anonymization
  distinctness similarity between the two anonymized speakers of a conversation - lower means the
               doctor and patient still sound like two different people
  utility      WER / CER between a fixed Hindi ASR's transcript of the original and of the anonymized file
               (the original's transcript is the pseudo-reference) - lower means content preserved
  naturalness  DNSMOS P.835 SIG / BAK / OVRL (Microsoft DNS-Challenge onnx models) - higher is better
  prosody      per-speaker pitch median and 10-90 range vs the original (1.0 = unchanged)

Runs in the `seedvc` env, offline, from the seed-vc checkout (same HF cache as stage 2):

  python evaluate.py --seedvc-root REPO --orig-dir DATA/wav --rttm-dir RTTM --files 2190472 2006763 \
      --systems A_baseline=runs/quality_ab/A_baseline F_v2_anon=runs/quality_ab_v2/F_v2_anon \
                Kiran=/home/kiran.mk/.../samples/{id}/anonymized.wav  --out runs/quality_ab_v2/eval

A system is name=dir (expects <dir>/<id>.wav) or name=pattern containing {id}.
"""
import os
import re
import sys
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_rttm, find_rttm, log, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seedvc-root", required=True, help="seed-vc checkout; its checkpoints/hf_cache holds the eval models")
    p.add_argument("--orig-dir", required=True)
    p.add_argument("--rttm-dir", required=True, help="reference RTTMs (<id>_SPEAKER.rttm) or stage-1 hypotheses (<id>.rttm)")
    p.add_argument("--systems", nargs="+", required=True, help="name=dir or name=pattern-with-{id}")
    p.add_argument("--files", nargs="+", default=None, help="file ids; default: every wav the first system has")
    p.add_argument("--out", required=True)
    p.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--asr-model", default="openai/whisper-large-v3")
    p.add_argument("--asr-lang", default="hi")
    p.add_argument("--dnsmos-dir", default=None, help="dir with sig_bak_ovr.onnx and model_v8.onnx (default $HF_HOME/dnsmos)")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--no-dnsmos", action="store_true")
    p.add_argument("--max-asr-sec", type=float, default=None, help="only transcribe the first N seconds (speed)")
    p.add_argument("--hyp-rttm-dir", default=None,
                   help="stage-1 diarization hypotheses (<id>.rttm). Enables DER and speaker-attributed cpWER/tcpWER "
                        "(hypothesis = anonymized audio segmented by these RTTMs; reference = original audio + --rttm-dir)")
    p.add_argument("--tcp-collar", type=float, default=5.0, help="tcpWER collar in seconds (meeteval default 5)")
    p.add_argument("--no-tcp", action="store_true", help="skip the per-turn ASR for cpWER/tcpWER (saves ~half the ASR time)")
    p.add_argument("--asr-backend", choices=["whisper", "indic", "sarvam", "mms"], default="whisper",
                   help="indic: ai4bharat IndicConformer (22 Indic languages, ONNX); sarvam: Sarvam AI REST API (needs internet + "
                        "SARVAM_API_KEY in the environment, <=30 s per request); mms: facebook/mms-1b-all with the language adapter; "
                        "whisper: any Whisper checkpoint via --asr-model")
    p.add_argument("--sarvam-model", default="saaras:v4")
    p.add_argument("--sarvam-mode", default="transcribe", help="transcribe | verbatim | translit | codemix")
    p.add_argument("--mms-model", default="facebook/mms-1b-all")
    p.add_argument("--indic-model", default="ai4bharat/indic-conformer-600m-multilingual")
    p.add_argument("--indic-decoder", choices=["ctc", "rnnt"], default="ctc",
                   help="IndicConformer decoder. The ONNX RNNT decoder returns empty or truncated output on some turns "
                        "(both original and anonymized audio); CTC is robust. Whichever is chosen, an empty result on >= 1 s of "
                        "audio is retried with the other decoder")
    p.add_argument("--text-only", action="store_true",
                   help="skip speaker verification and DNSMOS (text metrics, pitch and gender only) - fast re-scoring")
    p.add_argument("--gt-dir", default=None,
                   help="human transcripts <id>.txt (TSV: file utt spk name start end text). When given, WER/CER/WIL/cpWER/"
                        "tcpWER are computed against them (the 'original' row then shows the ASR's own error rate)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def load_16k(path):
    import soundfile as sf
    import numpy as np
    import torch
    import torchaudio
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy()
    return x.astype(np.float32)


def speaker_segments(rttm_dir, file_id):
    p = find_rttm(rttm_dir, file_id)
    if p is None:
        return None
    segs = {}
    for s in read_rttm(p):
        segs.setdefault(s["spk"], []).append((s["start"], s["end"]))
    return segs


def speaker_audio(x, segs, max_sec=90, min_len=0.5):
    import numpy as np
    pieces, tot = [], 0
    for a, b in segs:
        if b - a < min_len:
            continue
        pieces.append(x[int(a * 16000):int(b * 16000)])
        tot += b - a
        if tot > max_sec:
            break
    return np.concatenate(pieces) if pieces else None


# --------------------------------------------------------------------------- #
class SpeakerVerifier:
    def __init__(self, name, device):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        import torch
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = WavLMForXVector.from_pretrained(name).to(device).eval()
        self.device = device
        self.torch = torch

    def embed(self, x):
        torch = self.torch
        embs = []
        step = 16000 * 10
        for i in range(0, max(1, len(x) - 16000), step):
            seg = x[i:i + step]
            if len(seg) < 16000:
                continue
            inp = self.fe(seg, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                e = self.model(**inp).embeddings[0]
            embs.append(torch.nn.functional.normalize(e, dim=-1))
            if len(embs) >= 12:
                break
        if not embs:
            return None
        return torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=-1)

    @staticmethod
    def cos(a, b):
        return float((a * b).sum()) if a is not None and b is not None else None


class ASR:
    def __init__(self, name, lang, device):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=name,
                             torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                             device=0 if device.type == "cuda" else -1)
        self.lang = None if lang in (None, "", "auto") else lang   # None: Whisper detects the language per chunk
        self.gen = {"task": "transcribe"} | ({"language": self.lang} if self.lang else {})

    def __call__(self, x):
        out = self.pipe({"raw": x, "sampling_rate": 16000}, chunk_length_s=30, batch_size=8, return_timestamps=True,
                        generate_kwargs=self.gen)
        return out["text"]

    def batch(self, clips, bs=16):
        """Transcribe many short clips (speaker turns); clips longer than 30 s are chunked by the pipeline."""
        import numpy as np
        texts = []
        for i in range(0, len(clips), bs):
            chunk = [{"raw": np.pad(c, (0, max(0, 16000 - len(c)))), "sampling_rate": 16000} for c in clips[i:i + bs]]
            outs = self.pipe(chunk, chunk_length_s=30, batch_size=bs, return_timestamps=True, generate_kwargs=self.gen)
            texts += [o["text"] for o in outs]
        return texts


class IndicASR:
    """ai4bharat/indic-conformer-600m-multilingual (custom ONNX code): model(wav[1, N] @16 kHz, lang, "ctc"|"rnnt") -> str.
    Gated repo: accept the terms once and export HF_TOKEN before staging. Long audio is transcribed turn by turn."""

    def __init__(self, name, lang, device, decoder="rnnt"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.lang = lang if lang and lang != "auto" else "hi"
        self.decoder = decoder
        self.name = name

    def one(self, x):
        import numpy as np
        torch = self.torch
        if len(x) < 1600:
            return ""
        wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]

        def decode(dec):
            out = self.model(wav, self.lang, dec)
            if isinstance(out, (list, tuple)):
                out = out[0] if out else ""
            return str(out).strip()

        txt = decode(self.decoder)
        if not txt and len(x) >= 16000:                      # decoder failure, not silence: try the other decoder
            txt = decode("ctc" if self.decoder == "rnnt" else "rnnt")
            self.n_fallback = getattr(self, "n_fallback", 0) + 1
        return txt

    def __call__(self, x, win_s=20.0):
        """Whole audio: fixed windows (used only when no turn segmentation is available)."""
        n = int(win_s * 16000)
        return " ".join(t for t in (self.one(x[i:i + n]) for i in range(0, len(x), n)) if t)

    def batch(self, clips, bs=None):
        return [self.one(c) for c in clips]


class SarvamASR:
    """Sarvam AI speech-to-text REST API: POST https://api.sarvam.ai/speech-to-text (multipart), header api-subscription-key,
    model saaras:v4, language_code like hi-IN / en-IN, at most 30 s of audio per request. Key from $SARVAM_API_KEY only."""
    LANG = {"hi": "hi-IN", "en": "en-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
            "ml": "ml-IN", "pa": "pa-IN", "ur": "ur-IN", "or": "od-IN", "as": "as-IN", "ne": "ne-IN", "auto": "unknown"}

    def __init__(self, model, lang, mode="transcribe"):
        import requests
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY is not set in the environment")
        self.requests, self.key, self.model, self.mode = requests, key, model, mode
        self.lang = self.LANG.get(lang or "hi", lang if lang and "-" in lang else "hi-IN")
        self.name = f"sarvam {model} ({self.lang}, {mode})"
        self.n_calls = 0

    def one(self, x):
        import io
        import time as _t
        import soundfile as sf
        if len(x) < 1600:
            return ""
        buf = io.BytesIO()
        sf.write(buf, x[: 16000 * 30], 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        data = {"model": self.model, "language_code": self.lang, "mode": self.mode}
        for attempt in range(6):
            try:
                r = self.requests.post("https://api.sarvam.ai/speech-to-text", data=data, files={"file": ("clip.wav", buf.getvalue(), "audio/wav")},
                                       headers={"api-subscription-key": self.key}, timeout=120)
            except Exception as e:
                log(f"sarvam request error {e!r}, retry {attempt + 1}")
                _t.sleep(3 * (attempt + 1))
                continue
            self.n_calls += 1
            if r.status_code == 200:
                return str(r.json().get("transcript") or "")
            if r.status_code in (429, 500, 503):
                _t.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"sarvam HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("sarvam: giving up after retries")

    def __call__(self, x, win_s=28.0):
        n = int(win_s * 16000)
        return " ".join(t for t in (self.one(x[i:i + n]) for i in range(0, len(x), n)) if t)

    def batch(self, clips, bs=None):
        return [self.one(c) for c in clips]


class MMSASR:
    """facebook/mms-1b-all (wav2vec2 CTC, 1100+ languages) with the per-language adapter; ISO-639-3 codes (hi -> hin)."""
    ISO3 = {"hi": "hin", "en": "eng", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal",
            "pa": "pan", "ur": "urd", "or": "ory", "as": "asm", "ne": "npi"}

    def __init__(self, name, lang, device):
        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        self.torch, self.device = torch, device
        self.lang = self.ISO3.get(lang or "hi", lang or "hin")
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name, target_lang=self.lang, ignore_mismatched_sizes=True).to(device).eval()
        self.proc.tokenizer.set_target_lang(self.lang)
        self.model.load_adapter(self.lang)
        self.name = f"{name} ({self.lang})"

    def one(self, x):
        torch = self.torch
        if len(x) < 1600:
            return ""
        inp = self.proc(x, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inp).logits
        return self.proc.batch_decode(logits.argmax(-1))[0]

    def __call__(self, x, win_s=25.0):
        n = int(win_s * 16000)
        return " ".join(t for t in (self.one(x[i:i + n]) for i in range(0, len(x), n)) if t)

    def batch(self, clips, bs=None):
        return [self.one(c) for c in clips]


def read_gt(gt_dir, fid):
    """<gt_dir>/<id>.txt, TSV: file, utt_id, spk, name, start, end, text -> list of dicts sorted by start."""
    p = Path(gt_dir) / f"{fid}.txt"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        f = line.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        try:
            rows.append({"spk": f[2], "start": float(f[4]), "end": float(f[5]), "text": f[6].strip()})
        except ValueError:
            continue
    return sorted(rows, key=lambda r: r["start"])


def gt_seglst(rows, session):
    return [{"session_id": session, "speaker": r["spk"], "start_time": r["start"], "end_time": r["end"],
             "words": normalize_text(r["text"])} for r in rows if r["text"]]


def transcribe_by_turns(asr, x, segs, max_len=28.0, min_len=0.3):
    """Transcribe the (merged) speaker turns of `segs` in time order and join them: the same segmentation is used for
    the original and every anonymized version, so the text metrics compare like with like."""
    turns = sorted((a, b) for ss in merged_turns(segs).values() for a, b in ss if b - a >= min_len)
    clips = []
    for a, b in turns:
        while b - a > max_len:                     # long turns -> fixed sub-windows
            clips.append(x[int(a * 16000):int((a + max_len) * 16000)])
            a += max_len
        clips.append(x[int(a * 16000):int(b * 16000)])
    return " ".join(t for t in asr.batch(clips) if t) if clips else ""


def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[।॥.,!?;:\"'()\[\]\-]+", " ", s)   # danda, punctuation
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align_counts(a, b):
    """Levenshtein alignment of token lists -> (substitutions, deletions, insertions, hits)."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
    S = Dl = I = H = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                H += 1
            else:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            Dl += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, Dl, I, H


def wer_cer(ref, hyp):
    """Returns (WER, CER, WIL). WIL = 1 - H/N_ref * H/N_hyp (word information lost), all vs the reference words."""
    r, h = normalize_text(ref), normalize_text(hyp)
    rw, hw = r.split(), h.split()
    S, D, I, H = align_counts(rw, hw)
    wer = (S + D + I) / max(1, len(rw))
    wil = 1.0 - (H / max(1, len(rw))) * (H / max(1, len(hw)))
    rc, hc = r.replace(" ", ""), h.replace(" ", "")
    cer = edit_distance(rc, hc) / max(1, len(rc))
    return wer, cer, wil


def der(ref_segs, hyp_segs, collar=0.25, frame=0.01):
    """Diarization error rate (ref/hyp: dict spk -> [(start, end)]), NIST-style: collar around reference
    boundaries, overlap counted, best speaker mapping over permutations (fine for <= 5 speakers)."""
    import itertools
    import numpy as np
    end = max([b for v in list(ref_segs.values()) + list(hyp_segs.values()) for _, b in v] + [0.0])
    T = int(end / frame) + 2

    def paint(segs):
        m = np.zeros((len(segs), T), dtype=bool)
        for k, (spk, ss) in enumerate(segs.items()):
            for a, b in ss:
                m[k, int(a / frame):int(b / frame)] = True
        return m

    R, Hm = paint(ref_segs), paint(hyp_segs)
    keep = np.ones(T, dtype=bool)                    # frames outside the collar
    c = int(collar / frame)
    for ss in ref_segs.values():
        for a, b in ss:
            for t in (a, b):
                keep[max(0, int(t / frame) - c):int(t / frame) + c] = False
    R, Hm = R[:, keep], Hm[:, keep]
    total = R.sum()
    if total == 0:
        return None
    K = max(len(R), len(Hm))                          # pad the smaller side with silent speakers
    R = np.concatenate([R, np.zeros((K - len(R), R.shape[1]), dtype=bool)])
    Hm = np.concatenate([Hm, np.zeros((K - len(Hm), Hm.shape[1]), dtype=bool)])
    n_ref, n_hyp = R.sum(0), Hm.sum(0)
    miss_fa = (np.clip(n_ref - n_hyp, 0, None) + np.clip(n_hyp - n_ref, 0, None)).sum()
    best_conf = None
    for perm in itertools.permutations(range(K)):
        correct = (R & Hm[list(perm)]).sum(0)
        conf = (np.minimum(n_ref, n_hyp) - correct).sum()
        best_conf = conf if best_conf is None else min(best_conf, conf)
    return float((miss_fa + best_conf) / total)


def merged_turns(segs, gap=0.3):
    """dict spk -> [(a, b)] -> same dict with turns closer than `gap` merged (fewer, longer ASR calls)."""
    out = {}
    for spk, ss in segs.items():
        cur = []
        for a, b in sorted(ss):
            if cur and a - cur[-1][1] <= gap:
                cur[-1] = (cur[-1][0], max(cur[-1][1], b))
            else:
                cur.append((a, b))
        out[spk] = cur
    return out


def seglst(asr, x, segs, session, min_len=0.6):
    """Speaker-attributed transcript in meeteval SegLST form: one entry per (merged) speaker turn."""
    items, clips = [], []
    for spk, ss in merged_turns(segs).items():
        for a, b in ss:
            if b - a < min_len:
                continue
            items.append({"session_id": session, "speaker": spk, "start_time": float(a), "end_time": float(b)})
            clips.append(x[int(a * 16000):int(b * 16000)])
    texts = asr.batch(clips) if clips else []
    for it, txt in zip(items, texts):
        it["words"] = normalize_text(txt)
    return items


def tcp_cp_wer(ref_lst, hyp_lst, collar=5.0):
    """(cpWER, tcpWER) with meeteval; None if meeteval is not installed."""
    try:
        import meeteval
        from meeteval.io import SegLST
    except Exception:
        return None, None
    ref, hyp = SegLST(ref_lst), SegLST(hyp_lst)

    def rate(res):
        if hasattr(res, "error_rate"):
            return float(res.error_rate)
        vals = list(res.values()) if isinstance(res, dict) else [res]
        return float(sum(v.errors for v in vals) / max(1, sum(v.length for v in vals)))

    cp = rate(meeteval.wer.cpwer(reference=ref, hypothesis=hyp))
    tcp = rate(meeteval.wer.tcpwer(reference=ref, hypothesis=hyp, collar=collar))
    return cp, tcp


def pitch_class(f0):
    return None if f0 is None else ("high" if f0 >= 165.0 else "low")


class DNSMOS:
    """P.835 DNSMOS (microsoft/DNS-Challenge). sig_bak_ovr.onnx takes 9.01 s of 16 kHz audio."""
    INPUT_LENGTH = 9.01

    def __init__(self, model_dir):
        import onnxruntime as ort
        import numpy as np
        self.np = np
        d = Path(model_dir)
        self.sess = ort.InferenceSession(str(d / "sig_bak_ovr.onnx"), providers=["CPUExecutionProvider"])
        p808 = d / "model_v8.onnx"
        self.p808 = ort.InferenceSession(str(p808), providers=["CPUExecutionProvider"]) if p808.exists() else None
        self.p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self.p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self.p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def _mel(self, audio):
        import librosa
        m = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(m, ref=self.np.max) + 40) / 40).T

    def __call__(self, x, max_windows=60):
        np = self.np
        fs = 16000
        n = int(self.INPUT_LENGTH * fs)
        audio = x
        while len(audio) < n:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        idxs = np.linspace(0, hops - 1, min(hops, max_windows)).astype(int)
        sig, bak, ovr, p808 = [], [], [], []
        for i in idxs:
            seg = audio[int(i * fs):int((i + self.INPUT_LENGTH) * fs)]
            if len(seg) < n:
                continue
            raw = self.sess.run(None, {"input_1": seg.astype("float32")[None, :]})[0][0]
            sig.append(self.p_sig(raw[0])); bak.append(self.p_bak(raw[1])); ovr.append(self.p_ovr(raw[2]))
            if self.p808 is not None:
                try:
                    feat = self._mel(seg[:-160]).astype("float32")[None, :, :]
                    p808.append(float(self.p808.run(None, {"input_1": feat})[0][0][0]))
                except Exception:
                    pass
        out = {"SIG": float(np.mean(sig)), "BAK": float(np.mean(bak)), "OVRL": float(np.mean(ovr))}
        if p808:
            out["P808_MOS"] = float(np.mean(p808))
        return out


def pitch_stats(x):
    import numpy as np
    import torch
    import torchaudio
    xt = torch.from_numpy(x[: 16000 * 90])[None]
    if xt.size(-1) < 16000:
        return None
    f0 = torchaudio.functional.detect_pitch_frequency(xt, 16000, frame_time=0.02, freq_low=60, freq_high=500)[0].numpy()
    hop = xt.size(-1) // len(f0)
    e = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(len(f0))])
    v = f0[(e > 0.02) & (f0 > 60) & (f0 < 500)]
    if len(v) < 10:
        return None
    return {"f0_median": float(np.median(v)), "f0_range": float(np.percentile(v, 90) - np.percentile(v, 10))}


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    # absolute paths BEFORE chdir into the seed-vc checkout
    args.orig_dir, args.rttm_dir, args.out = (str(Path(p).resolve()) for p in (args.orig_dir, args.rttm_dir, args.out))
    args.systems = [f"{s.split('=', 1)[0]}={Path(s.split('=', 1)[1]).resolve()}" for s in args.systems]
    if args.dnsmos_dir:
        args.dnsmos_dir = str(Path(args.dnsmos_dir).resolve())
    root = Path(args.seedvc_root).resolve()
    os.chdir(root)
    # same cache the anonymizer/staging use, unless the caller points HF_HUB_CACHE elsewhere
    os.environ.setdefault("HF_HUB_CACHE", str(root / "checkpoints" / "hf_cache"))
    import torch
    import numpy as np
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        if not gt_dir.is_dir() or not any(gt_dir.glob("*.txt")):
            sys.exit(f"--gt-dir {gt_dir} has no <id>.txt transcripts (was data/GT uploaded?)")
    systems = {}
    for s in args.systems:
        name, val = s.split("=", 1)
        systems[name] = val if "{id}" in val else str(Path(val) / "{id}.wav")
    first = systems[next(iter(systems))]
    if args.files:
        files = args.files
    elif "{id}.wav" in first and Path(first).parent.exists():
        files = sorted(p.stem for p in Path(first).parent.glob("*.wav"))
    else:
        sys.exit("--files is required when the first system is a pattern")
    log(f"{len(systems)} systems x {len(files)} files -> {out_dir}")

    if args.text_only:
        class _NoSV:
            def embed(self, x):
                return None

            @staticmethod
            def cos(a, b):
                return None
        sv = _NoSV()
        args.no_dnsmos = True
        log("text-only mode: speaker verification and DNSMOS skipped")
    else:
        sv = SpeakerVerifier(args.sv_model, device)
    asr = None
    if not args.no_asr:
        try:
            if args.asr_backend == "indic":
                asr = IndicASR(args.indic_model, args.asr_lang, device, args.indic_decoder)
                log(f"ASR: {args.indic_model} lang={asr.lang} decoder={args.indic_decoder}")
            elif args.asr_backend == "sarvam":
                asr = SarvamASR(args.sarvam_model, args.asr_lang, args.sarvam_mode)
                log(f"ASR: {asr.name}")
            elif args.asr_backend == "mms":
                asr = MMSASR(args.mms_model, args.asr_lang, device)
                log(f"ASR: {asr.name}")
            else:
                asr = ASR(args.asr_model, args.asr_lang, device)
                log(f"ASR: {args.asr_model} lang={args.asr_lang}")
        except Exception as e:
            log(f"ASR unavailable ({e!r}); WER/CER skipped. Stage it with stage_models_seedvc.py --with-eval / --with-indic")
    def speaker_gender(x, ss, f0):
        return pitch_class(f0)      # pitch-class rule (low <165 Hz / high); no gender classifier by design

    def transcribe(x, segs):
        """Text of a whole recording: turn-wise for the Indic backend (or whenever turns exist), whole-file for Whisper."""
        if isinstance(asr, (IndicASR, SarvamASR, MMSASR)) and segs:
            return transcribe_by_turns(asr, x, segs)
        return asr(x[: int(args.max_asr_sec * 16000)] if args.max_asr_sec else x)
    dns = None
    if not args.no_dnsmos:
        ddir = args.dnsmos_dir or str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "dnsmos")
        try:
            dns = DNSMOS(ddir)
        except Exception as e:
            log(f"DNSMOS unavailable ({e!r}); skipping. Stage the onnx files with stage_models_seedvc.py --with-eval")

    records = []
    for fid in files:
        orig = load_16k(Path(args.orig_dir) / f"{fid}.wav")
        segs = speaker_segments(args.rttm_dir, fid) or {}
        if not segs:
            segs = {"spk0": [(0.0, len(orig) / 16000)]}       # no RTTM: single-speaker clip, the whole file is one turn
        spk_order = sorted(segs, key=lambda k: -sum(b - a for a, b in segs[k]))[:2]
        orig_emb = {k: sv.embed(speaker_audio(orig, segs[k])) for k in spk_order}
        # anchors for the similarity scale on THIS data: same speaker vs itself (two halves) and the two real speakers
        same = []
        for k in spk_order:
            sa = speaker_audio(orig, segs[k], max_sec=180)
            if sa is not None and len(sa) > 16000 * 8:
                h = len(sa) // 2
                s = SpeakerVerifier.cos(sv.embed(sa[:h]), sv.embed(sa[h:]))
                if s is not None:
                    same.append(s)
        same_anchor = float(np.mean(same)) if same else None
        diff_anchor = SpeakerVerifier.cos(orig_emb[spk_order[0]], orig_emb[spk_order[1]]) if len(spk_order) == 2 else None

        def privacy_score(sim):
            """1.0 = as similar to the source as the source is to itself; 0.0 = as different as another real speaker."""
            if sim is None or same_anchor is None or diff_anchor is None or same_anchor - diff_anchor < 0.02:
                return None
            return float((sim - diff_anchor) / (same_anchor - diff_anchor))

        gt_rows = read_gt(args.gt_dir, fid) if args.gt_dir else None
        if args.gt_dir and not gt_rows:
            log(f"{fid}: WARNING no ground-truth transcript, text metrics fall back to the ASR of the original")
        orig_asr_txt = transcribe(orig, segs) if asr else None
        if gt_rows:
            orig_txt = " ".join(r["text"] for r in gt_rows if r["text"])      # human reference
            gt_lst = gt_seglst(gt_rows, fid)
        else:
            orig_txt, gt_lst = orig_asr_txt, None                              # fall back: ASR of the original is the reference
        orig_dns = dns(orig) if dns else None
        orig_pitch = {k: pitch_stats(speaker_audio(orig, segs[k])) for k in spk_order}
        # system diarization (stage 1) for this file: DER, and the segmentation the anonymized audio is attributed with
        hyp_segs, file_der = None, None
        if args.hyp_rttm_dir:
            hyp_segs = speaker_segments(args.hyp_rttm_dir, fid)
            if hyp_segs and segs:
                file_der = der(segs, hyp_segs)
        ref_lst = gt_lst if gt_lst else (seglst(asr, orig, segs, fid) if (asr and segs and not args.no_tcp) else None)
        if gt_lst and args.no_tcp:
            ref_lst = None
        orig_gender = {k: speaker_gender(orig, segs[k], orig_pitch[k]["f0_median"] if orig_pitch.get(k) else None)
                       for k in spk_order}
        orig_err = wer_cer(orig_txt, orig_asr_txt) if (gt_rows and orig_asr_txt is not None) else (0.0, 0.0, 0.0)
        rec0 = {"file": fid, "system": "original", "sv_sim_to_source": same_anchor, "sv_privacy_score": 1.0,
                "sv_same_speaker_anchor": same_anchor, "sv_diff_speaker_anchor": diff_anchor,
                "sv_sim_between_speakers": diff_anchor, "der": file_der,
                "n_speakers_ref": len(segs), "n_speakers_hyp": len(hyp_segs) if hyp_segs else None,
                "wer": orig_err[0], "cer": orig_err[1], "wil": orig_err[2], "gender_match": 1.0, "dnsmos": orig_dns,
                "pitch": orig_pitch, "gender": orig_gender, "reference": "human GT" if gt_rows else "ASR of original",
                "transcript": orig_asr_txt, "gt_text": orig_txt if gt_rows else None}
        if ref_lst is not None and asr and hyp_segs:   # diarization-only tcpWER: original audio attributed with the system RTTM
            rec0["cpwer"], rec0["tcpwer"] = tcp_cp_wer(ref_lst, seglst(asr, orig, hyp_segs, fid), args.tcp_collar)
        records.append(rec0)
        log(f"{fid} original: same-speaker anchor={same_anchor} different-speaker anchor={diff_anchor} "
            f"DER={file_der} tcpWER(diar only)={rec0.get('tcpwer')} dnsmos={orig_dns}")
        for name, pat in systems.items():
            path = Path(pat.format(id=fid))
            rec = {"file": fid, "system": name, "path": str(path)}
            if not path.exists():
                rec["error"] = "missing"
                records.append(rec)
                continue
            try:
                y = load_16k(path)
                if len(y) != len(orig):
                    rec["length_mismatch_s"] = round((len(y) - len(orig)) / 16000, 2)
                    y = y[: len(orig)] if len(y) > len(orig) else np.pad(y, (0, len(orig) - len(y)))
                an_emb = {k: sv.embed(speaker_audio(y, segs[k])) for k in spk_order}
                sims = [SpeakerVerifier.cos(orig_emb[k], an_emb[k]) for k in spk_order]
                sims = [s for s in sims if s is not None]
                rec["sv_sim_to_source"] = float(np.mean(sims)) if sims else None
                rec["sv_privacy_score"] = privacy_score(rec["sv_sim_to_source"])
                rec["sv_sim_to_source_per_spk"] = {k: SpeakerVerifier.cos(orig_emb[k], an_emb[k]) for k in spk_order}
                rec["sv_sim_between_speakers"] = SpeakerVerifier.cos(an_emb[spk_order[0]], an_emb[spk_order[1]]) if len(spk_order) == 2 else None
                if asr:
                    hyp = transcribe(y, segs)
                    rec["wer"], rec["cer"], rec["wil"] = wer_cer(orig_txt, hyp)
                    rec["transcript"] = hyp
                    if ref_lst is not None:
                        # full-system speaker-attributed error: anonymized audio, segmented by the system's own diarization
                        sys_segs = hyp_segs or segs
                        rec["cpwer"], rec["tcpwer"] = tcp_cp_wer(ref_lst, seglst(asr, y, sys_segs, fid), args.tcp_collar)
                rec["dnsmos"] = dns(y) if dns else None
                rec["pitch"] = {k: pitch_stats(speaker_audio(y, segs[k])) for k in spk_order}
                rec["pitch_ratio"] = {k: {"median": rec["pitch"][k]["f0_median"] / orig_pitch[k]["f0_median"],
                                          "range": rec["pitch"][k]["f0_range"] / orig_pitch[k]["f0_range"]}
                                      for k in spk_order if rec["pitch"].get(k) and orig_pitch.get(k)}
                rec["gender"] = {k: speaker_gender(y, segs[k], rec["pitch"][k]["f0_median"] if rec["pitch"].get(k) else None)
                                 for k in spk_order}
                gm = [rec["gender"][k] == orig_gender[k] for k in spk_order if rec["gender"].get(k) and orig_gender.get(k)]
                rec["gender_match"] = float(np.mean(gm)) if gm else None
                _sim = rec['sv_sim_to_source']
                log(f"{fid} {name}: sim_to_source={'-' if _sim is None else f'{_sim:.3f}'} privacy_score={rec['sv_privacy_score']} "
                    f"between={rec['sv_sim_between_speakers']} wer={rec.get('wer')} wil={rec.get('wil')} "
                    f"tcpwer={rec.get('tcpwer')} gender_match={rec['gender_match']} dnsmos={rec['dnsmos']}")
            except Exception as e:
                import traceback
                rec["error"] = repr(e)
                rec["traceback"] = traceback.format_exc()
                log(f"{fid} {name}: FAILED {e!r}")
            records.append(rec)

    write_json(out_dir / "eval_records.json", records)

    # ---- summary table per system (mean over files) ----
    def mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    summary = {}
    for name in ["original"] + list(systems):
        rs = [r for r in records if r["system"] == name and "error" not in r]
        if not rs:
            continue
        summary[name] = {
            "n_files": len(rs),
            "sv_sim_to_source": mean(r.get("sv_sim_to_source") for r in rs),
            "sv_privacy_score": mean(r.get("sv_privacy_score") for r in rs),
            "sv_sim_between_speakers": mean(r.get("sv_sim_between_speakers") for r in rs),
            "der": mean(r.get("der") for r in records if r["system"] == "original"),
            "wer": mean(r.get("wer") for r in rs), "cer": mean(r.get("cer") for r in rs),
            "wil": mean(r.get("wil") for r in rs),
            "cpwer": mean(r.get("cpwer") for r in rs), "tcpwer": mean(r.get("tcpwer") for r in rs),
            "gender_match": mean(r.get("gender_match") for r in rs),
            "dnsmos_OVRL": mean((r.get("dnsmos") or {}).get("OVRL") for r in rs),
            "dnsmos_SIG": mean((r.get("dnsmos") or {}).get("SIG") for r in rs),
            "dnsmos_BAK": mean((r.get("dnsmos") or {}).get("BAK") for r in rs),
            "pitch_median_ratio": mean(v["median"] for r in rs for v in (r.get("pitch_ratio") or {}).values()),
            "pitch_range_ratio": mean(v["range"] for r in rs for v in (r.get("pitch_ratio") or {}).values()),
        }
    write_json(out_dir / "eval_summary.json", summary)

    f = lambda v, nd=3: "-" if v is None else f"{v:.{nd}f}"
    der_mean = summary.get("original", {}).get("der")
    lines = []
    if der_mean is not None:
        lines += [f"Diarization (stage 1, {args.hyp_rttm_dir}): mean DER = {der_mean:.3f} over {summary['original']['n_files']} files "
                  f"(collar 0.25 s, overlap scored). The 'original' row's cpWER/tcpWER is the diarization-only error: original audio "
                  f"attributed with the system RTTM; every other row is the full system (anonymized audio + system RTTM).", ""]
    lines += ["| system | files | privacy score (1 = still the same person, 0 = as different as another real speaker, lower is better) | "
              "raw speaker sim to source | sim between the 2 anonymized speakers (lower = distinct; 'original' row = the 2 real speakers) | "
              "gender preserved | WER vs original ASR | CER | WIL | cpWER | tcpWER | DNSMOS OVRL | SIG | BAK | pitch median ratio | pitch range ratio |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['n_files']} | {f(s['sv_privacy_score'], 2)} | {f(s['sv_sim_to_source'])} | {f(s['sv_sim_between_speakers'])} | "
                     f"{f(s['gender_match'], 2)} | {f(s['wer'])} | {f(s['cer'])} | {f(s['wil'])} | {f(s['cpwer'])} | {f(s['tcpwer'])} | "
                     f"{f(s['dnsmos_OVRL'], 2)} | {f(s['dnsmos_SIG'], 2)} | {f(s['dnsmos_BAK'], 2)} | "
                     f"{f(s['pitch_median_ratio'], 2)} | {f(s['pitch_range_ratio'], 2)} |")
    lines.append("")
    ref_note = (f"All text metrics are against the HUMAN transcripts in {args.gt_dir}; the 'original' row is therefore the ASR's own "
                "error on the untouched audio, and every other row's excess over it is what anonymization cost."
                if args.gt_dir else
                "All text metrics use the ASR transcript of the ORIGINAL as reference (no human transcripts), so they measure change "
                "caused by the system, not absolute ASR accuracy.")
    asr_note = (f"ASR: {args.indic_model} ({args.indic_decoder}, lang {args.asr_lang}), transcribed turn by turn on the reference "
                "segmentation for every system alike." if args.asr_backend == "indic" else f"ASR: {args.asr_model}.")
    gender_note = "Gender preserved: fraction of speakers whose anonymized median pitch falls in the same class (split 165 Hz) as the original."
    lines.append(gender_note + " WIL = word information lost (1 - hits/N_ref * hits/N_hyp). cpWER/tcpWER (meeteval): "
                 f"speaker-attributed word error, tcpWER with a {args.tcp_collar:.0f} s time collar, so they penalise diarization errors "
                 "as well as content drift. " + ref_note + " " + asr_note)
    lines.append("Reading guide: the privacy score places the anonymized speaker on a scale anchored on this very recording: "
                 "1.0 = as similar to the source as two halves of the source speaker's own speech, 0.0 = as different as the other "
                 "real speaker in the conversation (WavLM-SV cosine; raw values are compressed on noisy Hindi field audio, so read "
                 "the anchored score, not the raw one). Scores below 0 mean the anonymized voice is further from the source than the "
                 "other real speaker in the same room is; part of that distance can be the vocoder's channel rather than identity, so "
                 "compare systems against each other rather than reading the score as a probability. "
                 "Distinctness wants the two anonymized speakers no more similar than the two "
                 "real ones. WER/CER are against the ASR transcript of the ORIGINAL, so they measure content drift caused by "
                 "anonymization, not absolute ASR accuracy. DNSMOS: 1-5, higher is better. Pitch ratios of 1.0 mean the speaker's "
                 "pitch level and expressiveness were preserved.")
    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    log(f"wrote {out_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
