#!/usr/bin/env python
"""
Download all Seed-VC checkpoints (DiT seed-uvit-whisper-small-wavenet, CAM++,
BigVGAN v2 22 kHz, openai/whisper-small) into <seedvc-root>/checkpoints by calling
seed-vc's own inference.load_models() once. RUN ON THE MASTER NODE inside the
`seedvc` env. Loads on CPU; nothing heavy is computed.

Optional --smoke-wav converts 3 s of audio on CPU with 4 diffusion steps.
"""
import os
import sys
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedvc-root", required=True)
    ap.add_argument("--smoke-wav", default=None)
    ap.add_argument("--with-f0", action="store_true",
                    help="also download the F0-conditioned 44.1 kHz model, RMVPE and BigVGAN-44k (for --f0-condition)")
    ap.add_argument("--with-v2", action="store_true",
                    help="also download the V2 model (v2/cfm_small.pth, v2/ar_base.pth, ASTRAL quantizers, hubert-large-ll60k)")
    ap.add_argument("--with-eval", action="store_true",
                    help="also download the evaluation models (WavLM-SV, Hindi-capable Whisper ASR, DNSMOS onnx)")
    ap.add_argument("--sv-model", default="microsoft/wavlm-base-plus-sv")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3")
    ap.add_argument("--with-indic", action="store_true", help="stage ai4bharat IndicConformer (gated: export HF_TOKEN first)")
    ap.add_argument("--indic-model", default="ai4bharat/indic-conformer-600m-multilingual")
    ap.add_argument("--dnsmos-dir", default=None, help="where to put DNSMOS onnx files (default $HF_HOME/dnsmos)")
    args = ap.parse_args()

    root = Path(args.seedvc_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["HF_HUB_CACHE"] = str(root / "checkpoints" / "hf_cache")

    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

    import inference as svc
    ns = argparse.Namespace(f0_condition=False, checkpoint=None, config=None, fp16=True)
    print("== load_models() - downloads into", root / "checkpoints")
    model, semantic_fn, f0_fn, vocoder_fn, campplus, to_mel, mel_args = svc.load_models(ns)
    print("   OK. mel args:", mel_args)
    if args.with_f0:
        print("== load_models(f0_condition=True) - 44.1 kHz DiT + RMVPE + BigVGAN 44k")
        svc.load_models(argparse.Namespace(f0_condition=True, checkpoint=None, config=None, fp16=True))
        print("   OK (F0 models staged)")

    if args.with_v2:
        print("== V2: instantiate configs/v2/vc_wrapper.yaml and load_checkpoints() (downloads ~3 GB)")
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig
        cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
        w = instantiate(cfg)
        w.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        print("   OK (V2 CFM + AR + ASTRAL quantizers + hubert-large-ll60k staged)")
        del w

    if args.with_eval:
        print(f"== eval models: {args.sv_model}, {args.asr_model}, DNSMOS onnx")
        from transformers import AutoFeatureExtractor, WavLMForXVector, WhisperProcessor, WhisperForConditionalGeneration
        AutoFeatureExtractor.from_pretrained(args.sv_model)
        WavLMForXVector.from_pretrained(args.sv_model)
        print("   speaker-verification model OK")
        WhisperProcessor.from_pretrained(args.asr_model)
        WhisperForConditionalGeneration.from_pretrained(args.asr_model)
        print("   ASR model OK")
        import urllib.request
        ddir = Path(args.dnsmos_dir or Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "dnsmos")
        ddir.mkdir(parents=True, exist_ok=True)
        for fn in ("sig_bak_ovr.onnx", "model_v8.onnx"):
            dst = ddir / fn
            if not dst.exists():
                url = f"https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/{fn}"
                try:
                    urllib.request.urlretrieve(url, dst)
                    print(f"   downloaded {fn} -> {dst}")
                except Exception as e:
                    print(f"   WARNING: could not download {fn} ({e!r}); DNSMOS will be skipped")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print("   WARNING: onnxruntime not installed in this env -> pip install onnxruntime")

    if args.with_indic:
        print(f"== Indic ASR: {args.indic_model} (gated; token from HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'NOT SET'})")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.indic_model, token=os.environ.get("HF_TOKEN"))
        try:
            import onnxruntime as ort
            print(f"   onnxruntime {ort.__version__} providers={ort.get_available_providers()}")
        except ImportError:
            print("   WARNING: onnxruntime missing -> pip install onnxruntime-gpu==1.20.1 onnx==1.20.1")
        from transformers import AutoModel
        import torch
        m = AutoModel.from_pretrained(args.indic_model, trust_remote_code=True)
        sr_wav = torch.zeros(1, 16000 * 2)
        if args.smoke_wav:
            import soundfile as sf
            import torchaudio
            x, sr = sf.read(args.smoke_wav, dtype="float32", always_2d=True)
            sr_wav = torch.from_numpy(x.mean(axis=1))[None][:, : sr * 10]
            if sr != 16000:
                sr_wav = torchaudio.functional.resample(sr_wav, sr, 16000)
        out = m(sr_wav, "hi", "ctc")
        print(f"   OK ctc transcript of smoke clip: {str(out)[:120]!r}")
        del m

    total = 0
    for p in (root / "checkpoints").rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    print(f"   checkpoints dir size: {total / 2**30:.2f} GB")

    if args.smoke_wav:
        import soundfile as sf
        import torchaudio
        import numpy as np
        sr = mel_args["sampling_rate"]
        x, xsr = sf.read(args.smoke_wav, dtype="float32", frames=16000 * 3, always_2d=True)
        w16 = torch.from_numpy(x[:, 0])[None]
        if xsr != 16000:
            w16 = torchaudio.functional.resample(w16, xsr, 16000)
        ref_dir = root / "examples" / "reference"
        ref_path = sorted(ref_dir.glob("*.wav"))[0]
        r, rsr = sf.read(str(ref_path), dtype="float32", always_2d=True)
        ref = torchaudio.functional.resample(torch.from_numpy(r[:, 0])[None], rsr, sr)[:, : sr * 5]
        ref16 = torchaudio.functional.resample(ref, sr, 16000)
        with torch.inference_mode():
            S_alt = semantic_fn(w16)
            S_ori = semantic_fn(ref16)
            src = torchaudio.functional.resample(w16, 16000, sr)
            mel, mel2 = to_mel(src), to_mel(ref)
            feat2 = torchaudio.compliance.kaldi.fbank(ref16, num_mel_bins=80, dither=0, sample_frequency=16000)
            style2 = campplus((feat2 - feat2.mean(0, keepdim=True)).unsqueeze(0))
            cond = model.length_regulator(S_alt, ylens=torch.LongTensor([mel.size(2)]), n_quantizers=3, f0=None)[0]
            pc = model.length_regulator(S_ori, ylens=torch.LongTensor([mel2.size(2)]), n_quantizers=3, f0=None)[0]
            cat = torch.cat([pc, cond], dim=1)
            vc = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]), mel2, style2, None, 4, inference_cfg_rate=0.7)
            wav = vocoder_fn(vc[:, :, mel2.size(-1):].float()).squeeze().numpy()
        out = root / "checkpoints" / "smoke_test_output.wav"
        sf.write(str(out), wav, sr)
        print(f"   smoke conversion OK -> {out} ({len(wav) / sr:.1f}s, ref {ref_path.name})")
    print("== staging complete. Compute nodes can now run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
