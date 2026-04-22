import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from melotts_py.download_utils import load_or_download_config, load_or_download_model
from melotts_py.models import SynthesizerTrn


DEFAULT_BERT_MODEL = "bert-base-uncased"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class BertExportWrapper(torch.nn.Module):
    def __init__(self, model: AutoModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        return hidden


class MeloTTSInferWrapper(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale,
        length_scale,
        noise_scale_w,
        sdp_ratio,
    ):
        audio, *_ = self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tone,
            language=language,
            bert=bert,
            ja_bert=ja_bert,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sdp_ratio=sdp_ratio,
        )
        return audio


def export_bert(output_dir: Path, opset: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BERT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_BERT_MODEL)
    model.eval()

    vocab_path = output_dir / "bert-base-uncased-vocab.txt"
    with vocab_path.open("w", encoding="utf-8") as f:
        for token, _ in sorted(tokenizer.vocab.items(), key=lambda item: item[1]):
            f.write(token + "\n")

    wrapper = BertExportWrapper(model)
    wrapper.eval()
    sample = tokenizer("Hello from MeloTTS.", return_tensors="pt")
    output_path = output_dir / "bert_base_uncased.onnx"

    torch.onnx.export(
        wrapper,
        (
            sample["input_ids"],
            sample["attention_mask"],
            sample.get("token_type_ids", torch.zeros_like(sample["input_ids"])),
        ),
        str(output_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {1: "token_count"},
            "attention_mask": {1: "token_count"},
            "token_type_ids": {1: "token_count"},
            "last_hidden_state": {1: "token_count"},
        },
        opset_version=opset,
        dynamo=False,
    )


def build_melo_model(device: str, use_hf: bool, config_path: str | None, ckpt_path: str | None):
    hps = load_or_download_config("EN", use_hf=use_hf, config_path=config_path)
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model,
    ).to(device)
    checkpoint = load_or_download_model("EN", device, use_hf=use_hf, ckpt_path=ckpt_path)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, hps


def export_melo_acoustic(
    output_dir: Path,
    device: str,
    opset: int,
    use_hf: bool,
    config_path: str | None,
    ckpt_path: str | None,
) -> None:
    model, hps = build_melo_model(device, use_hf, config_path, ckpt_path)
    wrapper = MeloTTSInferWrapper(model)
    wrapper.eval()

    seq_len = 32
    x = torch.randint(0, len(hps.symbols), (1, seq_len), dtype=torch.long, device=device)
    x_lengths = torch.tensor([seq_len], dtype=torch.long, device=device)
    sid = torch.tensor([0], dtype=torch.long, device=device)
    tone = torch.randint(0, hps.num_tones, (1, seq_len), dtype=torch.long, device=device)
    language = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
    bert = torch.zeros((1, 1024, seq_len), dtype=torch.float32, device=device)
    ja_bert = torch.zeros((1, 768, seq_len), dtype=torch.float32, device=device)
    noise_scale = torch.tensor(0.6, dtype=torch.float32, device=device)
    length_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    noise_scale_w = torch.tensor(0.8, dtype=torch.float32, device=device)
    sdp_ratio = torch.tensor(0.2, dtype=torch.float32, device=device)

    output_path = output_dir / "melotts_en_infer.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (
                x,
                x_lengths,
                sid,
                tone,
                language,
                bert,
                ja_bert,
                noise_scale,
                length_scale,
                noise_scale_w,
                sdp_ratio,
            ),
            str(output_path),
            input_names=[
                "x",
                "x_lengths",
                "sid",
                "tone",
                "language",
                "bert",
                "ja_bert",
                "noise_scale",
                "length_scale",
                "noise_scale_w",
                "sdp_ratio",
            ],
            output_names=["audio"],
            dynamic_axes={
                "x": {1: "text_length"},
                "tone": {1: "text_length"},
                "language": {1: "text_length"},
                "bert": {2: "text_length"},
                "ja_bert": {2: "text_length"},
                "audio": {2: "sample_count"},
            },
            opset_version=opset,
            dynamo=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export English BERT and MeloTTS models to ONNX.")
    parser.add_argument("--output-dir", default="models", help="Output directory for ONNX artifacts.")
    parser.add_argument("--device", default="cpu", help="Export device, for example cpu or cuda:0.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--no-hf", action="store_true", help="Use direct download URLs instead of Hugging Face.")
    parser.add_argument("--config-path", default=None, help="Optional local MeloTTS config.json path.")
    parser.add_argument("--ckpt-path", default=None, help="Optional local MeloTTS checkpoint.pth path.")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT export.")
    parser.add_argument("--skip-acoustic", action="store_true", help="Skip acoustic model export.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_bert:
        export_bert(output_dir, args.opset)
        print(f"Exported BERT ONNX to: {output_dir / 'bert_base_uncased.onnx'}")
        print(f"Saved BERT vocab to: {output_dir / 'bert-base-uncased-vocab.txt'}")

    if not args.skip_acoustic:
        export_melo_acoustic(
            output_dir=output_dir,
            device=args.device,
            opset=args.opset,
            use_hf=not args.no_hf,
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
        )
        print(f"Exported MeloTTS acoustic ONNX to: {output_dir / 'melotts_en_infer.onnx'}")


if __name__ == "__main__":
    main()
