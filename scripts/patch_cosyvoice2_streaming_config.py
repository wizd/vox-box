#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def _ensure_dynamic_chunk_strategy(cfg: str) -> str:
    if 'key: "dynamic_chunk_strategy"' in cfg:
        return re.sub(
            r'(key:\s*"dynamic_chunk_strategy"\s*,?\s*\n\s*value:\s*\{string_value:\s*")[^"]*("\})',
            r'\1time_based\2',
            cfg,
            count=1,
        )

    marker = (
        "  {\n"
        "   key: \"model_dir\",\n"
        "   value: {string_value:\"./CosyVoice2-0.5B\"}\n"
        "  }\n"
        "]"
    )
    replacement = (
        "  {\n"
        "   key: \"model_dir\",\n"
        "   value: {string_value:\"./CosyVoice2-0.5B\"}\n"
        "  },\n"
        "  {\n"
        "   key: \"dynamic_chunk_strategy\",\n"
        "   value: {string_value:\"time_based\"}\n"
        "  }\n"
        "]"
    )
    if marker not in cfg:
        raise ValueError("Cannot find model_dir parameter block in cosyvoice2 config")
    return cfg.replace(marker, replacement, 1)


def _set_token2wav_instance_count(cfg: str, count: int) -> str:
    return re.sub(r"(instance_group\s*\[\s*\{\s*count:\s*)\d+", rf"\g<1>{count}", cfg, count=1, flags=re.S)


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch CosyVoice2 streaming config for smoother cadence.")
    parser.add_argument("--cosyvoice2-config", required=True, help="Path to model_repo_cosyvoice2/cosyvoice2/config.pbtxt")
    parser.add_argument("--token2wav-config", required=True, help="Path to model_repo_cosyvoice2/token2wav/config.pbtxt")
    parser.add_argument("--token2wav-count", type=int, default=2, help="token2wav instance_group count")
    args = parser.parse_args()

    cosyvoice2_path = Path(args.cosyvoice2_config)
    token2wav_path = Path(args.token2wav_config)

    if not cosyvoice2_path.exists():
        raise SystemExit(f"Missing cosyvoice2 config: {cosyvoice2_path}")
    if not token2wav_path.exists():
        raise SystemExit(f"Missing token2wav config: {token2wav_path}")

    cosy_cfg = cosyvoice2_path.read_text()
    token2wav_cfg = token2wav_path.read_text()

    cosy_new = _ensure_dynamic_chunk_strategy(cosy_cfg)
    token2wav_new = _set_token2wav_instance_count(token2wav_cfg, args.token2wav_count)

    if cosy_new != cosy_cfg:
        cosyvoice2_path.write_text(cosy_new)
    if token2wav_new != token2wav_cfg:
        token2wav_path.write_text(token2wav_new)

    print("patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
