#!/usr/bin/env python3
import argparse
from pathlib import Path


SENTINEL = "def _pad_short_mel(mel):"


def _insert_once(text, needle, insert):
    if insert in text:
        return text, False
    if needle not in text:
        raise ValueError(f"Missing needle: {needle!r}")
    return text.replace(needle, insert, 1), True


def apply_patch(target_path: Path) -> bool:
    text = target_path.read_text()
    changed = False

    if SENTINEL not in text:
        marker = "        # keep overlap mel and hift cache\n"
        helper = (
            "        # keep overlap mel and hift cache\n"
            "        def _pad_short_mel(mel):\n"
            "            min_mel_frames = 3\n"
            "            pad_frames = max(0, min_mel_frames - mel.shape[2])\n"
            "            if pad_frames > 0:\n"
            "                mel = F.pad(mel, (0, pad_frames), mode='constant', value=0)\n"
            "            return mel, pad_frames\n"
            "\n"
        )
        text, did = _insert_once(text, marker, helper)
        changed = changed or did

    pad_line = "            tts_mel, mel_pad_frames = _pad_short_mel(tts_mel)\n"

    if pad_line not in text:
        text, did = _insert_once(
            text,
            "        if finalize is False:\n"
            "            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)\n",
            "        if finalize is False:\n"
            + pad_line
            + "            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)\n",
        )
        changed = changed or did

    trim_block_nonfinal = (
        "            if mel_pad_frames > 0:\n"
        "                trim_samples = mel_pad_frames * 480\n"
        "                if tts_speech.shape[1] > trim_samples:\n"
        "                    tts_speech = tts_speech[:, :-trim_samples]\n"
        "                    tts_source = tts_source[:, :, :-trim_samples]\n"
        "                else:\n"
        "                    tts_speech = tts_speech[:, :0]\n"
        "                    tts_source = tts_source[:, :, :0]\n"
        "\n"
    )
    anchor_nonfinal = (
        "            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],\n"
    )
    text, did = _insert_once(text, anchor_nonfinal, trim_block_nonfinal + anchor_nonfinal)
    changed = changed or did

    finalize_anchor = (
        "        else:\n"
        "            if speed != 1.0:\n"
        "                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'\n"
        "                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')\n"
        "            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)\n"
    )
    if finalize_anchor in text:
        text, did = _insert_once(
            text,
            finalize_anchor,
            "        else:\n"
            "            if speed != 1.0:\n"
            "                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'\n"
            "                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')\n"
            + pad_line
            + "            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)\n",
        )
        changed = changed or did

    trim_block_final = (
        "            if mel_pad_frames > 0:\n"
        "                trim_samples = mel_pad_frames * 480\n"
        "                if tts_speech.shape[1] > trim_samples:\n"
        "                    tts_speech = tts_speech[:, :-trim_samples]\n"
        "                else:\n"
        "                    tts_speech = tts_speech[:, :0]\n"
        "\n"
    )
    anchor_final = (
        "            if self.hift_cache_dict[uuid] is not None:\n"
        "                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)\n"
        "        return tts_speech\n"
    )
    text, did = _insert_once(
        text,
        anchor_final,
        "            if self.hift_cache_dict[uuid] is not None:\n"
        "                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)\n"
        + trim_block_final
        + "        return tts_speech\n",
    )
    changed = changed or did

    if changed:
        target_path.write_text(text)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch CosyVoice2 token2wav for short mel padding.")
    parser.add_argument(
        "--target",
        required=True,
        help="Path to token2wav/1/model.py inside container",
    )
    args = parser.parse_args()
    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")
    changed = apply_patch(target)
    print("patched" if changed else "already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
