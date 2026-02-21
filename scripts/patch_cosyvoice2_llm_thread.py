#!/usr/bin/env python3
import argparse
from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise ValueError(f"Missing target snippet: {old!r}")
    return text.replace(old, new, 1), True


def apply_patch(target_path: Path) -> bool:
    text = target_path.read_text()
    changed = False

    text, did = _replace_once(
        text,
        "    def forward_llm(self, input_ids):\n",
        "    def forward_llm(self, input_ids, llm_responses_holder=None):\n",
    )
    changed = changed or did

    text, did = _replace_once(
        text,
        "        llm_responses = llm_request.exec(decoupled=self.decoupled)\n"
        "        if self.decoupled:\n",
        "        llm_responses = llm_request.exec(decoupled=self.decoupled)\n"
        "        if self.decoupled and llm_responses_holder is not None:\n"
        "            llm_responses_holder[0] = llm_responses\n"
        "        if self.decoupled:\n",
    )
    changed = changed or did

    old_llm_thread = (
        "    def _llm_gen_thread(self, generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag):\n"
        "        for generated_ids in generated_ids_iter:\n"
        "            generated_ids = generated_ids.tolist()\n"
        "            if len(generated_ids) == 0:\n"
        "                break\n"
        "            semantic_token_ids_arr.extend(generated_ids)\n"
        "        llm_is_done_flag[0] = True\n"
    )
    new_llm_thread = (
        "    def _llm_gen_thread(self, generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag, llm_error_holder, cancel_flag, llm_responses_holder):\n"
        "        try:\n"
        "            for generated_ids in generated_ids_iter:\n"
        "                if cancel_flag[0]:\n"
        "                    llm_responses = llm_responses_holder[0]\n"
        "                    if llm_responses is not None and hasattr(llm_responses, 'cancel'):\n"
        "                        llm_responses.cancel()\n"
        "                    break\n"
        "                generated_ids = generated_ids.tolist()\n"
        "                if len(generated_ids) == 0:\n"
        "                    break\n"
        "                semantic_token_ids_arr.extend(generated_ids)\n"
        "        except Exception as exc:\n"
        "            if not cancel_flag[0]:\n"
        "                llm_error_holder[0] = str(exc)\n"
        "        finally:\n"
        "            llm_is_done_flag[0] = True\n"
    )
    text, did = _replace_once(text, old_llm_thread, new_llm_thread)
    changed = changed or did

    text, did = _replace_once(
        text,
        "            generated_ids_iter = self.forward_llm(input_ids)\n"
        "\n"
        "            token2wav_request_id = request_id or str(uuid4())\n",
        "            llm_responses_holder = [None]\n"
        "            generated_ids_iter = self.forward_llm(input_ids, llm_responses_holder)\n"
        "\n"
        "            token2wav_request_id = request_id or str(uuid4())\n",
    )
    changed = changed or did

    text, did = _replace_once(
        text,
        "                semantic_token_ids_arr = []\n"
        "                llm_is_done_flag = [False]\n"
        "\n"
        "                llm_thread = threading.Thread(\n"
        "                    target=self._llm_gen_thread,\n"
        "                    args=(generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag)\n"
        "                )\n",
        "                semantic_token_ids_arr = []\n"
        "                llm_is_done_flag = [False]\n"
        "                llm_error_holder = [None]\n"
        "                cancel_flag = [False]\n"
        "                request_cancelled = False\n"
        "\n"
        "                llm_thread = threading.Thread(\n"
        "                    target=self._llm_gen_thread,\n"
        "                    args=(generated_ids_iter, semantic_token_ids_arr, llm_is_done_flag, llm_error_holder, cancel_flag, llm_responses_holder)\n"
        "                )\n",
    )
    changed = changed or did

    text, did = _replace_once(
        text,
        "                while True:\n"
        "                    pending_num = len(semantic_token_ids_arr) - token_offset\n"
        "\n"
        "                    if llm_is_done_flag[0]:\n"
        "                        break\n",
        "                while True:\n"
        "                    if response_sender.is_cancelled():\n"
        "                        request_cancelled = True\n"
        "                        cancel_flag[0] = True\n"
        "                        llm_responses = llm_responses_holder[0]\n"
        "                        if llm_responses is not None and hasattr(llm_responses, 'cancel'):\n"
        "                            llm_responses.cancel()\n"
        "                        self.logger.log_info('request cancelled by client, stopping generation')\n"
        "                        break\n"
        "\n"
        "                    pending_num = len(semantic_token_ids_arr) - token_offset\n"
        "\n"
        "                    if llm_error_holder[0] is not None:\n"
        "                        raise pb_utils.TritonModelException(f\"LLM stream thread failed: {llm_error_holder[0]}\")\n"
        "\n"
        "                    if llm_is_done_flag[0]:\n"
        "                        break\n",
    )
    changed = changed or did

    text, did = _replace_once(
        text,
        "                        audio_tensor = pb_utils.Tensor(\"waveform\", sub_tts_speech.numpy().flatten().astype(\"float32\"))\n"
        "                        inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])\n"
        "                        response_sender.send(inference_response)\n",
        "                        if response_sender.is_cancelled():\n"
        "                            request_cancelled = True\n"
        "                            cancel_flag[0] = True\n"
        "                            llm_responses = llm_responses_holder[0]\n"
        "                            if llm_responses is not None and hasattr(llm_responses, 'cancel'):\n"
        "                                llm_responses.cancel()\n"
        "                            self.logger.log_info('request cancelled before sending chunk')\n"
        "                            break\n"
        "\n"
        "                        audio_tensor = pb_utils.Tensor(\"waveform\", sub_tts_speech.numpy().flatten().astype(\"float32\"))\n"
        "                        inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])\n"
        "                        response_sender.send(inference_response)\n",
    )
    changed = changed or did

    text, did = _replace_once(
        text,
        "                this_tts_speech_token = torch.tensor(semantic_token_ids_arr).unsqueeze(dim=0).to(torch.int32).to(self.device)\n"
        "                sub_tts_speech = self.forward_token2wav(this_tts_speech_token, token2wav_request_id, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, token_offset, True)\n"
        "                audio_tensor = pb_utils.Tensor(\"waveform\", sub_tts_speech.numpy().flatten().astype(\"float32\"))\n"
        "                inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])\n"
        "                response_sender.send(inference_response)\n"
        "\n"
        "                llm_thread.join()\n"
        "                response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)\n"
        "                self.logger.log_info(\"send tritonserver_response_complete_final to end\")\n",
        "                if not request_cancelled:\n"
        "                    this_tts_speech_token = torch.tensor(semantic_token_ids_arr).unsqueeze(dim=0).to(torch.int32).to(self.device)\n"
        "                    sub_tts_speech = self.forward_token2wav(this_tts_speech_token, token2wav_request_id, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, token_offset, True)\n"
        "                    audio_tensor = pb_utils.Tensor(\"waveform\", sub_tts_speech.numpy().flatten().astype(\"float32\"))\n"
        "                    inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])\n"
        "                    response_sender.send(inference_response)\n"
        "\n"
        "                llm_thread.join()\n"
        "                if llm_error_holder[0] is not None and not request_cancelled:\n"
        "                    raise pb_utils.TritonModelException(f\"LLM stream thread failed: {llm_error_holder[0]}\")\n"
        "                response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)\n"
        "                if request_cancelled:\n"
        "                    self.logger.log_info(\"send tritonserver_response_complete_final after cancellation\")\n"
        "                else:\n"
        "                    self.logger.log_info(\"send tritonserver_response_complete_final to end\")\n",
    )
    changed = changed or did

    if changed:
        target_path.write_text(text)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch CosyVoice2 model.py for cancellation-aware request lifecycle.")
    parser.add_argument("--target", required=True, help="Path to cosyvoice2/1/model.py inside container")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    changed = apply_patch(target)
    print("patched" if changed else "already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
