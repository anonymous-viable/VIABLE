from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

class InferenceQwen():
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_local_path, dtype="auto", attn_implementation="flash_attention_2", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_local_path)
        self.tok = self.processor.tokenizer

    @torch.inference_mode()
    def processing_item(self, img_paths, question, max_new_tokens=512):
        content = []
        for p in img_paths:
            content.append({"type": "image", "image": p})

        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, do_sample=True, temperature=0.7, max_new_tokens=max_new_tokens, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]