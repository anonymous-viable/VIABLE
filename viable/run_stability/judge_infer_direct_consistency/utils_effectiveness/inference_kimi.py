from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch


class InferenceKimi():
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_local_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_local_path, trust_remote_code=True
        )
        self.tok = self.processor.tokenizer

    @torch.inference_mode()
    def processing_item(self, img_paths, question, max_new_tokens=256):
        images = [Image.open(p) for p in img_paths]

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, do_sample=True, temperature=0.7, max_new_tokens=max_new_tokens,
            eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
