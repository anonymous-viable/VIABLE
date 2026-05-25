from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image

class InferenceYoutu():
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path
        cfg = AutoConfig.from_pretrained(model_local_path, trust_remote_code=True)
        cfg._attn_implementation = "flash_attention_2"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_local_path, config=cfg, torch_dtype="auto",
            device_map="auto", trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)

    @torch.inference_mode()
    def processing_item(self, img_paths, question, max_new_tokens=256):
        content = []
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, do_sample=True, temperature=0.7, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs[0]
