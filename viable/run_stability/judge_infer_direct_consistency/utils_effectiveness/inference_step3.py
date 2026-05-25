from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

class InferenceStep3():
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path

        # Key mapping required for Step3-VL architecture
        key_mapping = {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        }

        self.processor = AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_local_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            key_mapping=key_mapping
        ).eval()

    @torch.inference_mode()
    def processing_item(self, img_paths, question, max_new_tokens=256):
        # Prepare content with images and text
        content = []
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=max_new_tokens
        )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode response
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]
