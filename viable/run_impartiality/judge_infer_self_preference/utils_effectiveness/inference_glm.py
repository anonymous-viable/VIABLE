from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch

class InferenceGLM():
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path
        self.processor = AutoProcessor.from_pretrained(self.model_local_path,trust_remote_code=True)
        self.model = Glm4vForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=self.model_local_path,torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa",trust_remote_code=True).eval()
        self.tok=self.processor.tokenizer

    @torch.inference_mode()
    def processing_item(self, img_paths, question, max_new_tokens=256):
        # Prepare content with images and text
        content = []
        for p in img_paths:
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": question+"\nYou are assisting a blind user. Keep the answer under 256 tokens."})
        messages = [{"role": "user", "content": content}]

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        # Remove token_type_ids if present (required for GLM models)
        inputs.pop("token_type_ids", None)

        # Prepare EOS token IDs
        tok = self.processor.tokenizer
        eos_ids = list(self.model.generation_config.eos_token_id) if isinstance(self.model.generation_config.eos_token_id, list) else [self.model.generation_config.eos_token_id]

        # Add custom EOS token for box ending
        end_of_box_id = tok.convert_tokens_to_ids("<|end_of_box|>")
        if end_of_box_id is not None and end_of_box_id not in eos_ids:
            eos_ids.append(end_of_box_id)

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tok.pad_token_id or tok.eos_token_id
        )

        # Decode response
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Manually truncate at first occurrence of <|end_of_box|>
        if "<|end_of_box|>" in output_text:
            output_text = output_text.split("<|end_of_box|>")[0]

        return output_text
