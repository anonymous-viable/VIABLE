from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch

class InferenceMiniCPM():
    def __init__(self, model_local_path):
        self.model_local_path = model_local_path
        self.model = AutoModel.from_pretrained(
            self.model_local_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_local_path,
            trust_remote_code=True
        )

    @torch.inference_mode()
    def processing_item(self, img_paths, question, max_new_tokens=256):
        # Load images
        images = []
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            images.append(img)

        # Prepare messages in the format: [image1, image2, ..., question]
        content = images + [question]
        msgs = [{'role': 'user', 'content': content}]

        # Generate response using chat API
        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens
        )

        return answer
