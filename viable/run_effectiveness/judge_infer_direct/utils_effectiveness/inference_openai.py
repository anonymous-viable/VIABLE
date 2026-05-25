import base64
import os
from openai import OpenAI


class InferenceOpenAI:
    def __init__(self, model_name="gpt-5.4", api_key=None, base_url=None):
        self.model_name = model_name

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set")

        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_media_type(self, image_path):
        ext = image_path.lower().split('.')[-1]
        media_types = {
            'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'png': 'image/png', 'gif': 'image/gif', 'webp': 'image/webp'
        }
        return media_types.get(ext, 'image/jpeg')

    def processing_item(self, img_paths, question, max_new_tokens=256, temperature=0.7):
        content = []
        for img_path in img_paths:
            img_base64 = self._encode_image_to_base64(img_path)
            media_type = self._get_media_type(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{img_base64}"}
            })
        content.append({"type": "text", "text": question})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        return response.choices[0].message.content.strip()
