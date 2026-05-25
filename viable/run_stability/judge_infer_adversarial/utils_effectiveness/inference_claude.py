import base64
import os
from anthropic import Anthropic


class InferenceClaude:
    def __init__(self, model_name="claude-sonnet-4-6", api_key=None, base_url=None):
        self.model_name = model_name

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("API key not provided and ANTHROPIC_API_KEY environment variable not set")

        if base_url is None:
            base_url = os.getenv("ANTHROPIC_BASE_URL")

        if base_url:
            self.client = Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = Anthropic(api_key=api_key)

    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

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
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self._get_media_type(img_path),
                    "data": self._encode_image_to_base64(img_path)
                }
            })
        content.append({"type": "text", "text": question})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}]
        )

        result_text = ""
        for block in response.content:
            if block.type == 'thinking':
                result_text += f"<thinking>\n{block.thinking}\n</thinking>\n\n"
            elif hasattr(block, 'text'):
                result_text += block.text
        return result_text.strip()
