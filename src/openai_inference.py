"""OpenAI Inference Class."""
import time
import requests
from utils import encode_image_to_base64, load_config
from openai import AzureOpenAI  # openai>=1.0.0
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import json
from openai import OpenAI
import google.generativeai as genai
import mimetypes
from anthropic import Anthropic

class OpenAIInference:
    """OpenAI Inference Class."""
    def __init__(self, config_path):
        """Initializes a OpenAIInference instance with the given configuration file.

        Args:
            config_path (str): The file path to the YAML configuration file.
        """
        self.clients = load_config(config_path)['clients']

    def gemini_inference(self, system_prompt, prompt, engine, image_paths=None):
        genai.configure(
                api_key=self.clients[engine]['api_key'],
                transport="rest",
                client_options={"api_endpoint": self.clients[engine]['endpoint']}
            )

        model = genai.GenerativeModel(self.clients[engine]['name'])

        contents = []
        prompt = system_prompt + "\n\n" + prompt if system_prompt else prompt
        if prompt:
            contents.append({"text": prompt})

        if image_paths:
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for path in image_paths:
                try:
                    mime_type, _ = mimetypes.guess_type(path)
                    if mime_type is None:
                        mime_type = "image/png"
                    encoded_data = encode_image_to_base64(path)
                    contents.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_data
                        }
                    })
                except Exception as e:
                    print(f"Failed to read image: {path}, error: {e}")
                    continue

        # system_prompt is only supported by some models; in some cases it needs to be integrated through model config or prompt
        response = model.generate_content(
            contents=contents,
        )

        return response.text

    def build_image_part(self, path):
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            raise ValueError(f"Cannot determine MIME type for file: {path}")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": encode_image_to_base64(path),
            },
        }

    def claude_inference(self, system_prompt, prompt, engine, image_paths=None, max_tokens=3000):
        client = Anthropic(
            base_url=self.clients[engine]['endpoint'],
            api_key=self.clients[engine]['api_key'],
        )

        # Construct content
        content = []

        # Add image parts
        for path in image_paths:
            content.append(self.build_image_part(path))

        # Add text part
        content.append({
            "type": "text",
            "text": prompt
        })

        # Call Claude
        response = client.messages.create(
            model=self.clients[engine]['name'],
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        return response.content[0].text
    def run(self, system_prompt, prompt, image_paths=None, temperature=0.0, max_tokens=3000, engine="gpt-4o", max_attempts=10):
        """Runs the OpenAI inference.

        Args:
            prompt (str): The prompt to be used for the inference.
            image_paths (List[str]): The list of image paths to be used for the inference.
            temperature (float): The temperature to be used for the inference.
            max_tokens (int): The maximum number of tokens to be used for the inference.

        Returns:
            str: The response from the OpenAI inference.
        """
        global api_total_cost
        if engine == 'R1':
            client = ChatCompletionsClient(
                endpoint=self.clients[engine]['endpoint'],
                credential=AzureKeyCredential(self.clients[engine]['api_key']),
            )
        elif engine == 'Qwen2.5-VL-72B-Instruct' or engine == 'QVQ-72B-Preview' or engine == 'Qwen2-VL-72B-Instruct':
            client = OpenAI(
                api_key=self.clients[engine]['api_key'],
                base_url=self.clients[engine]['endpoint'],
            )
        elif engine != 'Gemini-2.5-Flash-Preview' and engine != 'Claude-3-7-sonnet':
            client = AzureOpenAI(
                azure_endpoint=self.clients[engine]['endpoint'],
                api_key=self.clients[engine]['api_key'],
                api_version=self.clients[engine]['api_version']
            )
        messages = [{"role": "system", "content": f"{system_prompt}"}]
        # Build user message
        if image_paths:
            if isinstance(image_paths, str):
                image_paths = [image_paths]  # convert single path to list
            content = [{"type": "text", "text": prompt}]
            for path in image_paths:
                image_b64 = encode_image_to_base64(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        flag = 0
        while flag == 0 and max_attempts > 0:
            max_attempts -= 1
            try:
                # Gemini multi-image input branch
                if engine == 'Gemini-2.5-Flash-Preview':
                    result = self.gemini_inference(system_prompt, prompt, engine, image_paths)
                    return result
                elif engine == 'Claude-3-7-sonnet':
                    result = self.claude_inference(system_prompt, prompt, engine, image_paths)
                    return result
                if engine == 'o1' or engine == 'o3-mini' or engine == 'o3'  or engine == 'o4-mini':
                    response = client.chat.completions.create(
                        model=self.clients[engine]['name'],
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        frequency_penalty=0,
                    )
                elif engine == 'Qwen2.5-VL-32B-Instruct':
                    payload = {
                        "model": self.clients[engine]['name'],
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    response = requests.post(
                        url=f"{self.clients[engine]['endpoint']}/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=payload
                    )
                    result = response.json()['choices'][0]['message']['content']
                    flag = 1
                    continue
                elif engine == 'Qwen2.5-VL-72B-Instruct' or engine == 'QVQ-72B-Preview' or engine == 'QWQ-32B-Preview':
                    response = client.chat.completions.create(
                        model=self.clients[engine]['name'],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                elif  engine == 'Qwen2-VL-72B-Instruct':
                    response = client.chat.completions.create(
                        model=self.clients[engine]['name'],
                        messages=messages,
                        temperature=temperature
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.clients[engine]['name'],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        frequency_penalty=0,
                    )
                
                result = response.choices[0].message.content
                flag = 1
            except Exception as e:
                print("openai error:", e)
                result = "openai error, retry"
                time.sleep(2)
        return result


if __name__ == "__main__":
    """Main function to run the OpenAI inference."""
    # Example usage
    config_path = '../config/clients.yaml'
    openai_inference = OpenAIInference(config_path)
    system_prompt = "You are a helpful assistant that can describe the images."
    engine = 'o4-mini'
    prompt = "describe the images"
    image_paths = []
    response = openai_inference.run(system_prompt, prompt, image_paths=image_paths, engine=engine)
    print(response)
