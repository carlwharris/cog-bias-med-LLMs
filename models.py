from openai import OpenAI
import os
import transformers
import torch

class llm_model:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name in ["gpt-3.5-turbo", "gpt-4"]:
            self.api_key = os.environ.get('OPENAI_API_KEY')
            self.client = OpenAI(api_key=self.api_key)

        if self.model_name == 'PMC_LLAMA_7B':
            self.model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')

    def query_model(self, prompt):
        if self.model_name in ["gpt-3.5-turbo", "gpt-4"]:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "system", "content": prompt}])
        
        if self.model_name == 'PMC_LLAMA_7B':
            batch = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            )

            with torch.no_grad():
                response = self.model.generate(inputs=batch["input_ids"], max_length=400, do_sample=True, top_k=50)
                response = self.tokenizer.decode(response[0])
            response = response.replace(r'"', "")
        
        return response
    
