from openai import OpenAI
import os

class llm_model:
    def __init__(self, model_name):
        self.model_name = model_name

        if "gpt" in self.model_name:
            self.api_key = os.environ.get('OPENAI_API_KEY')
            self.client = OpenAI(api_key=self.api_key)

        if 'LLAMA' in self.model_name:
            import transformers
            import torch

        if self.model_name == 'PMC_LLAMA_7B':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check for GPU

            self.model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B').to(self.device, dtype=torch.float16)
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')

        if self.model_name == 'PMC_LLaMA_13B':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check for GPU

            self.model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B').to(self.device, dtype=torch.float16)
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
            

    def query_model(self, prompt):
        if "gpt" in self.model_name:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "system", "content": prompt}])
            response = completion.choices[0].message.content
        
        if 'LLAMA' in self.model_name:
            batch = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device) 

            with torch.no_grad():
                response = self.model.generate(inputs=batch["input_ids"], max_new_tokens=400, do_sample=True, top_k=50)
                response = self.tokenizer.decode(response[0])
            
        response = response.replace(r'"', "")
        
        return response
    
