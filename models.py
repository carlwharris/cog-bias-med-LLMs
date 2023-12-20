from openai import OpenAI
import os
import requests

#Add a for loop to iterate throguh all models
#Add a function to load all models
#Add a function to query all models

class llm_model:
    def __init__(self, model_name, use_GPU=True):
        self.model_name = model_name

        if "gpt" in self.model_name.lower():
            self.api_key = os.environ.get('OPENAI_API_KEY')
            self.client = OpenAI(api_key=self.api_key)

        if 'gpt' not in self.model_name.lower():
            import transformers
            import torch

            if torch.cuda.is_available() and use_GPU:
                self.device = 'cuda'
                dtype = torch.float16
            else:
                self.device = 'cpu'
                dtype = torch.float32

        if self.model_name == 'PMC_LLAMA_7B':
            self.model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B').to(self.device, dtype=dtype)
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')

        if self.model_name == 'PMC_LLaMA_13B':
            self.model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B').to(self.device, dtype=dtype)
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
        
        if self.model_name== 'ClinicalBERT':
            self.model=transformers.AutoModelForCausalLM.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(self.device, dtype=dtype)
            self.tokenizer=transformers.AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        if self.model_name=='BioMegatron-11B':
            self.model=transformers.AutoModelForCausalLM.from_pretrained('emilyalsentzer/BioMegatron-11B').to(self.device, dtype=dtype)
            self.tokenizer=transformers.AutoTokenizer.from_pretrained('emilyalsentzer/BioMegatron-11B')
        
        if self.model_name=="Mixtral-8x7B-v0.1": #Outperforms Llama 2 70B on USMLE questions (see paper) 
            self.model=transformers.AutoModelForCausalLM.from_pretrained('mistralai/Mixtral-8x7B-v0.1').to(self.device, dtype=dtype)  #call model from huggingface
            self.tokenizer=transformers.AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1') #call tokenizer from huggingface

        if self.model_name=="Clinical-T5-Large":
            self.model=transformers.AutoModelForCausalLM.from_pretrained('xyla/Clinical-T5-Large').to(self.device, dtype=dtype)
            self.tokenizer=transformers.AutoTokenizer.from_pretrained('xyla/Clinical-T5-Large')
        
        if  self.model_name=="meditron-70b":
            self.model=transformers.AutoModelForCausalLM.from_pretrained('epfl-llm/meditron-70b').to(self.device, dtype=dtype)
            self.tokenizer=transformers.AutoTokenizer.from_pretrained('epfl-llm/meditron-70b')

        if self.model_name=="meditron-70b-finetuned-usmed":
            self.model=transformers.AutoModelForCausalLM.from_pretrained('epfl-llm/meditron-70b-finetuned-usmed').to(self.device, dtype=dtype)
            self.tokenizer=transformers.AutoTokenizer.from_pretrained('epfl-llm/meditron-70b-finetuned-usmed')
            
    def query_model(self, prompt):
        if "gpt" in self.model_name.lower():
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "system", "content": prompt}])
            response = completion.choices[0].message.content
        
        if 'gpt' not in self.model_name.lower():
            import torch 

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
    
