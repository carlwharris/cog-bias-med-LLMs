from openai import OpenAI
import os
import requests
import re
import google.generativeai as palm
import json
import replicate

openai_models = ['gpt-3.5-turbo-0613', 'gpt-4-0613']
google_models = ['text-bison-001']
replicate_models = ['llama-2-70b-chat', 'mixtral-8x7b-instruct-v0.1']
hf_models = ['pmc-llama-13b', 'medalpaca-13b', 'meditron-7b', \
             'meditron-7b-chat', 'meditron-70b']

with open("api_config.json", "r") as jsonfile:
    api_config = json.load(jsonfile)
class llm_model:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name in openai_models:
            self.api_key = api_config['OpenAI']["API_KEY"]
            self.client = OpenAI(api_key=self.api_key)
            
        if self.model_name in google_models:
            self.api_key = api_config['Google']["API_KEY"]
            palm.configure(api_key=self.api_key)
        
        if self.model_name in replicate_models:
            self.api_key = api_config['Replicate']["API_KEY"]
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
        
        if self.model_name in hf_models:
            self.api_url = api_config[self.model_name]["API_URL"]
            self.api_key = api_config[self.model_name]["API_KEY"]

    def query_model(self, prompt):
        # OPENAI MODELS
        if self.model_name in openai_models:
            response = self._query_openai(prompt)
        
        # GOOGLE MODELS
        if self.model_name == 'text-bison-001':
            response = self._query_google(prompt)
        
        # REPLICATE MODELS
        if self.model_name == 'llama-2-70b-chat':
            response = self._query_replicate(prompt, max_new_tokens=40)
        
        if self.model_name == 'mixtral-8x7b-instruct-v0.1':
            response = self._query_replicate(prompt, max_new_tokens=40)

        # HUGGINGFACE MODELS            
        if self.model_name == 'pmc-llama-13b':
            response = self._query_hf(prompt, max_new_tokens=40)

        if self.model_name == "medalpaca-13b":
            response = self._query_hf(prompt, max_new_tokens=400)
        
        if self.model_name == "meditron-7b-chat":
            response = self._query_hf(prompt, max_new_tokens=40)

        if self.model_name == "meditron-7b":
            response = self._query_hf(prompt, max_new_tokens=40)

        if self.model_name == 'meditron-70b':
            response = self._query_hf(prompt, max_new_tokens=40)
        
        return response
    
    def _query_openai(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "system", "content": prompt}])
        response = completion.choices[0].message.content
        return response

    def _query_google(self, prompt):
        completion = palm.generate_text(
            model=f'models/{self.model_name}',
            prompt=prompt,
        )
        response = completion.result

        if response is None:
            response = "NR"
            return response
    
    def _query_replicate(self, prompt, max_new_tokens=40):
        if self.model_name == 'llama-2-70b-chat':
            url = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        elif self.model_name == 'mixtral-8x7b-instruct-v0.1':
            url = "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10"
        
        output = replicate.run(url, input={"prompt": prompt, "max_new_tokens": max_new_tokens})
        response = ''.join(output)
        
        # If response starts with a space, remove it
        if response[0] == ' ':
            response = response[1:]
        return response

    def _query_hf(self, prompt, max_new_tokens=400):
        def query(payload):
            response = requests.post(self.api_url, headers=headers, json=payload)
            return response.json()
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        output = query({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
            }
        })

        return output[0]['generated_text']
