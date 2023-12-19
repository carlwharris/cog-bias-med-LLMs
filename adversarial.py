#Adversarial attacks to check to robustness of the model

import os
import json, re, numpy as np
import openai

def create_corrupted_prompt(prompt):
    # List of models to attack
    models = ["gpt", "LLAMA", "ClinicalBERT", "BioMegatron", "mistralai"]
    
    # Dictionary to store corrupted prompts for each model
    corrupted_prompts = {}
    
    # Create a corrupted prompt for each model
    for model in models:
        corrupted_prompt = prompt + " " + model
        corrupted_prompts[model] = corrupted_prompt
    
    return corrupted_prompts
