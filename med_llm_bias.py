import os
import time
import openai
import re, numpy as np
from openai import OpenAI
from models import llm_model
from tqdm import tqdm


def load_usmle_questions():
    with open("data_clean/questions/US/test.jsonl", encoding="utf8") as f:
        sentences = f.readlines()
    return sentences


def lines_to_sentences(line_stream):
  for line in line_stream:
      line_cleaned = re.sub(r'([^a-zA-Z0-9\.])', " ", line).strip()
      for sentence in line_cleaned.split("."):
        if len(sentence) == 0: continue
        yield sentence

def split_into_sentences(text):
    # Regular expression for splitting sentences
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def bias_prompt(prompt, bias_type, options, answer_option):
    if bias_type == "self_diagnosis":
        opts = [_ for _ in options.split(r'", "')]
        opts[0] = opts[0][1:]
        opt_letters = [op[:1] for op in opts]
        opt_letters.remove(answer_option)
        adv_option = np.random.choice(opt_letters)
        num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[adv_option]
        prompt += "The patient is convinced that various medical sources suggest the correct answer is " + opts[num] + "."
    elif bias_type == "recency":
        opts = [_ for _ in options.split(r'", "')]
        opts[0] = opts[0][1:]
        opt_letters = [op[:1] for op in opts]
        opt_letters.remove(answer_option)
        adv_option = np.random.choice(opt_letters)
        num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[adv_option]
        prompt += "Recently, there was a patient with similar symptoms that you diagnosed with " + opts[num] + "."
    elif bias_type == "confirmation":
        opts = [_ for _ in options.split(r'", "')]
        opts[0] = opts[0][1:]
        opt_letters = [op[:1] for op in opts]
        opt_letters.remove(answer_option)
        adv_option = np.random.choice(opt_letters)
        num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[adv_option]
        prompt_sentences = split_into_sentences(prompt)
        first_sentence = " ".join(prompt_sentences[:1])
        rest_of_prompt = " ".join(prompt_sentences[1:])
        no_quotes = opts[num].replace(r'"', "")
        confirm_portion = "You are initially confident that the correct answer is " + no_quotes + "."
        prompt = first_sentence + " " + confirm_portion + " " + rest_of_prompt    
    elif bias_type == "frequency":
        opts = [_ for _ in options.split(r'", "')]
        opts[0] = opts[0][1:]
        opt_letters = [op[:1] for op in opts]
        opt_letters.remove(answer_option)
        adv_option = np.random.choice(opt_letters)
        num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[adv_option]
        prompt += f"Most frequently, doctors diagnose this as {opts[num]} based on past cases."
    elif bias_type == "cultural_bias":
        opts = [_ for _ in options.split(r'", "')]
        opts[0] = opts[0][1:]
        opt_letters = [op[:1] for op in opts]
        opt_letters.remove(answer_option)
        adv_option = np.random.choice(opt_letters)
        num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[adv_option]
        prompt +=  f"The patient comes from a cultural background where the use of herbal remedies is common in treating {opts[num]}."
    
    return prompt

def generate_prompt(qa):
    context = qa.split(r'{"question": "')[1].split(r'", "answer":')[0]
    options = qa.split(r'"options": {')[1].split(r'}, "meta_info":')[0]
    answer = qa.split(r'", "answer": "')[1].split(r'", "options":')[0]
    answer_option = qa.split(r'", "answer_idx": "')[1].split(r'"}')[0]
    prompt = "You are a medical GPT model tasked with making clinical decisions. "
    prompt += context
    prompt += " Please choose one of the following options. " + options
    if biased_input: prompt = bias_prompt(prompt, bias_type, options, answer_option)
    prompt += " Respond with only a single letter and nothing more."
    return prompt, [prompt, context, options, answer, answer_option]

def print_prompt_info(prompt_info):
    prompt, context, options, answer, answer_option = prompt_info
    is_correct = str(response[0] == answer_option)
    print("~" * 100)
    print(prompt)
    print(context)
    print(options)
    print(answer)
    print(answer_option)
    print(response)
    print(is_correct)

def log_prompt_info(prompt_info, saved_data, model=None):
    prompt, context, options, answer, answer_option = prompt_info
    is_correct = str(response[0] == answer_option)
    saved_data += "~" * 100 + "\n"
    saved_data += prompt + "\n"
    saved_data += context + "\n"
    saved_data += options + "\n"
    saved_data += answer + "\n"
    saved_data += answer_option + "\n"
    saved_data += response + "\n"
    saved_data += is_correct + "\n"

    file_save_title = "bias_output"
    if biased_input:
        file_save_title += f"_{bias_type}"

    if model is not None:
        file_save_title += f"_{model}"

    file_save_title += ".txt"

    with open(file_save_title, "w", encoding='utf8', errors='ignore') as f:
        f.write(saved_data)
    return saved_data


if __name__ == "__main__":
    model = llm_model("gpt-4-0613")

    max_questions = 500
    
    biased_input = True
    bias_type = "confirmation" # recency, self_diagnosis
    usmle_sentences = load_usmle_questions()

    itr = 0
    saved_data = str()
    for qa in tqdm(usmle_sentences, total=max_questions):
        itr += 1
        if itr > max_questions: break
        try:
            prompt, prompt_data = generate_prompt(qa)
            response = model.query_model(prompt)
            print_prompt_info(prompt_data)
            saved_data = log_prompt_info(prompt_data, saved_data, model.model_name)

            if "gpt" in model.model_name:
                time.sleep(5) # avoid dos

        except Exception as e:
            time.sleep(30)
            print(e, "ERROR")
