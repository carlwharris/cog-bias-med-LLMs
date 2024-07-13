# cog-bias-med-LLMS
Addressing common clinical biases in medical language models. Released as an [arXiv preprint](https://arxiv.org/abs/2402.08113). Data available via Google Drive link [here](https://drive.google.com/file/d/1GsHYY1xm9JggQALzRoqXJ3MQEHogz7dw/view?usp=sharing).

# Installation

## Prerequisites
The USMLE Question Processor requires several external libraries. These dependencies are listed in `requirements.txt`. To install them, follow these steps:

1. Clone the repository or download the source code.
2. Navigate to the directory containing `requirements.txt`.
3. Run the following command in your terminal or command prompt:

   ```bash
   pip install -r requirements.txt

# Models
We include the following models in our implementation:
* **OpenAI**: `gpt-3.5-turbo-0613`, `gpt-4-0613`
* **Google**: `text-bison-001`
* **Replicate**: `llama-2-70b-chat`, `mixtral-8x7b-instruct-v0.1`
* **HuggingFace**: `pmc-llama-13b`, `meditron-7b`, `meditron-70b`


# Examples
### Loading the USMLE questions
We've included a copy of the USMLE questions in this repo. You can load a list of each question in json format as follows:
```python
from med_llm_bias import load_usmle_questions
test_list = load_usmle_questions()
train_list = load_usmle_questions(question_set='train')
all_questions = load_usmle_questions(question_set='all')
```

### Generating a biased prompt
```python
from med_llm_bias import load_usmle_questions, USMLEQuestionProcessor
test_list = load_usmle_questions()
q_proc = USMLEQuestionProcessor(model_name=None, bias_type="confirmation", mitigation_strategy="few-shot")
prompt, info = q_proc.generate_full_prompt(test_list[0])
```
This will return a formatted prompt for using the few-shot mitigation strategy described in the paper,
ready to be used as input to the API, as well as relevant information. Importantly,
`info['answer_idx']` contains the correct answer index (e.g., `'A', 'B'`, etc.), 
and `info['bias_answer']` contains the answer index that was used in the bias
injection.

For `bias_type`, you can choose from the following options: `None` (no bias), 
`"self_diagnosis"`, `"recency"`, `"confirmation"`, `"frequency"`, `"cultural"`, 
`"status_quo"`, `"false_consensus"`.

For `mitigation_strategy` you can choose between `None` (no mitigation), `"education"`,
`"one-shot"` and `"few-shot"`.

### Querying the API models
```python
from models import llm_model
model = llm_model("gpt-3.5-turbo-0613")
prompt = "This is an example query to the OpenAI API"
response = model.query_model(prompt)
```

# Dataset


# Configuring API Keys

## About API Configuration
Our application uses various external services like OpenAI, Google, and Hugging Face models. To access these services, API keys are required. These keys are stored in `api_config.json`. You must obtain and configure your own API keys. Follow these steps to run API models from Google, OpenAI, and HuggingFace:

1. **Rename `.api_config.json` to `api_config.json`**: A template has been provided in
`.api_config.json`. Rename this file to `api_config.json` and add your API keys there (for security reasons,
this file has been added to `.gitignore` so it won't be added to your git repository).

2. **Obtain API Keys**:
   - **OpenAI**: Create an account at [OpenAI](https://openai.com/). After logging in, access your API keys section and generate a new key.
   - **Google Cloud Services**: Go to the [Google Cloud Console](https://console.cloud.google.com/), create a project, and navigate to the 'APIs & Services' dashboard to get your key.
   - **Hugging Face Models**: Register at [Hugging Face](https://huggingface.co/). Go to your profile settings to find your API keys.
   - **Replicate**: Sign up at [Replicate](https://replicate.com/). Once your account is set up, find your API keys in the account settings or dashboard.

3. **Update the File**:
   - Open `api_config.json` in a text editor.
   - Replace the placeholder keys with your own keys. For example, change `"API_KEY": "sk-0NoVblPZ..."` to `"API_KEY": "your_actual_api_key_here"`.
   - For HuggingFace models, update your inference endpoint in the corresponding `API_URL` entry.
