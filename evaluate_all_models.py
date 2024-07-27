import re
import os
import pandas as pd

model = "gpt-3.5-turbo-0613"
max_q = 5000
step_restriction = None #'step2&3' # None, 'step1', 'step2&3'

# ae_models = ['llama-2-70b-chat', 'pmc-llama-13b']
ae_models = ['pmc-llama-13b']

subdir_path = os.path.join("final_results", model)

# List all files in the directory
f_names = [f for f in os.listdir(subdir_path) if "bias_output" in f]

if model in ae_models:
    f_names = [f for f in f_names if "ae_" in f]
else:
    f_names = [f for f in f_names if "ae_" not in f]

with open("data_clean/questions/US/test.jsonl", encoding="utf8") as f:
    test_sentences = f.readlines()

step_info = []
for sentence in test_sentences:
    meta_info = re.search(r'"meta_info":\s*"([^"]+)"', sentence)
    meta_info = meta_info.group(1) if meta_info else None
    step_info.append(meta_info)

def eval_file(f_name):
    with open(f_name, "r", encoding='utf8', errors='ignore') as f:
        lines = f.readlines()

    q_num = 0
    total = 0
    correct = 0
    refused_to_answer = 0

    skip_next = False
    for line in lines:
        line = line.replace("\n", "")

        if "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" in line:
            q_num += 1

        if q_num > max_q:
            break
        
        if step_restriction is not None:
            if step_info[q_num-1] != step_restriction:
                continue
            
        if "RESPONSE" in line:
            if "NR" in line:
                refused_to_answer += 1
                total += 1
                skip_next = True
                continue

        if "IS_CORRECT" in line:
            if skip_next:
                skip_next = False
                continue
            if "True" in line:
                correct += 1
            if "NR" in line:
                refused_to_answer += 1
        
            total += 1

    import warnings
    if step_restriction is None and total != 1273:
        # Throw a warning using warnings
        warnings.warn("total != 1273")
    
    naive_accuracy = round(correct/total, 3)
    ref_ans = round(refused_to_answer/total, 3)
    corrected_accuracy = round(correct/(total - refused_to_answer), 3)
    return naive_accuracy, ref_ans, corrected_accuracy, correct, total

def parse_file_name(f_name):
    bias_name = "none"
    mitigation_strategy = "none"
    model_name = ""

    models = ['gpt-4-0613', 'mixtral-8x7b-instruct-v0.1', 'gpt-3.5-turbo-0613', 'text-bison-001', 'pmc-llama-13b', 'llama-2-70b-chat', 
              'meditron-70b']
    bias_types = ["self_diagnosis", "recency", "confirmation_v2", "confirmation", "frequency", "cultural",  "status_quo", "false_consensus_v2", "false_consensus"]
    mitigation_strategies = ["mitigated", "education", "one-shot", "few-shot"]

    for strat in mitigation_strategies:
        if strat in f_name:
            if strat == "mitigated":
                mitigation_strategy = "education"
            else:
                mitigation_strategy = strat
            break
    
    for model in models:
        if model in f_name:
            model_name = model
            break

    for bias_type in bias_types:
        if bias_type in f_name:
            bias_name = bias_type
            break

    return model_name, bias_name, mitigation_strategy

def bias_type_sort_key(column):
    return [bias_types.index(bias) if bias in bias_types else len(bias_types) for bias in column]


results_df = pd.DataFrame(columns=['model', 'bias_type', 'mitigation_strategy', 'naive_accuracy', 'ref_ans', 'corrected_accuracy', 'correct', 'total'])
for f_name in f_names:

    model_name, bias_name, mitigation_strategy = parse_file_name(f_name)
    naive_accuracy, ref_ans, corrected_accuracy, correct, total = eval_file(os.path.join(subdir_path, f_name))

    new_row = pd.DataFrame({'model': [model_name], 'bias_type': [bias_name], 'mitigation_strategy': [mitigation_strategy], 'naive_accuracy': [naive_accuracy], 'ref_ans': [ref_ans], 'corrected_accuracy': [corrected_accuracy], 'correct': [correct], 'total': [total]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    bias_types = ["none", "self_diagnosis", "recency", "confirmation", "frequency", "cultural",  "status_quo", "false_consensus"]
    # Sort by bias type
    # results_df = results_df.sort_values(by=['bias_type'], key=lambda x: [bias_types.index(i) for i in x])
    # results_df = results_df.sort_values(by=['mitigation_strategy'])

    # Create a temporary column for sorting bias_type
    results_df['bias_type_order'] = results_df['bias_type'].apply(lambda x: bias_types.index(x) if x in bias_types else len(bias_types))

    # Sort by mitigation_strategy first, and then by the temporary bias_type_order column
    results_df = results_df.sort_values(by=['mitigation_strategy', 'bias_type_order'])

    # Drop the temporary sorting column
    results_df = results_df.drop(columns=['bias_type_order'])

print(results_df)