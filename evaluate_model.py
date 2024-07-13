import re


f_name = "ae_bias_output_confirmation_llama-2-70b-chat.txt"
f_name = "results/pmc-llama-13b/bias_output_pmc-llama-13b.txt"
# f_name = "bias_output_mixtral-8x7b-instruct-v0.1.txt"
f_name = "ae_bias_output_few-shot_llama-2-70b-chat.txt"
# f_name = "bias_output_confirmation-mitigated_pmc-llama-13b.txt"

with open(f_name, "r", encoding='utf8', errors='ignore') as f:
    lines = f.readlines()

with open("data_clean/questions/US/test.jsonl", encoding="utf8") as f:
    test_sentences = f.readlines()

step_info = []
for sentence in test_sentences:
    meta_info = re.search(r'"meta_info":\s*"([^"]+)"', sentence)
    meta_info = meta_info.group(1) if meta_info else None
    step_info.append(meta_info)

step_restriction = None #'step2&3' # None, 'step1', 'step2&3'

itr = 0
total = 0
q_num = 0
correct = 0
refused_to_answer = 0

max_q = 5000
skip_next = False
for line in lines:
    line = line.replace("\n", "")

    if "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" in line:
        q_num += 1
        start_q = True

    if q_num > max_q:
        break
    
    if step_restriction is not None:
        if step_info[q_num-1] != step_restriction:
            continue

    # if "NORESPONSE" in line: 
    #     refused_to_answer += 1
        
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

    
    itr += 1

import warnings
if step_restriction is None and total != 1273:
    # Throw a warning using warnings
    warnings.warn("total != 1273")


print("naive accuracy (where non-response = incorrect)", correct/total)
print("% refused to answer", refused_to_answer/total)
print("accuracy (corrected for non-response)", correct/(total - refused_to_answer))
print(correct,total)