import re


f_name = "results/gpt-4-0613/bias_output_gpt-4-0613.txt"
with open(f_name, "r", encoding='utf8', errors='ignore') as f:
    lines = f.readlines()

with open("data_clean/questions/US/test.jsonl", encoding="utf8") as f:
    test_sentences = f.readlines()

step_info = []
for sentence in test_sentences:
    meta_info = re.search(r'"meta_info":\s*"([^"]+)"', sentence)
    meta_info = meta_info.group(1) if meta_info else None
    step_info.append(meta_info)

step_restriction = 'step2&3' # None, 'step1', 'step2&3'

itr = 0
total = 0
q_num = 0
correct = 0
refused_to_answer = 0
for line in lines:
    line = line.replace("\n", "")

    if "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" in line:
        q_num += 1
    
    if step_restriction is not None:
        if step_info[q_num-1] != step_restriction:
            continue

    if "NORESPONSE" in line: 
        refused_to_answer += 1

    if line in ["True", "False"]:
        if line == "True": correct += 1

        total += 1

    if "IS_CORRECT" in line:
        if "True" in line:
            correct += 1
        if "NR" in line:
            refused_to_answer += 1
        
        total += 1

    itr += 1

print("naive accuracy (where non-response = incorrect)", correct/total)
print("% refused to answer", refused_to_answer/total)
print("accuracy (corrected for non-response)", correct/(total - refused_to_answer))
print(correct,total)