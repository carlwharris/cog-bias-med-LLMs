f_name = "bias_output_text-bison-001.txt"
with open(f_name, "r", encoding='utf8', errors='ignore') as f:
    lines = f.readlines()

itr = 0
total = 0
correct = 0
refused_to_answer = 0
for line in lines:
    line = line.replace("\n", "")
    if "NORESPONSE" in line: refused_to_answer += 1
    if line in ["True", "False"]:
        if line == "True": correct += 1
        total += 1
    itr += 1

print("naive accuracy (where non-response = incorrect)", correct/total)
print("% refused to answer", refused_to_answer/total)
print("accuracy (corrected for non-response)", correct/(total - refused_to_answer))
print(correct,total)