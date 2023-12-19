with open("bias_output_confirmation.txt", "r", encoding='utf8', errors='ignore') as f:
    lines = f.readlines()

itr = 0
total = 0
correct = 0
for line in lines:
    line = line.replace("\n", "")
    if line in ["True", "False"]:
        if line == "True": correct += 1
        total += 1
    itr += 1
print(correct/total)
print(correct,total)