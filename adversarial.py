#Adversarial attacks to check to robustness of the model

#All the different bias are adversarial attacks on the model
def create_adversarial_prompts(prompt):
    # List of adversarial attacks
    attacks = ["self_diagnosis", "recency", "confirmation", "frequency", "cultural_bias", "blind_spot", "status_quo", "false_consensus"]
    #confirmation bias was really bad for the model
    # Dictionary to store adversarial prompts for each attack

    adversarial_prompts = {
        "self_diagnosis": "I think I have",
        "recency": "I recently",
        "confirmation": "I am sure",
        "frequency": "I often",
        "cultural_bias": "I am",
        "blind_spot": "I am not",
        "status_quo": "I am",
        "false_consensus": "I think"
    }

    # Create an adversarial prompt for each attack
    for attack in attacks:
        adversarial_prompt = prompt + " " + attack
        adversarial_prompts[attack] = adversarial_prompt
    
    return adversarial_prompt


# Import necessary libraries


import matplotlib.pyplot as plt

# Define the attacks and their success rates
attacks = ["self_diagnosis", "recency", "confirmation", "frequency", "cultural_bias", "blind_spot", "status_quo", "false_consensus"]
success_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Create a circle plot of the attacks and their success rates
fig, ax = plt.subplots()
ax.pie(success_rates, labels=attacks, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
plt.title('Success Rates of Adversarial Attacks on GPT')
plt.show()



