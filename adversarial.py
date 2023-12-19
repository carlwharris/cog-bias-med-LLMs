#Adversarial attacks to check to robustness of the model

#All the different bias are adversarial attacks on the model
def create_adversarial_prompts(prompt):
    # List of adversarial attacks
    attacks = ["self_diagnosis", "recency", "confirmation", "frequency", "cultural_bias", "blind_spot", "status_quo", "false_consensus"]

    #confirmation bias is really good one


    # Dictionary to store adversarial prompts for each attack
    adversarial_prompts = {}
    
    # Create an adversarial prompt for each attack
    for attack in attacks:
        adversarial_prompt = prompt + " " + attack
        adversarial_prompts[attack] = adversarial_prompt
    
    return adversarial_prompts


