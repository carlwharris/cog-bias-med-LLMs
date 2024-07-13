

def parse_results_file(f_name):
    bias_name = None
    mitigation_strategy = None
    model_name = ""

    models = ['gpt-4-0613', 'mixtral-8x7b-instruct-v0.1', 'gpt-3.5-turbo-0613', 'text-bison-001', 'pmc-llama-13b', 'llama-2-70b-chat', 
              'meditron-70b']
    bias_types = ["self_diagnosis", "recency", "confirmation", "frequency", "cultural",  "status_quo", "false_consensus"]
    mitigation_strategies = ["education", "one-shot", "few-shot"]

    for mitigation_strategy in mitigation_strategies:
        if mitigation_strategy in f_name:
            mitigation_strategy = mitigation_strategy
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
