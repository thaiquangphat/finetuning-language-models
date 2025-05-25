# Get the model name, finetune type and task
def extract_info(filename):
    """
    Extracts model information from a prediction filename.
    
    This function parses filenames in the format 'prediction-{model_name}-{type}-{task}.json'
    to extract the model name, fine-tuning type, and task type.
    
    Args:
        filename (str): The prediction filename to parse
        
    Returns:
        tuple: A tuple containing:
            - model_name (str): Name of the model used for prediction
            - type_ (str): Type of fine-tuning method used
            - task (str): Type of NLP task performed
        Returns (None, None, None) if filename format is invalid
    """
    basename = filename[:-5]
    if not basename.startswith('prediction-'):
        return None, None, None
    parts = basename[len('prediction-'):].split('-')
    if len(parts) < 3:
        return None, None, None
    model_name = '-'.join(parts[:-2])
    type_ = parts[-2]
    task = parts[-1]
    return model_name, type_, task