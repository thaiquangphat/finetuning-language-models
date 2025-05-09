# Get the model name, finetune type and task
def extract_info(filename):
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