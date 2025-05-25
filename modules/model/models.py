from modules.model.llms.t5 import ModelT5ForQuestionAnswering, ModelT5ForTranslation, ModelT5ForTextSentiment
from modules.model.llms.bart import ModelBartForQuestionAnswering, ModelBartForTranslation, ModelBartForTextSentiment
from modules.model.llms.gpt2 import ModelGPT2ForQuestionAnswering, ModelGPT2ForTranslation, ModelGPT2ForTextSentiment

# ============================= LOADING T5 MODEL ============================= #

def load_t5_base(name='t5-base', finetune_type='full', task='question_answering', device='cpu'):
    """
    Loads and configures a T5-Base model for a specific NLP task.
    
    This function handles loading the T5 model with different fine-tuning approaches:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - Adapters
    
    Args:
        name (str): Name of the pretrained model (default: 't5-base')
        finetune_type (str): Fine-tuning technique to use ('full', 'lora', or 'adapters')
        task (str): Target NLP task ('question_answering', 'english_to_german_translation', or 'text_sentiment_analysis')
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured T5 model for the specified task
            - tokenizer: The corresponding T5 tokenizer
            
    Raises:
        NotImplementedError: If the specified task is not implemented for T5
    """

    if task == 'question_answering':
        model, tokenizer = ModelT5ForQuestionAnswering(name=name, finetune_type=finetune_type, device=device)
    elif task == 'english_to_german_translation':
        model, tokenizer = ModelT5ForTranslation(name=name, finetune_type=finetune_type, device=device)
    elif task == 'text_sentiment_analysis':
        model, tokenizer = ModelT5ForTextSentiment(name=name, finetune_type=finetune_type, device=device)
    else:
        raise NotImplementedError(f"Task {task} is not implemented for T5 model.")
    
    return model, tokenizer

# ============================= LOADING BART MODEL ============================= #

def load_bart_base(name='bart-base', finetune_type='full', task='qa', device='cpu'):
    """
    Loads and configures a BART-Base model for a specific NLP task.
    
    This function handles loading the BART model with different fine-tuning approaches:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - Adapters
    
    Args:
        name (str): Name of the pretrained model (default: 'bart-base')
        finetune_type (str): Fine-tuning technique to use ('full', 'lora', or 'adapters')
        task (str): Target NLP task ('question_answering', 'english_to_german_translation', or 'text_sentiment_analysis')
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured BART model for the specified task
            - tokenizer: The corresponding BART tokenizer
            
    Raises:
        NotImplementedError: If the specified task is not implemented for BART
    """

    if task == 'question_answering':
        model, tokenizer = ModelBartForQuestionAnswering(name=name, finetune_type=finetune_type, device=device)
    elif task == 'english_to_german_translation':
        model, tokenizer = ModelBartForTranslation(name=name, finetune_type=finetune_type, device=device)
    elif task == 'text_sentiment_analysis':
        model, tokenizer = ModelBartForTextSentiment(name=name, finetune_type=finetune_type, device=device)
    else:
        raise NotImplementedError(f"Task {task} is not implemented for Bart model.")
    
    return model, tokenizer

# ============================= LOADING GPT2 MODEL ============================= #

def load_gpt_2(name='gpt2', finetune_type='full', task='qa', device='cpu'):
    """
    Loads and configures a GPT-2 model for a specific NLP task.
    
    This function handles loading the GPT-2 model with different fine-tuning approaches:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - Adapters
    
    Args:
        name (str): Name of the pretrained model (default: 'gpt2')
        finetune_type (str): Fine-tuning technique to use ('full', 'lora', or 'adapters')
        task (str): Target NLP task ('question_answering', 'english_to_german_translation', or 'text_sentiment_analysis')
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured GPT-2 model for the specified task
            - tokenizer: The corresponding GPT-2 tokenizer
            
    Raises:
        NotImplementedError: If the specified task is not implemented for GPT-2
    """
    
    if task == 'question_answering':
        model, tokenizer = ModelGPT2ForQuestionAnswering(name=name, finetune_type=finetune_type, device=device)
    elif task == 'english_to_german_translation':
        model, tokenizer = ModelGPT2ForTranslation(name=name, finetune_type=finetune_type, device=device)
    elif task == 'text_sentiment_analysis':
        model, tokenizer = ModelGPT2ForTextSentiment(name=name, finetune_type=finetune_type, device=device)
    else:
        raise NotImplementedError(f"Task {task} is not implemented for GPT-2 model.")
    
    return model, tokenizer
