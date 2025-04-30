from modules.model.llms.t5 import ModelT5ForQuestionAnswering, ModelT5ForTranslation, ModelT5ForTextSentiment
from modules.model.llms.bart import ModelBartForQuestionAnswering, ModelBartForTranslation, ModelBartForTextSentiment
from modules.model.llms.gpt2 import ModelGPT2ForQuestionAnswering, ModelGPT2ForTranslation, ModelGPT2ForTextSentiment

# ============================= LOADING T5 MODEL ============================= #

def load_t5_base(name='t5-base', finetune_type='full', task='question_answering', device='cpu'):
    """
    Load the T5-Base model.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): t5-base model mapped to device.
        tokenizer (AutoTokenizer): t5-base tokenizer mapped to device.
    """

    if task == 'question_answering':
        model, tokenizer = ModelT5ForQuestionAnswering(name=name, finetune_type=finetune_type, device=device)
    elif task == 'translation':
        model, tokenizer = ModelT5ForTranslation(name=name, finetune_type=finetune_type, device=device)
    elif task == 'text_sentiment_analysis':
        model, tokenizer = ModelT5ForTextSentiment(name=name, finetune_type=finetune_type, device=device)
    else:
        raise NotImplementedError(f"Task {task} is not implemented for T5 model.")
    
    return model, tokenizer

# ============================= LOADING BART MODEL ============================= #

def load_bart_base(name='bart-base', finetune_type='full', task='qa', device='cpu'):
    """
    Load the Bart-Base model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (BartForConditionalGeneration): bart-base model mapped to device.
        tokenizer (BartTokenizer): bart-base tokenizer mapped to device.
    """

    if task == 'question_answering':
        model, tokenizer = ModelBartForQuestionAnswering(name=name, finetune_type=finetune_type, device=device)
    elif task == 'translation':
        model, tokenizer = ModelBartForTranslation(name=name, finetune_type=finetune_type, device=device)
    elif task == 'text_sentiment_analysis':
        model, tokenizer = ModelBartForTextSentiment(name=name, finetune_type=finetune_type, device=device)
    else:
        raise NotImplementedError(f"Task {task} is not implemented for Bart model.")
    
    return model, tokenizer

# ============================= LOADING GPT2 MODEL ============================= #

def load_gpt_2(name='gpt2', finetune_type='full', task='qa', device='cpu'):

    """
    Load the GPT-2 model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForCausalLM): gpt2-base model mapped to device.
        tokenizer (AutoTokenizer): gpt2-base tokenizer mapped to device.
    """
    
    if task == 'question_answering':
        model, tokenizer = ModelGPT2ForQuestionAnswering(name=name, finetune_type=finetune_type, device=device)
    elif task == 'translation':
        model, tokenizer = ModelGPT2ForTranslation(name=name, finetune_type=finetune_type, device=device)
    elif task == 'text_sentiment_analysis':
        model, tokenizer = ModelGPT2ForTextSentiment(name=name, finetune_type=finetune_type, device=device)
    else:
        raise NotImplementedError(f"Task {task} is not implemented for GPT-2 model.")
    
    return model, tokenizer
