import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, # For T5
    BartTokenizer, BartForConditionalGeneration, BartConfig, # For BART
    ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig # For ProphetNet
)
from adapters import AutoAdapterModel

# ============================= LOADING MODEL ============================= #

def load_t5_base(name='t5-base', finetune_type='full', device='cpu'):
    """
    Load the T5-Base model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): t5-base model mapped to device.
        tokenizer (AutoTokenizer): t5-base tokenizer mapped to device.
    """

    if finetune_type != 'adapters':
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        config = T5Config.from_pretrained(name)
        model = AutoAdapterModel.from_pretrained(name, config=config)
    
    tokenizer = AutoTokenizer.from_pretrained(name)

    model.to(device)

    return model, tokenizer

def load_bart_base(name='bart-base', finetune_type='full', device='cpu'):
    """
    Load the Bart-Base model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (BartForConditionalGeneration): bart-base model mapped to device.
        tokenizer (BartTokenizer): bart-base tokenizer mapped to device.
    """

    if finetune_type != 'adapters':
        model = BartForConditionalGeneration.from_pretrained(f'facebook/{name}')
    else:
        config = BartConfig.from_pretrained(f'facebook/{name}')
        model = AutoAdapterModel.from_pretrained(f'facebook/{name}', config=config)

    tokenizer = BartTokenizer.from_pretrained(f'facebook/{name}')

    model.to(device)

    return model, tokenizer

def load_prophetnet_large(name='prophetnet-large-uncased', finetune_type='full', device='cpu'):
    """
    Load the ProphetNet-Large model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (ProphetNetForConditionalGeneration): prophetnet-base model mapped to device.
        tokenizer (ProphetNetTokenizer): prophetnet-base tokenizer mapped to device.
    """
    if finetune_type != 'adapters':
        model = ProphetNetForConditionalGeneration.from_pretrained(f'microsoft/{name}')
    else:
        config = ProphetNetConfig.from_pretrained(f'microsoft/{name}')
        model = AutoAdapterModel.from_pretrained(f'microsoft/{name}', config=config)

    tokenizer = ProphetNetTokenizer.from_pretrained(f'microsoft/{name}')

    model.to(device)

    return model, tokenizer