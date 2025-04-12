import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, # For T5
    BartTokenizer, BartForConditionalGeneration, # For BART
    ProphetNetTokenizer, ProphetNetForConditionalGeneration # For ProphetNet
)

# ============================= LOADING MODEL ============================= #

def load_t5_base(device='cpu'):
    """
    Load the T5-Base model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): t5-base model mapped to device.
        tokenizer (AutoTokenizer): t5-base tokenizer mapped to device.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    model.to(device)

    return model, tokenizer

def load_bart_base(device='cpu'):
    """
    Load the Bart-Base model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (BartForConditionalGeneration): bart-base model mapped to device.
        tokenizer (BartTokenizer): bart-base tokenizer mapped to device.
    """

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    model.to(device)

    return model, tokenizer

def load_prophetnet_large(device='cpu'):
    """
    Load the ProphetNet-Large model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (ProphetNetForConditionalGeneration): prophetnet-base model mapped to device.
        tokenizer (ProphetNetTokenizer): prophetnet-base tokenizer mapped to device.
    """

    model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")
    tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")

    model.to(device)

    return model, tokenizer