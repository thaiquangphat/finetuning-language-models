import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, # For T5
    BartTokenizer, BartForConditionalGeneration, BartConfig, # For BART
    AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel # For GPT-2
)
from modules.model.custom_model import GPT2ForExtractiveQA
from peft import LoraConfig, get_peft_model, TaskType
from adapters import AutoAdapterModel
from peft import get_peft_model, TaskType

# ============================= LOADING MODEL ============================= #

def load_t5_base(name='t5-base', finetune_type='full', task='qa', device='cpu'):
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

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(name)

    elif finetune_type == 'lora':
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "k", "v", "o"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)

    else: # adapters
        config = T5Config.from_pretrained(name)
        model = AutoAdapterModel.from_pretrained(name, config=config)

        model.add_adapter(task)
        model.set_active_adapters(task)
    
    # Move model to device
    model.to(device)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)

    return model, tokenizer

def load_bart_base(name='bart-base', finetune_type='full', task='qa', device='cpu'):
    """
    Load the Bart-Base model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (BartForConditionalGeneration): bart-base model mapped to device.
        tokenizer (BartTokenizer): bart-base tokenizer mapped to device.
    """

    if 'models' in name:
        model_path = name
    else:
        model_path = f'facebook/{name}'

    if finetune_type == 'full':
        model = BartForConditionalGeneration.from_pretrained(model_path)
    
    elif finetune_type == 'lora':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=32, # 16, 32
            lora_alpha=64, # 64,32
            lora_dropout=0.1, # 0.05, 0.1
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    
    else:
        # Define Adapter Configuration using PEFT (Pfeiffer is one of the popular configurations)
        adapter_config = BartConfig.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=adapter_config)

        # Adding a task-specific adapter
        model.add_adapter(adapter_name=task)
        model.set_active_adapters(task)
    
    # Move model to device
    model.to(device)

    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)

    return model, tokenizer

def load_gpt_2(name='gpt2', finetune_type='full', task='qa', device='cpu'):

    """
    Load the GPT-2 model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForCausalLM): gpt2-base model mapped to device.
        tokenizer (AutoTokenizer): gpt2-base tokenizer mapped to device.
    """
    model_path = name

    if finetune_type == 'full':
        if task == 'question_answering':
            model = GPT2ForExtractiveQA.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
    
    elif finetune_type == 'lora':
        if task == 'question_answering':
            model = GPT2ForExtractiveQA.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    else: # TODO: adapters for GPT2
        adapter_config = GPT2Config.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=adapter_config)

        # Adding a task-specific adapter
        model.add_adapter(adapter_name=task)
        model.set_active_adapters(task)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer