import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, # For T5
    BartTokenizer, BartForConditionalGeneration, BartConfig, # For BART
    ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig # For ProphetNet
)
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig
from adapters import AutoAdapterModel, BartAdapterModel, AdapterConfig
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
            target_modules=['q', 'v'],
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

    # Define adapter creation function
    def make_adapter(in_dim, bottleneck_dim, out_dim):
        adapter_layers = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
        )
        return adapter_layers

    if finetune_type == 'full':
        model = BartForConditionalGeneration.from_pretrained(f'facebook/{name}')
    
    elif finetune_type == 'lora':
        model = AutoModelForSeq2SeqLM.from_pretrained(f'facebook/{name}')
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    
    else:
        # 1. Load BART-base model
        model = BartForConditionalGeneration.from_pretrained(f'facebook/{name}')
        
        # 2. Add adapters manually
        bottleneck_size = 32  # Hyperparameter
        hidden_size = model.config.d_model  # 768 for BART-base

        # 3. Add adapters to encoder layers
        for block_idx in range(len(model.model.encoder.layers)):
            # Encoder: Insert 1st adapter after self-attention output
            orig_layer_1 = model.model.encoder.layers[block_idx].self_attn.out_proj
            adapter_layers_1 = make_adapter(
                in_dim=hidden_size,
                bottleneck_dim=bottleneck_size,
                out_dim=hidden_size
            )
            # Create a new Sequential, but keep the reference to the original layer
            model.model.encoder.layers[block_idx].self_attn.out_proj = nn.Sequential(orig_layer_1, adapter_layers_1)

            # Encoder: Insert 2nd adapter after feed-forward output
            orig_layer_2 = model.model.encoder.layers[block_idx].fc2
            adapter_layers_2 = make_adapter(
                in_dim=hidden_size,
                bottleneck_dim=bottleneck_size,
                out_dim=hidden_size
            )
            model.model.encoder.layers[block_idx].fc2 = nn.Sequential(orig_layer_2, adapter_layers_2)

        # 4. Add adapters to decoder layers
        for block_idx in range(len(model.model.decoder.layers)):
            # Decoder: Insert 1st adapter after self-attention output
            orig_layer_1 = model.model.decoder.layers[block_idx].self_attn.out_proj
            adapter_layers_1 = make_adapter(
                in_dim=hidden_size,
                bottleneck_dim=bottleneck_size,
                out_dim=hidden_size
            )
            model.model.decoder.layers[block_idx].self_attn.out_proj = nn.Sequential(orig_layer_1, adapter_layers_1)

            # Decoder: Insert 2nd adapter after feed-forward output
            orig_layer_2 = model.model.decoder.layers[block_idx].fc2
            adapter_layers_2 = make_adapter(
                in_dim=hidden_size,
                bottleneck_dim=bottleneck_size,
                out_dim=hidden_size
            )
            model.model.decoder.layers[block_idx].fc2 = nn.Sequential(orig_layer_2, adapter_layers_2)
    
    # Move model to device
    model.to(device)

    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained(f'facebook/{name}')

    return model, tokenizer

# TODO: ProphetNet Lora and Adapters
def load_prophetnet_large(name='prophetnet-large-uncased', finetune_type='full', task='qa', device='cpu'):
    """
    Load the ProphetNet-Large model.

    Args:
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (ProphetNetForConditionalGeneration): prophetnet-base model mapped to device.
        tokenizer (ProphetNetTokenizer): prophetnet-base tokenizer mapped to device.
    """

    if finetune_type == 'full':
        model = ProphetNetForConditionalGeneration.from_pretrained(f'microsoft/{name}')
    
    elif finetune_type == 'lora':
        model = ProphetNetForConditionalGeneration.from_pretrained(f'microsoft/{name}')
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query_proj", "value_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    
    else:
        config = ProphetNetConfig.from_pretrained(f'microsoft/{name}')
        model = AutoAdapterModel.from_pretrained(f'microsoft/{name}', config=config)

    tokenizer = ProphetNetTokenizer.from_pretrained(f'microsoft/{name}')

    return model, tokenizer