from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, # prepare model for training
    PeftModel, PeftConfig # for loading the finetuned lora model
)
from adapters import AutoAdapterModel
import torch.nn as nn

# ============================= MODEL FOR QUESTION ANSWERING ============================= #

class GPT2ForExtractiveQA(GPT2LMHeadModel):
    """
    A custom GPT-2 model for extractive question answering.
    
    This class extends GPT2LMHeadModel to add extractive QA capabilities by:
    1. Adding custom linear layers for predicting start and end positions
    2. Implementing a custom forward pass that returns start/end logits
    3. Computing loss for training when labels are provided
    
    Attributes:
        start_classifier (nn.Linear): Linear layer for predicting start positions
        end_classifier (nn.Linear): Linear layer for predicting end positions
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Adding custom linear layers for start and end positions
        self.start_classifier = nn.Linear(config.n_embd, 1)
        self.end_classifier = nn.Linear(config.n_embd, 1)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, start_positions=None, end_positions=None):
        # Get the GPT2 outputs
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Predict start and end positions using linear layers
        start_logits = self.start_classifier(hidden_states).squeeze(-1)  # Shape: (batch_size, seq_len)
        end_logits = self.end_classifier(hidden_states).squeeze(-1)    # Shape: (batch_size, seq_len)

        # If labels are provided (for training), calculate the loss
        if labels is not None:
            # Compute the loss (using CrossEntropyLoss)
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            return loss, (start_logits, end_logits)
        
        return start_logits, end_logits
    
def get_gpt2_tokenizer(name='gpt2'):
    """
    Creates and configures a GPT-2 tokenizer with special tokens.
    
    This function initializes a GPT-2 tokenizer and adds necessary special tokens
    for question answering tasks, particularly the [PAD] token.
    
    Args:
        name (str): Name of the pretrained model (default: 'gpt2')
        
    Returns:
        GPT2Tokenizer: Configured tokenizer with special tokens
    """
    # Create the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(name)

    # Add special tokens for question answering
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]'
    })

    return tokenizer

def ModelGPT2ForQuestionAnswering(name='gpt2', finetune_type='full', device='cpu'):
    """
    Loads and configures a GPT-2 model for extractive question answering.
    
    This function handles loading the GPT-2 model with different fine-tuning approaches:
    - Full fine-tuning using custom GPT2ForExtractiveQA model
    - LoRA (Low-Rank Adaptation)
    - Adapters (TODO)
    
    Args:
        name (str): Name of the pretrained model (default: 'gpt2')
        finetune_type (str): Fine-tuning technique to use ('full', 'lora', or 'adapters')
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured GPT-2 model for extractive QA
            - tokenizer: The corresponding GPT-2 tokenizer
    """
    model_path = name

    # Load the model
    if finetune_type == 'full':
        model = GPT2ForExtractiveQA.from_pretrained(model_path) # GPT2 uses custom model for extractive QA

    elif finetune_type == 'lora':
        if 'ft' not in model_path: # prepare model for training
            bnb_config=BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                r=32,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_proj"],
                bias="none"
            )

            model = GPT2ForExtractiveQA.from_pretrained(model_path, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)
        
        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = GPT2ForExtractiveQA.from_pretrained(
                peft_config.base_model_name_or_path, 
                quantization_config=bnb_config
            ) # load the base model

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # TODO: adapters
        config = GPT2Config.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=config)

        model.add_adapter("question_answering")
        model.set_active_adapters("question_answering")

    # Create the tokenizer
    tokenizer = get_gpt2_tokenizer(model_path)

    # Resize the token embeddings to match the tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TRANSLATION ============================= #

def ModelGPT2ForTranslation(name='gpt2', finetune_type='full', device='cpu'):
    """
    Loads and configures a GPT-2 model for translation tasks.
    
    This function handles loading the GPT-2 model with different fine-tuning approaches:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - Adapters (TODO)
    
    Args:
        name (str): Name of the pretrained model (default: 'gpt2')
        finetune_type (str): Fine-tuning technique to use ('full', 'lora', or 'adapters')
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured GPT-2 model for translation
            - tokenizer: The corresponding GPT-2 tokenizer
    """
    model_path = name

    # Load the model
    if finetune_type == 'full':
        model = GPT2LMHeadModel.from_pretrained(model_path)

    elif finetune_type == 'lora':
        if 'ft' not in model_path: # prepare model for training
            bnb_config=BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                r=32,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_proj"],
                bias="none"
            )

            model = GPT2LMHeadModel.from_pretrained(name, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)
        
        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = GPT2LMHeadModel.from_pretrained(
                peft_config.base_model_name_or_path, 
                quantization_config=bnb_config
            ) # load the base model

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # TODO: adapters
        config = GPT2Config.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=config)

        model.add_adapter("translation")
        model.set_active_adapters("translation")

    # Create the tokenizer
    tokenizer = get_gpt2_tokenizer(model_path)

    # Resize the token embeddings to match the tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # Set the pad token id in the model config
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TEXT SENTIMENT ============================= #
def ModelGPT2ForTextSentiment(name='gpt2', finetune_type='full', device='cpu'):
    """
    Loads and configures a GPT-2 model for text sentiment analysis.
    
    This function handles loading the GPT-2 model with different fine-tuning approaches:
    - Full fine-tuning
    - LoRA (Low-Rank Adaptation)
    - Adapters (TODO)
    
    Args:
        name (str): Name of the pretrained model (default: 'gpt2')
        finetune_type (str): Fine-tuning technique to use ('full', 'lora', or 'adapters')
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer) where:
            - model: The configured GPT-2 model for sentiment analysis
            - tokenizer: The corresponding GPT-2 tokenizer
    """

    model_path = name

    # Load the model
    if finetune_type == 'full':
        model = GPT2LMHeadModel.from_pretrained(model_path)

    elif finetune_type == 'lora':
        if 'ft' not in name: # prepare model for training
            bnb_config=BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                r=32,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_proj"],
                bias="none"
            )

            model = GPT2LMHeadModel.from_pretrained(model_path, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = GPT2LMHeadModel.from_pretrained(
                peft_config.base_model_name_or_path, 
                quantization_config=bnb_config
            ) # load the base model

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # TODO: adapters
        config = GPT2Config.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=config)

        model.add_adapter("text_sentiment_analysis")
        model.set_active_adapters("text_sentiment_analysis")

    # Create the tokenizer
    tokenizer = get_gpt2_tokenizer(model_path)

    # Resize the token embeddings to match the tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # Set the pad token id in the model config
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # Move model to device
    model.to(device)

    return model, tokenizer