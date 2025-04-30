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

def ModelGPT2ForQuestionAnswering(name='gpt2', finetune_type='full', device='cpu'):
    """
    Load the GPT-2 model for question answering.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForCausalLM): gpt2 model mapped to device.
        tokenizer (AutoTokenizer): gpt2 tokenizer mapped to device.
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
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TRANSLATION ============================= #

def ModelGPT2ForTranslation(name='gpt2', finetune_type='full', device='cpu'):
    """
    Load the GPT-2 model for translation.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForCausalLM): gpt2 model mapped to device.
        tokenizer (AutoTokenizer): gpt2 tokenizer mapped to device.
    """
    model_path = name

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForCausalLM.from_pretrained(model_path)

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

            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)
        
        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = AutoModelForCausalLM.from_pretrained(
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
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TEXT SENTIMENT ============================= #
def ModelGPT2ForTextSentiment(name='gpt2', finetune_type='full', device='cpu'):
    """
    Load the GPT-2 model for text sentiment analysis.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForCausalLM): gpt2 model mapped to device.
        tokenizer (AutoTokenizer): gpt2 tokenizer mapped to device.
    """

    model_path = name

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForCausalLM.from_pretrained(model_path)

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

            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = AutoModelForCausalLM.from_pretrained(
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
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer