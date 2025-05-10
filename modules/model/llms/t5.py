from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, T5Config, 
    BitsAndBytesConfig, AutoModelForSeq2SeqLM
)
from peft import (
    LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, # prepare model for training
    PeftModel, PeftConfig # for loading the finetuned lora model
)
from adapters import AutoAdapterModel

# ============================= MODEL FOR QUESTION ANSWERING ============================= #

def ModelT5ForQuestionAnswering(name='t5-base', finetune_type='full', device='cpu'):
    """
    Load the T5-Base model for question answering.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): t5-base model mapped to device.
        tokenizer (AutoTokenizer): t5-base tokenizer mapped to device.
    """
    if 'ft' in name:
        model_path = name
    else:
        model_path = f'google/{name}'

    # Load the model
    if finetune_type == 'full':
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
                target_modules=["q", "k", "v"], # ["q", "k", "v", "o"]
                bias="none"
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                peft_config.base_model_name_or_path, 
                quantization_config=bnb_config
            ) # load the base model

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # adapters
        # remove 'google' from model path if load from hf
        if 'ft' not in model_path:
            model_hf = model_path[7:]

        config = T5Config.from_pretrained(model_hf)
        model = AutoAdapterModel.from_pretrained(model_hf, config=config)

        model.add_adapter("question_answering")
        model.set_active_adapters("question_answering")

    # Create the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TEXT TRANSLATION ============================= #

def ModelT5ForTranslation(name='t5-base', finetune_type='full', device='cpu'):
    """
    Load the T5-Base model for translation.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): t5-base model mapped to device.
        tokenizer (AutoTokenizer): t5-base tokenizer mapped to device.
    """
    if 'ft' in name:
        model_path = name
    else:
        model_path = f'google/{name}'
    
    # Load the model
    if finetune_type == 'full':
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
                target_modules=["q", "k", "v"], # ["q", "k", "v", "o"]
                bias="none"
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)
        
        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                peft_config.base_model_name_or_path, 
                quantization_config=bnb_config
            ) # load the base model

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # adapters
        # remove 'google' from model path if load from hf
        if 'ft' not in model_path:
            model_hf = model_path[7:]

        config = T5Config.from_pretrained(model_hf)
        model = AutoAdapterModel.from_pretrained(model_hf, config=config)

        model.add_adapter("translation")
        model.set_active_adapters("translation")

    # Create the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TEXT SENTIMENT ANALYSIS ============================= #

def ModelT5ForTextSentiment(name='t5-base', finetune_type='full', device='cpu'):
    """
    Load the T5-Base model for text sentiment analysis.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): t5-base model mapped to device.
        tokenizer (AutoTokenizer): t5-base tokenizer mapped to device.
    """
    if 'ft' in name:
        model_path = name
    else:
        model_path = f'google/{name}'

    # Load the model
    if finetune_type == 'full':
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
                target_modules=["q", "k", "v"], # ["q", "k", "v", "o"]
                bias="none"
            )

            model = T5ForConditionalGeneration.from_pretrained(model_path, quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

        else: # load the model for inference
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            base_model = T5ForConditionalGeneration.from_pretrained(
                peft_config.base_model_name_or_path, 
                quantization_config=bnb_config
            ) # load the base model

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode
        
    else: # adapters
        # remove 'google' from model path if load from hf
        if 'ft' not in model_path:
            model_hf = model_path[7:]

        config = T5Config.from_pretrained(model_hf)
        model = AutoAdapterModel.from_pretrained(model_hf, config=config)

        model.add_adapter("text_sentiment_analysis")
        model.set_active_adapters("text_sentiment_analysis")

    # Create the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer