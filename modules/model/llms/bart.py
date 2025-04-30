from transformers import (
    BartTokenizer, BartForConditionalGeneration, BartConfig, 
    AutoModelForSeq2SeqLM, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from adapters import AutoAdapterModel

# ============================= MODEL FOR QUESTION ANSWERING ============================= #

def ModelBartForQuestionAnswering(name='bart-base', finetune_type='full', device='cpu'):
    """
    Load the Bart-Base model for question answering.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): bart-base model mapped to device.
        tokenizer (AutoTokenizer): bart-base tokenizer mapped to device.
    """

    if 'models' in name: # Load the finetuned model from local path
        model_path = name
    else: # Load the finetuned model from HuggingFace
        model_path = f'facebook/{name}'

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    elif finetune_type == 'lora':
        bnb_config=BitsAndBytesConfig(
            load_in_8bit=True
        )
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
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

    else: # adapters
        config = BartConfig.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=config)

        model.add_adapter('question_answering')
        model.set_active_adapters('question_answering')

    # Create the tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TRANSLATION ============================= #

def ModelBartForTranslation(name='bart-base', finetune_type='full', device='cpu'):
    """
    Load the Bart-Base model for translation.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): bart-base model mapped to device.
        tokenizer (AutoTokenizer): bart-base tokenizer mapped to device.
    """

    if 'models' in name: # Load the finetuned model from local path
        model_path = name
    else: # Load the finetuned model from HuggingFace
        model_path = f'facebook/{name}'

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    elif finetune_type == 'lora':
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

    else: # adapters
        config = BartConfig.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=config)

        model.add_adapter('translation')
        model.set_active_adapters('translation')

    # Create the tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer

# ============================= MODEL FOR TEXT SENTIMENT ============================= #

def ModelBartForTextSentiment(name='bart-base', finetune_type='full', device='cpu'):
    """
    Load the Bart-Base model for text sentiment analysis.

    Args:
        name (str): name of the pretrained model.
        finetune_type (str): name of the finetune technique.
        task (str): name of the finetune task.
        device (torch): device used for training: cuda or cpu.
    Returns: 
        model (AutoModelForSeq2SeqLM): bart-base model mapped to device.
        tokenizer (AutoTokenizer): bart-base tokenizer mapped to device.
    """

    if 'models' in name: # Load the finetuned model from local path
        model_path = name
    else: # Load the finetuned model from HuggingFace
        model_path = f'facebook/{name}'

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    elif finetune_type == 'lora':
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

    else: # adapters
        config = BartConfig.from_pretrained(model_path)
        model = AutoAdapterModel.from_pretrained(model_path, config=config)

        model.add_adapter('text_sentiment_analysis')
        model.set_active_adapters('text_sentiment_analysis')

    # Create the tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer
    