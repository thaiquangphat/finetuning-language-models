from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, T5Config, 
    BitsAndBytesConfig, AutoModelForSeq2SeqLM
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
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

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(name)

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

        model = AutoModelForSeq2SeqLM.from_pretrained(name, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)

        # get the model with LoRA
        model = get_peft_model(model, lora_config)

    else: # adapters
        config = T5Config.from_pretrained(name)
        model = AutoAdapterModel.from_pretrained(name, config=config)

        model.add_adapter("question_answering")
        model.set_active_adapters("question_answering")

    # Create the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(name)

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

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(name)

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

        model = AutoModelForSeq2SeqLM.from_pretrained(name, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)

        # get the model with LoRA
        model = get_peft_model(model, lora_config)
    else: # adapters
        config = T5Config.from_pretrained(name)
        model = AutoAdapterModel.from_pretrained(name, config=config)

        model.add_adapter("translation")
        model.set_active_adapters("translation")

    # Create the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(name)

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

    # Load the model
    if finetune_type == 'full':
        model = AutoModelForSeq2SeqLM.from_pretrained(name)

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

        model = T5ForConditionalGeneration.from_pretrained(name, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)

        # get the model with LoRA
        model = get_peft_model(model, lora_config)
    else: # adapters
        config = T5Config.from_pretrained(name)
        model = AutoAdapterModel.from_pretrained(name, config=config)

        model.add_adapter("text_sentiment_analysis")
        model.set_active_adapters("text_sentiment_analysis")

    # Create the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(name)

    # Move model to device
    model.to(device)

    return model, tokenizer