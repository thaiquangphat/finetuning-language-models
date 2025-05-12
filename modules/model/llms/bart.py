from transformers import (
    BartTokenizer, BartForConditionalGeneration, BartConfig, 
    AutoModelForSeq2SeqLM, BitsAndBytesConfig
)
from peft import (
    LoraConfig, PrefixTuningConfig, 
    TaskType, PeftType,
    get_peft_model, prepare_model_for_kbit_training, # prepare model for training
    PeftModel, PeftConfig # for loading the finetuned lora model
)
from adapters import AutoAdapterModel, AdapterConfig, BartAdapterModel
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
from torch import nn
import torch.nn.functional as F
from modules.train.ultis import debug_print # For debugging

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

    if 'ft' in name: # Load the finetuned model from local path
        model_path = name
    else: # Load the finetuned model from HuggingFace
        model_path = f'facebook/{name}'

    # Load the model
    if finetune_type == 'full':
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    elif finetune_type == 'lora':
        if 'ft' not in model_path: # prepare model for training
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4"
            # )

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                r=32,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # ["q_proj", "v_proj", "k_proj", "out_proj"]
                bias="none"
            )

            # model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            # model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

            # debug_print(title='BART LoRA training Model', task_type='SEQ_2_SEQ_LM')

        else: # load the model for inference
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True
            # )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            # base_model = AutoModelForSeq2SeqLM.from_pretrained(
            #     peft_config.base_model_name_or_path,
            #     quantization_config=bnb_config
            # ) # load the base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode
            
    else: # adapters
        # config = BartConfig.from_pretrained(model_path)
        # model = _BartAdapterModel.from_pretrained(model_path, config=config)
        if 'ft' not in model_path: # prepare model for training
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4"
            # )

            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                num_virtual_tokens=512,
                encoder_hidden_size=model.config.d_model,
            )

            # model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            # model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, peft_config)

            # debug_print(title='BART LoRA training Model', task_type='SEQ_2_SEQ_LM')

        else: # load the model for inference
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True
            # )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            # base_model = AutoModelForSeq2SeqLM.from_pretrained(
            #     peft_config.base_model_name_or_path,
            #     quantization_config=bnb_config
            # ) # load the base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

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

    if 'ft' in name: # Load the finetuned model from local path
        model_path = name
    else: # Load the finetuned model from HuggingFace
        model_path = f'facebook/{name}'

    # Load the model
    if finetune_type == 'full':
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    elif finetune_type == 'lora':
        if 'ft' not in model_path: # prepare model for training
            # bnb_config=BitsAndBytesConfig(
            #     load_in_8bit=True
            # )
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                r=32,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # ["q_proj", "v_proj", "k_proj", "out_proj"]
                bias="none"
            )

            # model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            # model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

        else: # load the model for inference
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True
            # )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            # base_model = AutoModelForSeq2SeqLM.from_pretrained(
            #     peft_config.base_model_name_or_path,
            #     quantization_config=bnb_config
            # ) # load the base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # adapters
        # config = BartConfig.from_pretrained(model_path)
        # model = _BartAdapterModel.from_pretrained(model_path, config=config)
        if 'ft' not in model_path: # prepare model for training
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4"
            # )

            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                num_virtual_tokens=512,
                encoder_hidden_size=model.config.d_model,
            )

            # model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            # model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, peft_config)

            # debug_print(title='BART LoRA training Model', task_type='SEQ_2_SEQ_LM')

        else: # load the model for inference
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True
            # )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            # base_model = AutoModelForSeq2SeqLM.from_pretrained(
            #     peft_config.base_model_name_or_path,
            #     quantization_config=bnb_config
            # ) # load the base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

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

    if 'ft' in name: # Load the finetuned model from local path
        model_path = name
    else: # Load the finetuned model from HuggingFace
        model_path = f'facebook/{name}'

    # Load the model
    if finetune_type == 'full':
        # Create the model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    elif finetune_type == 'lora':
        if 'ft' not in model_path: # prepare model for training
            # bnb_config=BitsAndBytesConfig(
            #     load_in_8bit=True
            # )
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                r=32,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # ["q_proj", "v_proj", "k_proj", "out_proj"]
                bias="none"
            )

            # model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            # model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, lora_config)

        else: # load the model for inference
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True
            # )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            # base_model = AutoModelForSeq2SeqLM.from_pretrained(
            #     peft_config.base_model_name_or_path,
            #     quantization_config=bnb_config
            # ) # load the base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    else: # adapters
        # config = BartConfig.from_pretrained(model_path)
        # model = _BartAdapterModel.from_pretrained(model_path, config=config)
        if 'ft' not in model_path: # prepare model for training
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4"
            # )

            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False, # Set to False for training
                num_virtual_tokens=512,
                encoder_hidden_size=model.config.d_model,
            )

            # model = AutoModelForSeq2SeqLM.from_pretrained(model_path, quantization_config=bnb_config)
            # model = prepare_model_for_kbit_training(model)

            # get the model with LoRA
            model = get_peft_model(model, peft_config)

            # debug_print(title='BART LoRA training Model', task_type='SEQ_2_SEQ_LM')

        else: # load the model for inference
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True
            # )

            peft_config = PeftConfig.from_pretrained(model_path) # load the finetuned model config
            # base_model = AutoModelForSeq2SeqLM.from_pretrained(
            #     peft_config.base_model_name_or_path,
            #     quantization_config=bnb_config
            # ) # load the base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False # stop updating the model weights
            ) # load the finetuned model

            model.eval() # set the model to evaluation mode

    # Create the tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # Move model to device
    model.to(device)

    return model, tokenizer
    