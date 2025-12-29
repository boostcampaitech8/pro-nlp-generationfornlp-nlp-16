import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training


CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"


def get_response_template(model_name: str, tokenizer) -> str:
    """
    Get response template based on model name
    
    간단한 fallback 함수. Config에 없을 때만 사용됨.
    """
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        return "<|im_start|>assistant"
    elif "gemma" in model_name_lower:
        return "<start_of_turn>model"
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        return "[/INST]"
    else:
        # 기본값
        return "<start_of_turn>model"


def load_model_and_tokenizer(model_name: str = "beomi/gemma-ko-2b", torch_dtype: str = "float16", quantization_config: dict = None):
    """
    Load model and tokenizer for training (with optional QLoRA quantization)

    Args:
        model_name: HuggingFace model name
        torch_dtype: Data type for model weights ('float16', 'bfloat16', 'float32', or 'auto')
        quantization_config: Dict containing bitsandbytes configuration
    """
    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    dtype = dtype_mapping.get(torch_dtype, torch.float16)

    bnb_config = None
    if quantization_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', False),
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', False),
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=dtype_mapping.get(quantization_config.get('bnb_4bit_compute_dtype', 'float16'), torch.float16)
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0} if bnb_config else None, # single GPU일 때만 명시
    )

    if bnb_config:
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        model.graidient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set chat template
    if not tokenizer.chat_template:
        print("내장된 Chat Template 없음. 커스텀 Chat Template 설정")
        tokenizer.chat_template = CHAT_TEMPLATE
    else:
        print("내장된 Chat Template 있음. 내장된 Chat Template 사용")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_model_for_inference(checkpoint_path: str, torch_dtype: str = "float16", quantization_config: dict = None):
    """
    Load model from checkpoint for inference
    
    Args:
        checkpoint_path: Path to checkpoint directory
        torch_dtype: Data type for model weights ('float16', 'bfloat16', 'float32', or 'auto')
        quantization_config: Dict containing bitsandbytes configuration
    """
    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    dtype = dtype_mapping.get(torch_dtype, torch.float16)

    bnb_config = None
    if quantization_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', False),
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', False),
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=dtype_mapping.get(quantization_config.get('bnb_4bit_compute_dtype', 'float16'), torch.float16)
        )
        print(f'Applying 4-bit QUantization for Inference: {bnb_config}')

    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map={"": 0}, # single GPU일 때만 명시
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )

    return model, tokenizer


def get_peft_config(
    r: int = 6,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Get PEFT (LoRA) configuration
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj']

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
    return peft_config


def setup_tokenizer_for_training(tokenizer):
    """
    Setup tokenizer for training (pad token, padding side)
    """
    # pad token 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    return tokenizer
