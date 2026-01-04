import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, LoraConfig


CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\\n' }}{% endif %}{% endfor %}"


def get_response_template(model_name: str, tokenizer) -> str:
    """
    Get response template based on model name
    """
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        return "<|im_start|>assistant\n"
    elif "gemma" in model_name_lower:
        return "<start_of_turn>model"
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        return "[/INST]"
    else:
        return "<start_of_turn>model"


def load_model_and_tokenizer(model_name: str = "beomi/gemma-ko-2b", torch_dtype: str = "float16"):
    """
    Load model and tokenizer for training
    """
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    dtype = dtype_mapping.get(torch_dtype, torch.float16)
    
    # 4bit 모델 감지
    is_bnb_4bit = "bnb-4bit" in model_name.lower()
    
    if is_bnb_4bit:
        print(f"🔧 4-bit 양자화 모델 감지: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if not tokenizer.chat_template:
        print("내장된 Chat Template 없음. 커스텀 Chat Template 설정")
        tokenizer.chat_template = CHAT_TEMPLATE
    else:
        print("내장된 Chat Template 있음. 내장된 Chat Template 사용")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_model_for_inference(checkpoint_path: str, torch_dtype: str = "float16"):
    """
    Load model from checkpoint for inference
    """
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    dtype = dtype_mapping.get(torch_dtype, torch.float16)
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    return tokenizer
