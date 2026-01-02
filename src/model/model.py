import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: unsloth not available. Using standard transformers.")


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


def load_model_and_tokenizer(
    model_name: str = "beomi/gemma-ko-2b", 
    torch_dtype: str = "float16", 
    quantization_config: dict = None,
    use_unsloth: bool = False,
    max_seq_length: int = 2048,
):
    """
    Load model and tokenizer for training (with optional QLoRA quantization and Unsloth)

    Args:
        model_name: HuggingFace model name
        torch_dtype: Data type for model weights ('float16', 'bfloat16', 'float32', or 'auto')
        quantization_config: Dict containing bitsandbytes configuration
        use_unsloth: Whether to use Unsloth for faster training
        max_seq_length: Maximum sequence length (for Unsloth)
    """
    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    dtype = dtype_mapping.get(torch_dtype, torch.float16)

    # Unsloth 사용 시
    if use_unsloth and UNSLOTH_AVAILABLE:
        print("🚀 Using Unsloth for faster training!")
        
        # Unsloth는 4bit 양자화를 자체적으로 처리
        load_in_4bit = quantization_config.get('load_in_4bit', False) if quantization_config else False
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype if dtype != "auto" else None,  # Unsloth는 None으로 auto 처리
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        
        print("✓ Unsloth model loaded successfully")
        
    # 기존 방식 (transformers + PEFT)
    else:
        if use_unsloth and not UNSLOTH_AVAILABLE:
            print("⚠️ Unsloth requested but not available. Using standard transformers.")
        
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
            model.gradient_checkpointing_enable()

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


def load_model_for_inference(
    checkpoint_path: str, 
    torch_dtype: str = "float16", 
    quantization_config: dict = None,
    use_unsloth: bool = False,
    max_seq_length: int = 2048,
):
    """
    Load model from checkpoint for inference
    
    Args:
        checkpoint_path: Path to checkpoint directory
        torch_dtype: Data type for model weights ('float16', 'bfloat16', 'float32', or 'auto')
        quantization_config: Dict containing bitsandbytes configuration
        use_unsloth: Whether to use Unsloth for faster inference
        max_seq_length: Maximum sequence length (for Unsloth)
    """
    # 일부 transformers 버전에서 config.quantization_config가 None일 때 __repr__ 호출 중 to_dict() 에러가 발생하는 경우가 있어
    # INFO 로그를 끄고 로딩한다.
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_warning()

    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto"
    }
    dtype = dtype_mapping.get(torch_dtype, torch.float16)

    # Unsloth 사용 시
    if use_unsloth and UNSLOTH_AVAILABLE:
        print("🚀 Using Unsloth for faster inference!")
        
        # Unsloth로 학습된 모델 로드
        load_in_4bit = quantization_config.get('load_in_4bit', False) if quantization_config else False
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=max_seq_length,
            dtype=dtype if dtype != "auto" else None,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        
        # Inference mode로 전환 (더 빠른 추론)
        FastLanguageModel.for_inference(model)
        print("✓ Unsloth model loaded for inference")
        
    # 기존 방식 (transformers + PEFT)
    else:
        if use_unsloth and not UNSLOTH_AVAILABLE:
            print("⚠️ Unsloth requested but not available. Using standard transformers.")
        
        bnb_config = None
        if quantization_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_config.get('load_in_4bit', False),
                bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', False),
                bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_compute_dtype=dtype_mapping.get(quantization_config.get('bnb_4bit_compute_dtype', 'float16'), torch.float16)
            )
            print(f'Applying 4-bit QUantization for Inference: {bnb_config}')

        # transformers/peft 조합에서 quantization_config가 None일 때 __repr__ 경로로 to_dict()가 호출되어 터지는 케이스가 있어
        # quantization_config가 있을 때만 인자로 전달한다.
        peft_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": {"": 0},  # single GPU일 때만 명시
        }
        if bnb_config:
            peft_kwargs["quantization_config"] = bnb_config

        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            **peft_kwargs,
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
    task_type: str = "CAUSAL_LM",
    use_unsloth: bool = False,
) -> LoraConfig:
    """
    Get PEFT (LoRA) configuration
    
    Args:
        use_unsloth: If True and using Unsloth, returns None (Unsloth handles LoRA internally)
    """
    # Unsloth를 사용하는 경우, get_peft_model을 직접 호출하므로 None 반환
    if use_unsloth and UNSLOTH_AVAILABLE:
        return None
    
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


def apply_lora_to_model(
    model,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    target_modules: list = None,
    use_gradient_checkpointing: str = "unsloth",
    use_rslora: bool = False,
    use_unsloth: bool = True,
):
    """
    Apply LoRA to model using Unsloth's get_peft_model
    
    Args:
        model: The model to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: Target modules for LoRA (None for auto-detection)
        use_gradient_checkpointing: Gradient checkpointing mode ("unsloth" or True/False)
        use_rslora: Whether to use RSLoRA
        use_unsloth: Whether to use Unsloth's optimized LoRA
    """
    if not use_unsloth or not UNSLOTH_AVAILABLE:
        raise ValueError("apply_lora_to_model requires Unsloth to be available")
    
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=42,
        use_rslora=use_rslora,
        loftq_config=None,
    )
    
    return model


def setup_tokenizer_for_training(tokenizer):
    """
    Setup tokenizer for training (pad token, padding side)
    """
    # pad token 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    return tokenizer
