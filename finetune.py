import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from data_preprocessing import load_and_preprocess_data
import os

def setup_model_and_tokenizer(model_path="meta-llama/Llama-3.2-1B-Instruct", use_4bit=False):
    """모델과 토크나이저 설정"""
    print(f"Loading model from {model_path}...")

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def setup_lora_config():
    """LoRA 설정"""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config

def setup_training_args(output_dir="./results", num_epochs=3, batch_size=1):
    """학습 파라미터 설정"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_steps=500,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=torch.cuda.is_available(),
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

def main():
    print("=" * 60)
    print("LLaMA 3.2 1B LoRA 파인튜닝 시작")
    print("=" * 60)

    # Hugging Face 로그인
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("⚠️ HF_TOKEN 환경변수가 설정되지 않았습니다.")
    print("Hugging Face Hub에 로그인 완료")

    print("\n1. 데이터셋 로드 및 전처리")
    train_dataset = load_and_preprocess_data("train")
    validation_dataset = load_and_preprocess_data("validation")

    train_subset = train_dataset.select(range(min(1000, len(train_dataset))))
    val_subset = validation_dataset.select(range(min(100, len(validation_dataset))))

    print(f"Using {len(train_subset)} training samples and {len(val_subset)} validation samples")

    print("\n2. 모델 및 토크나이저 설정")
    model, tokenizer = setup_model_and_tokenizer(use_4bit=torch.cuda.is_available())

    print("\n3. LoRA 설정 적용")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    print("\n4. 학습 설정")
    training_args = setup_training_args(
        output_dir="./results",
        num_epochs=3,
        batch_size=1
    )

    print("\n5. Trainer 설정")

    def formatting_func(example):
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        formatting_func=formatting_func,
        args=training_args,
        peft_config=lora_config,
    )

    print("\n6. 학습 시작")
    print("This will take some time...")
    trainer.train()

    print("\n7. 모델 저장")
    adapter_path = "./lora_adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\nLoRA adapter saved to {adapter_path}")
    print("파인튜닝 완료!")

if __name__ == "__main__":
    main()