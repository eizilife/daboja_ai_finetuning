#!/usr/bin/env python3
"""
보고서 요약 전용 Llama-3.2-1B 파인튜닝 스크립트
"""

import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
from huggingface_hub import login

def load_dataset_from_json(file_path: str) -> Dataset:
    """JSON 파일에서 데이터셋 로드"""
    print(f"📂 데이터셋 로드: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ {len(data)}개 데이터 로드 완료")
    return Dataset.from_list(data)

def setup_model_and_tokenizer(model_path="meta-llama/Llama-3.2-1B-Instruct", use_4bit=False):
    """모델과 토크나이저 설정"""
    print(f"🔄 모델 로드: {model_path}")

    # 양자화 설정 (GPU 사용 시)
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4bit 양자화 모델 준비
    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    print(f"✅ 모델 로드 완료 (Device: {'GPU' if torch.cuda.is_available() else 'CPU'})")
    return model, tokenizer

def setup_lora_config():
    """LoRA 설정 - 요약 태스크에 최적화"""
    lora_config = LoraConfig(
        r=32,  # 요약 태스크에는 조금 더 높은 rank 사용
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config

def setup_training_args(output_dir="./models/summary-model", num_epochs=3, batch_size=2):
    """학습 파라미터 설정 - 요약 태스크에 최적화"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # 배치 크기를 효과적으로 늘림
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_steps=200,
        logging_steps=10,
        learning_rate=1e-4,  # 요약 태스크에는 조금 낮은 학습률
        weight_decay=0.01,
        fp16=False,
        bf16=torch.cuda.is_available(),
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # Windows에서 안정성을 위해
        remove_unused_columns=False,
    )

def formatting_func(example):
    """데이터 포맷팅 함수"""
    return example["text"]

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("📝 보고서 요약 모델 파인튜닝 시작")
    print("=" * 80)

    # Hugging Face 로그인
    print("\n🔑 Hugging Face 로그인")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ 로그인 완료")
    else:
        print("⚠️ HF_TOKEN 환경변수가 설정되지 않았습니다.")

    # 1. 데이터셋 로드
    print("\n📊 데이터셋 로드")
    train_dataset = load_dataset_from_json("../data/processed/train_dataset.json")
    val_dataset = load_dataset_from_json("../data/processed/val_dataset.json")

    if train_dataset is None or val_dataset is None:
        print("❌ 데이터셋 로드 실패. 먼저 prepare_data.py를 실행하세요.")
        return

    # 데이터 개수 제한 (메모리 절약)
    max_train_samples = 5000
    max_val_samples = 500

    if len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
        print(f"🔢 학습 데이터를 {max_train_samples}개로 제한")

    if len(val_dataset) > max_val_samples:
        val_dataset = val_dataset.select(range(max_val_samples))
        print(f"🔢 검증 데이터를 {max_val_samples}개로 제한")

    print(f"📈 최종 데이터: 학습 {len(train_dataset)}개, 검증 {len(val_dataset)}개")

    # 2. 모델 및 토크나이저 설정
    print("\n🤖 모델 설정")
    use_4bit = torch.cuda.is_available()
    model, tokenizer = setup_model_and_tokenizer(use_4bit=use_4bit)

    # 3. LoRA 설정
    print("\n⚙️ LoRA 설정")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    # 파라미터 정보 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 학습 가능한 파라미터: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 4. 학습 설정
    print("\n🎯 학습 설정")
    training_args = setup_training_args(
        output_dir="../models/summary-lora",
        num_epochs=3,
        batch_size=1 if not torch.cuda.is_available() else 2
    )

    # 5. Trainer 설정
    print("\n🏃 Trainer 설정")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        args=training_args,
        peft_config=lora_config,
    )

    # 6. 학습 시작
    print("\n🚀 학습 시작")
    print("⏰ 요약 모델 파인튜닝은 시간이 걸릴 수 있습니다...")

    # 체크포인트 확인
    checkpoint_dir = "../report-summarization/models/summary-lora/checkpoint-400"
    if os.path.exists(checkpoint_dir):
        print(f"📌 체크포인트 발견: {checkpoint_dir}")
        print("🔄 이전 학습을 이어서 진행합니다...")
        resume_from_checkpoint = checkpoint_dir
    else:
        print("🆕 새로운 학습을 시작합니다...")
        resume_from_checkpoint = None

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("✅ 학습 완료!")
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {str(e)}")
        return

    # 7. 모델 저장
    print("\n💾 모델 저장")
    model_save_path = "../models/summary-lora-adapter"

    # 디렉토리 생성
    os.makedirs(model_save_path, exist_ok=True)

    # 모델 저장
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"✅ 모델 저장 완료: {model_save_path}")

    # 8. 학습 결과 저장
    print("\n📊 학습 로그 저장")
    log_history = trainer.state.log_history

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/training_log.json", 'w', encoding='utf-8') as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("🎉 보고서 요약 모델 파인튜닝 완료!")
    print(f"📁 모델 위치: {model_save_path}")
    print(f"📈 학습 로그: {results_dir}/training_log.json")
    print("=" * 80)

if __name__ == "__main__":
    main()