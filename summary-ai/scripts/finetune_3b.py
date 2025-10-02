#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Llama 3.2 3B 모델을 사용한 향상된 요약 모델 파인튜닝 스크립트
더 큰 모델과 더 많은 에폭으로 성능 개선
"""

import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

def setup_model_and_tokenizer(model_name="meta-llama/Llama-3.2-3B-Instruct", use_4bit=True):
    """3B 모델과 토크나이저 설정 - GPU 강제 사용"""
    print(f"🔄 모델 로드 중: {model_name}")

    # GPU 확인
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU가 사용 불가능합니다!")

    print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        print("⚡ 4비트 양자화 활성화 (GPU 메모리 효율)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0",  # GPU 0 강제 지정
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        print("✅ 4비트 양자화 모델 GPU에 로드 완료")
    else:
        print("💻 전체 정밀도로 GPU 사용")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # GPU 0 강제 지정
            trust_remote_code=True
        )
        print("✅ 전체 정밀도 모델 GPU에 로드 완료")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        print(f"📊 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

    return model, tokenizer

def setup_lora_config():
    """3B 모델에 최적화된 LoRA 설정"""
    return LoraConfig(
        r=16,  # 더 큰 rank로 표현력 증가
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

def setup_training_args(output_dir, num_epochs=3, batch_size=2):
    """학습 파라미터 설정 - finetune.py와 동일한 설정"""
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
        learning_rate=1e-4,  # finetune.py와 동일한 학습률
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

def create_prompt(example):
    """완성된 문장을 생성하도록 개선된 프롬프트"""
    instruction = """다음 텍스트를 읽고 핵심 내용을 2-3개의 완전한 문장으로 요약하세요.
요약은 반드시 완성된 문장으로 작성하고, 원문에 없는 내용은 추가하지 마세요."""

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
텍스트: {example['input']}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
요약: {example['output']}<|eot_id|>"""

    return {"text": prompt}

def load_and_prepare_dataset(file_path):
    """데이터셋 로드 및 전처리"""
    if not os.path.exists(file_path):
        print(f"⚠️ 파일이 없습니다: {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터 검증 및 정제
    cleaned_data = []
    for item in data:
        if len(item['input']) > 100 and len(item['output']) > 20:
            # 요약이 완전한 문장인지 확인
            if item['output'].strip().endswith(('.', '!', '?', '"', '다', '요')):
                cleaned_data.append(item)

    dataset = Dataset.from_list(cleaned_data)
    dataset = dataset.map(create_prompt, remove_columns=['input', 'output'])

    return dataset

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🚀 Llama 3.2 3B 요약 모델 파인튜닝 시작")
    print("=" * 80)

    # Hugging Face 로그인 (선택적)
    print("\n🔑 Hugging Face 로그인 시도...")
    try:
        # 환경 변수에서 토큰 가져오기 또는 로그인 건너뛰기
        import os
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("✅ Hugging Face 로그인 성공")
        else:
            print("⚠️ HF_TOKEN이 없습니다. 로그인 건너뜁니다.")
            print("   공개 모델만 사용 가능합니다.")
    except Exception as e:
        print(f"⚠️ Hugging Face 로그인 실패: {e}")
        print("   로그인 없이 계속 진행합니다.")

    # 1. 데이터셋 로드
    print("\n📂 데이터셋 로드")
    train_dataset = load_and_prepare_dataset("../data/processed/train_dataset.json")
    val_dataset = load_and_prepare_dataset("../data/processed/val_dataset.json")

    if train_dataset is None or val_dataset is None:
        print("❌ 데이터셋 로드 실패. prepare_data.py를 먼저 실행하세요.")
        return

    # 데이터 개수 제한 (3B 모델용으로 더 많이)
    max_train_samples = 10000  # 3B 모델이니까 더 많이!
    max_val_samples = 1000

    if len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
        print(f"🔢 학습 데이터를 {max_train_samples}개로 제한")

    if len(val_dataset) > max_val_samples:
        val_dataset = val_dataset.select(range(max_val_samples))
        print(f"🔢 검증 데이터를 {max_val_samples}개로 제한")

    print(f"📊 최종 데이터: 학습 {len(train_dataset)}개, 검증 {len(val_dataset)}개")

    # 2. 3B 모델 및 토크나이저 설정
    print("\n🤖 3B 모델 설정")
    use_4bit = torch.cuda.is_available()
    model, tokenizer = setup_model_and_tokenizer(use_4bit=use_4bit)

    # 3. LoRA 설정
    print("\n⚙️ LoRA 설정 (3B 최적화)")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    # 파라미터 정보
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 학습 가능한 파라미터: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 4. 학습 설정 (3B 모델용 긴 학습)
    print("\n🎯 학습 설정 (5 에폭, 충분한 학습)")
    training_args = setup_training_args(
        output_dir="../models/summary-3b-longrun",  # 새 폴더명
        num_epochs=5,  # 더 많은 에폭
        batch_size=1 if not torch.cuda.is_available() else 2
    )

    # 5. Formatting 함수 정의
    def formatting_func(example):
        return example["text"]

    # 6. Trainer 설정 (finetune.py와 동일한 방식)
    print("\n🏃 Trainer 설정")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        args=training_args,
        peft_config=None,  # 이미 LoRA가 적용된 모델이므로 None
    )

    # 7. 학습 시작
    print("\n🔥 학습 시작...")
    print("=" * 80)

    # 체크포인트가 있으면 재개
    checkpoint = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint = os.path.join(training_args.output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))
            print(f"♻️ 체크포인트에서 재개: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # 8. 모델 저장
    print("\n💾 최종 모델 저장")
    model_save_path = "../models/summary-3b-final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"✅ 3B 모델 저장 완료: {model_save_path}")

    # 9. 학습 결과 저장
    print("\n📊 학습 로그 저장")
    log_history = trainer.state.log_history

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/training_3b_log.json", 'w', encoding='utf-8') as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("🎉 Llama 3.2 3B 요약 모델 파인튜닝 완료!")
    print(f"📁 모델 위치: {model_save_path}")
    print(f"📈 학습 로그: {results_dir}/training_3b_log.json")
    print("=" * 80)

if __name__ == "__main__":
    main()