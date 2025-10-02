#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama 3.2 Korean Summarization Demo
한국어 문서 요약 모델 데모 스크립트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
import sys
import os

def setup_model(base_model_name="meta-llama/Llama-3.2-3B-Instruct",
               adapter_path="./summary-ai/models/summary-3b-final",
               use_4bit=True):
    """모델 및 토크나이저 설정"""

    print("🔄 모델 로딩 중...")
    print(f"베이스 모델: {base_model_name}")
    print(f"어댑터: {adapter_path}")

    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎮 사용 디바이스: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 모델 로드
    if use_4bit and device == "cuda":
        print("⚡ 4-bit 양자화 사용")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        print("🖥️ 일반 모드로 로드")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if device == "cpu":
            model = model.to(device)

    # LoRA 어댑터 적용
    if os.path.exists(adapter_path):
        print("🔧 LoRA 어댑터 적용 중...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ 파인튜닝된 모델 로드 완료")
    else:
        print("⚠️ 어댑터를 찾을 수 없습니다. 베이스 모델만 사용합니다.")

    model.config.use_cache = False

    return model, tokenizer

def create_prompt(text):
    """Llama 3.2 형식의 프롬프트 생성"""

    instruction = """다음 텍스트를 읽고 핵심 내용을 2-3개의 완전한 문장으로 요약하세요.
요약은 반드시 완성된 문장으로 작성하고, 원문에 없는 내용은 추가하지 마세요."""

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
텍스트: {text}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
요약: """

    return prompt

def summarize_text(model, tokenizer, text, max_new_tokens=200, temperature=0.3):
    """텍스트 요약 생성"""

    # 프롬프트 생성
    prompt = create_prompt(text)

    # 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("🤖 요약 생성 중...")

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 요약 부분만 추출
    if "요약: " in response:
        summary = response.split("요약: ")[-1].strip()
    else:
        summary = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    return summary

def interactive_demo(model, tokenizer):
    """대화형 데모"""

    print("\n" + "="*60)
    print("🚀 Llama 3.2 한국어 요약 모델 데모")
    print("="*60)
    print("요약할 텍스트를 입력하세요 (종료: 'quit' 또는 'exit')")
    print("-"*60)

    while True:
        print("\n📝 텍스트 입력:")
        text = input("> ")

        if text.lower() in ['quit', 'exit', '종료']:
            print("👋 데모를 종료합니다.")
            break

        if not text.strip():
            print("⚠️ 텍스트를 입력해주세요.")
            continue

        try:
            summary = summarize_text(model, tokenizer, text)

            print("\n📊 결과:")
            print(f"원문 길이: {len(text)}자")
            print(f"요약 길이: {len(summary)}자")
            print(f"압축률: {len(summary)/len(text)*100:.1f}%")
            print("\n✨ 요약:")
            print(f"{summary}")
            print("-"*60)

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

def batch_demo(model, tokenizer, texts):
    """배치 처리 데모"""

    print("\n📚 배치 요약 데모")
    print("-"*40)

    for i, text in enumerate(texts, 1):
        print(f"\n📄 문서 {i}:")
        print(f"원문: {text[:100]}..." if len(text) > 100 else f"원문: {text}")

        try:
            summary = summarize_text(model, tokenizer, text)
            print(f"요약: {summary}")
        except Exception as e:
            print(f"❌ 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description="Llama 3.2 Korean Summarization Demo")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                       help="베이스 모델 경로")
    parser.add_argument("--adapter", default="./summary-ai/models/summary-3b-final",
                       help="LoRA 어댑터 경로")
    parser.add_argument("--no-4bit", action="store_true",
                       help="4-bit 양자화 비활성화")
    parser.add_argument("--text", type=str,
                       help="요약할 텍스트 (직접 입력)")
    parser.add_argument("--file", type=str,
                       help="요약할 텍스트 파일 경로")
    parser.add_argument("--batch", action="store_true",
                       help="배치 처리 데모")

    args = parser.parse_args()

    # 모델 로드
    try:
        model, tokenizer = setup_model(
            base_model_name=args.model,
            adapter_path=args.adapter,
            use_4bit=not args.no_4bit
        )
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

    # 실행 모드 결정
    if args.text:
        # 단일 텍스트 요약
        print("\n📝 단일 텍스트 요약")
        summary = summarize_text(model, tokenizer, args.text)
        print(f"\n원문: {args.text}")
        print(f"요약: {summary}")

    elif args.file:
        # 파일에서 텍스트 읽어서 요약
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()

            print(f"\n📄 파일 요약: {args.file}")
            summary = summarize_text(model, tokenizer, text)
            print(f"요약: {summary}")

        except Exception as e:
            print(f"❌ 파일 읽기 실패: {e}")

    elif args.batch:
        # 배치 처리 데모
        sample_texts = [
            "정부는 내년부터 전기차 보조금을 현행 700만원에서 500만원으로 축소하기로 했다. 전기차 시장이 빠르게 성장하면서 보조금 부담이 커진 것이 주요 이유다.",
            "인공지능(AI) 기술의 발전으로 많은 산업 분야에서 자동화가 진행되고 있다. 특히 제조업과 서비스업에서 AI 도입이 활발하다.",
            "서울시는 대중교통 요금 인상을 검토하고 있다고 발표했다. 지하철과 버스 요금이 동시에 오를 가능성이 높다."
        ]
        batch_demo(model, tokenizer, sample_texts)

    else:
        # 대화형 모드 (기본)
        interactive_demo(model, tokenizer)

if __name__ == "__main__":
    main()