#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
요약 모델 테스트 스크립트
파인튜닝된 모델을 쉽게 테스트할 수 있는 인터랙티브 스크립트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys

def load_model(model_path="../models/latest-summary-model", use_gpu=True):
    """모델과 토크나이저 로드"""
    print("🔄 모델 로딩 중...")

    # 디바이스 설정
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"📱 사용 디바이스: {device}")

    # 베이스 모델 로드
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )

        # LoRA 가중치 로드 (파인튜닝된 모델)
        if os.path.exists(model_path):
            print(f"✅ 파인튜닝 모델 로드: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("⚠️ 파인튜닝 모델이 없습니다. 베이스 모델을 사용합니다.")
            model = base_model

        if device == "cpu":
            model = model.to(device)

        model.eval()
        return model, tokenizer, device

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

def summarize(text, model, tokenizer, device, max_length=150, temperature=0.7):
    """텍스트 요약 생성"""
    # 프롬프트 템플릿 - 완성된 문장 생성을 강조
    prompt = f"""다음 텍스트를 읽고 핵심 내용을 2-3개의 완전한 문장으로 요약하세요.
반드시 완성된 문장으로 끝내고, 원문에 있는 내용만 포함하세요.

텍스트: {text}

요약:"""

    # 토큰화
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    # 요약 생성 - 더 안정적인 생성을 위한 파라미터 조정
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            min_new_tokens=50,  # 최소 길이 증가
            temperature=0.6,  # 더 보수적인 온도
            top_p=0.85,  # top_p 감소
            top_k=50,  # top_k 추가
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # 반복 패널티 증가
            no_repeat_ngram_size=3  # n-gram 반복 방지
        )

    # 디코딩
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 요약 부분만 추출
    if "요약:" in full_output:
        summary = full_output.split("요약:")[-1].strip()
    else:
        summary = full_output[len(prompt):].strip()

    return summary

def test_with_examples(model, tokenizer, device):
    """예제 텍스트로 테스트"""
    examples = [
        """
        인공지능 기술의 발전으로 다양한 산업 분야에서 혁신이 일어나고 있습니다.
        특히 자연어 처리 분야에서는 GPT와 같은 대규모 언어 모델이 등장하면서
        기계 번역, 텍스트 요약, 질의응답 시스템 등이 크게 개선되었습니다.
        이러한 기술들은 업무 효율성을 높이고 사용자 경험을 향상시키는 데
        중요한 역할을 하고 있으며, 앞으로도 지속적인 발전이 예상됩니다.
        """,

        """
        기후 변화는 21세기 인류가 직면한 가장 심각한 문제 중 하나입니다.
        지구 온난화로 인한 해수면 상승, 극단적인 날씨 현상, 생태계 파괴 등이
        전 세계적으로 나타나고 있습니다. 이에 대응하기 위해 각국 정부는
        탄소 중립 목표를 설정하고 재생 에너지 개발에 투자를 확대하고 있습니다.
        개인 차원에서도 에너지 절약과 친환경 생활 실천이 필요합니다.
        """
    ]

    print("\n" + "="*60)
    print("📝 예제 텍스트 테스트")
    print("="*60)

    for i, text in enumerate(examples, 1):
        print(f"\n[예제 {i}]")
        print(f"원문 ({len(text)} 글자):")
        print(text[:200] + "..." if len(text) > 200 else text)

        summary = summarize(text, model, tokenizer, device)
        print(f"\n요약 ({len(summary)} 글자):")
        print(summary)
        print("-"*40)

def interactive_test(model, tokenizer, device):
    """대화형 테스트 모드"""
    print("\n" + "="*60)
    print("💬 대화형 테스트 모드")
    print("="*60)
    print("요약할 텍스트를 입력하세요. (종료: 'quit' 또는 'exit')")
    print("빈 줄을 두 번 입력하면 요약을 시작합니다.\n")

    while True:
        print("\n" + "▶ 텍스트 입력 (여러 줄 가능):")

        lines = []
        empty_line_count = 0

        while empty_line_count < 2:
            line = input()

            if line.lower() in ['quit', 'exit', '종료']:
                print("👋 테스트를 종료합니다.")
                return

            if line == "":
                empty_line_count += 1
            else:
                empty_line_count = 0
                lines.append(line)

        if not lines:
            continue

        text = "\n".join(lines)

        print("\n⏳ 요약 생성 중...")
        summary = summarize(text, model, tokenizer, device)

        print("\n📋 요약 결과:")
        print("-"*40)
        print(summary)
        print("-"*40)
        print(f"원문: {len(text)} 글자 → 요약: {len(summary)} 글자 (압축률: {len(summary)/len(text)*100:.1f}%)")

def test_from_file(model, tokenizer, device, file_path):
    """파일에서 텍스트를 읽어 요약"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"\n📄 파일: {file_path}")
        print(f"원문 길이: {len(text)} 글자")

        if len(text) > 2000:
            print("⚠️ 텍스트가 너무 깁니다. 처음 2000자만 사용합니다.")
            text = text[:2000]

        print("\n⏳ 요약 생성 중...")
        summary = summarize(text, model, tokenizer, device, max_length=200)

        print("\n📋 요약 결과:")
        print("-"*40)
        print(summary)
        print("-"*40)

        # 요약 결과 저장
        output_path = file_path.rsplit('.', 1)[0] + "_summary.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"💾 요약 저장: {output_path}")

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def main():
    """메인 함수"""
    print("="*60)
    print("🚀 요약 모델 테스트 프로그램")
    print("="*60)

    # 모델 경로 옵션
    model_paths = {
        "1": "../models/latest-summary-model",
        "2": "../models/summary-lora/checkpoint-939",
        "3": "../models/summary-lora/checkpoint-800",
        "4": "../models/summary-lora/checkpoint-600",
        "5": "../models/summary-3b-enhanced",  # 3B 모델 추가
        "6": "../models/summary-3b-final",  # 3B 최종 모델
        "7": "custom"
    }

    print("\n모델을 선택하세요:")
    print("1. latest-summary-model (기본)")
    print("2. checkpoint-939 (1B 최신)")
    print("3. checkpoint-800 (1B)")
    print("4. checkpoint-600 (1B)")
    print("5. summary-3b-enhanced (3B 향상)")
    print("6. summary-3b-final (3B 최종)")
    print("7. 직접 입력")

    choice = input("\n선택 (1-7): ").strip() or "1"

    if choice == "7":
        model_path = input("모델 경로 입력: ").strip()
    else:
        model_path = model_paths.get(choice, model_paths["1"])

    # GPU 사용 여부
    use_gpu = input("\nGPU를 사용하시겠습니까? (y/n, 기본: y): ").strip().lower() != 'n'

    # 모델 로드
    model, tokenizer, device = load_model(model_path, use_gpu)
    print("✅ 모델 로드 완료!\n")

    # 테스트 모드 선택
    while True:
        print("\n테스트 모드를 선택하세요:")
        print("1. 예제 텍스트로 테스트")
        print("2. 직접 입력하여 테스트")
        print("3. 파일에서 읽어 테스트")
        print("4. 종료")

        mode = input("\n선택 (1-4): ").strip()

        if mode == "1":
            test_with_examples(model, tokenizer, device)
        elif mode == "2":
            interactive_test(model, tokenizer, device)
        elif mode == "3":
            file_path = input("파일 경로 입력: ").strip()
            if file_path:
                test_from_file(model, tokenizer, device, file_path)
        elif mode == "4":
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()