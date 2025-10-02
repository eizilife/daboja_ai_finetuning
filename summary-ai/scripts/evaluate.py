#!/usr/bin/env python3
"""
보고서 요약 모델 평가 및 테스트 스크립트
"""

import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
from datetime import datetime

def load_base_model():
    """기본 모델 로드"""
    print("🔄 기본 모델 로드 중...")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    print("✅ 기본 모델 로드 완료")
    return model, tokenizer

def load_finetuned_model(base_model, adapter_path="report-summarization/models/summary-lora-adapter"):
    """파인튜닝된 모델 로드"""
    if not os.path.exists(adapter_path):
        print(f"❌ 어댑터 경로가 존재하지 않습니다: {adapter_path}")
        return None

    print("🔄 파인튜닝된 모델 로드 중...")
    try:
        finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ 파인튜닝된 모델 로드 완료")
        return finetuned_model
    except Exception as e:
        print(f"❌ 파인튜닝된 모델 로드 실패: {str(e)}")
        return None

def create_summary_prompt(text: str) -> str:
    """요약 프롬프트 생성"""
    return f"""### Instruction:
다음 문서를 읽고 핵심 내용을 요약해주세요.

### Input:
{text}

### Response:"""

def generate_summary(model, tokenizer, text: str, max_tokens: int = 300) -> str:
    """텍스트 요약 생성"""
    prompt = create_summary_prompt(text)

    # 토큰화
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500,
        padding=True
    )

    # GPU로 이동 (가능한 경우)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # 응답 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 프롬프트 제거 (응답만 추출)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response

def load_test_examples():
    """테스트 예제들 로드"""

    # 삼성전자 예제 (제공받은 텍스트)
    samsung_text = """당사는 TV, 냉장고, 세탁기, 에어컨, 스마트폰 등 완제품과 DRAM, NAND Flash, 모바일AP 등 반도체 부품 및 스마트폰용 OLED 패널 등을 생산ㆍ판매하고 있습니다. 또한, Harman에서는 디지털 콕핏(Digital Cockpit), 카오디오 등 전장제품과 포터블/사운드바 스피커 등 컨슈머 오디오 제품 등을 개발, 생산, 판매하고 있습니다.

2025년 반기 매출은 DX 부문이 95조 2,874억원(62%), DS 부문이 53조 61억원(34.5%)이며, SDC가 12조 2,469억원(8%), Harman은 7조 2,494억원(4.7%)입니다.

2025년 반기 TV의 평균 판매가격은 전년 연간평균 대비 약 4% 하락하였으며, 스마트폰은 전년 연간평균 대비 약 1% 상승하였습니다. 그리고 메모리 평균 판매가격은 전년 연간평균 대비 약 3% 하락하였으며, 스마트폰용 OLED 패널은 약 12% 하락하였습니다. 한편 디지털 콕핏의 평균 판매가격은 전년 연간평균 대비 약 2% 하락하였습니다."""

    test_examples = [
        {
            "title": "삼성전자 2025년 반기 실적",
            "text": samsung_text
        },
        {
            "title": "기업 분기 실적 보고서",
            "text": """ABC 기업이 2025년 1분기 연결기준 매출액 25조 1,200억원, 영업이익 3조 2,100억원을 기록했다고 발표했습니다. 이는 전년 동기 대비 매출액 18.5% 증가, 영업이익 24.3% 증가한 수치입니다.

주요 사업부문별로는 전자사업부가 매출 15조원(전체의 60%)으로 가장 큰 비중을 차지했으며, 화학사업부 6조원(24%), 바이오사업부 4조 1,200억원(16%) 순이었습니다.

지역별로는 국내 매출이 45%, 중국 25%, 북미 20%, 기타 지역 10%를 기록했습니다. 특히 중국 시장에서의 매출이 전년 동기 대비 35% 증가하며 성장을 견인했습니다.

영업이익률은 12.8%로 전년 동기 10.2% 대비 2.6%p 개선되었으며, 이는 원가 절감과 고부가가치 제품 판매 확대에 따른 것으로 분석됩니다."""
        },
        {
            "title": "반도체 산업 동향 보고서",
            "text": """2025년 글로벌 반도체 시장은 AI 붐과 데이터센터 수요 급증에 힘입어 사상 최대 규모를 기록할 것으로 전망됩니다. 시장조사기관 가트너는 올해 글로벌 반도체 매출이 전년 대비 16.8% 증가한 7,560억 달러에 달할 것으로 예측했습니다.

특히 메모리 반도체 시장이 두드러진 성장세를 보이고 있습니다. DDR5 DRAM과 고대역폭 메모리(HBM) 수요가 폭증하면서 메모리 반도체 가격이 상승세를 지속하고 있습니다. 삼성전자, SK하이닉스 등 한국 기업들이 이 분야에서 시장을 주도하고 있습니다.

한편 시스템 반도체 부문에서는 TSMC가 3나노 공정 양산을 본격화하며 기술 격차를 더욱 벌리고 있습니다. 애플, 엔비디아 등 주요 고객사들의 차세대 칩 수요가 몰리면서 TSMC의 시장 지배력이 더욱 강화될 전망입니다.

지정학적 리스크도 업계의 주요 변수로 작용하고 있습니다. 미중 기술 패권 경쟁이 심화되면서 글로벌 반도체 공급망 재편이 가속화되고 있으며, 각국의 반도체 자급자족 정책도 시장 구조 변화를 이끌고 있습니다."""
        }
    ]

    return test_examples

def compare_models(base_model, finetuned_model, tokenizer, test_examples):
    """기본 모델과 파인튜닝된 모델 비교"""
    print("\n" + "="*100)
    print("📊 모델 성능 비교 결과")
    print("="*100)

    comparison_results = []

    for i, example in enumerate(test_examples, 1):
        print(f"\n🔸 테스트 {i}: {example['title']}")
        print("-" * 60)

        print("📄 원본 텍스트:")
        print(example['text'][:200] + "..." if len(example['text']) > 200 else example['text'])

        # 기본 모델 요약
        print("\n🤖 기본 모델 요약:")
        base_summary = generate_summary(base_model, tokenizer, example['text'])
        print(base_summary)

        # 파인튜닝된 모델 요약 (있는 경우)
        finetuned_summary = ""
        if finetuned_model:
            print("\n🎯 파인튜닝된 모델 요약:")
            finetuned_summary = generate_summary(finetuned_model, tokenizer, example['text'])
            print(finetuned_summary)

        # 결과 저장
        comparison_results.append({
            "title": example['title'],
            "original_text": example['text'],
            "base_model_summary": base_summary,
            "finetuned_model_summary": finetuned_summary
        })

        print("\n" + "-" * 60)

    return comparison_results

def save_evaluation_results(results, output_path="report-summarization/results/evaluation_results.json"):
    """평가 결과 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    eval_data = {
        "evaluation_date": datetime.now().isoformat(),
        "num_tests": len(results),
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 평가 결과 저장: {output_path}")

def main():
    """메인 실행 함수"""
    print("🚀 보고서 요약 모델 평가 시작")
    print("=" * 80)

    try:
        # Hugging Face 로그인
        # 환경변수에서 토큰 가져오기
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        # 기본 모델 로드
        base_model, tokenizer = load_base_model()

        # 파인튜닝된 모델 로드 시도
        finetuned_model = load_finetuned_model(base_model)

        if not finetuned_model:
            print("⚠️ 파인튜닝된 모델을 찾을 수 없습니다. 기본 모델만 테스트합니다.")

        # 테스트 예제 로드
        test_examples = load_test_examples()
        print(f"\n📋 {len(test_examples)}개 테스트 예제 준비 완료")

        # 모델 비교
        results = compare_models(base_model, finetuned_model, tokenizer, test_examples)

        # 결과 저장
        save_evaluation_results(results)

        print("\n✅ 평가 완료!")

        # 요약 통계
        print("\n📈 평가 요약:")
        print(f"   - 테스트 케이스: {len(results)}개")
        print(f"   - 기본 모델 평균 요약 길이: {sum(len(r['base_model_summary']) for r in results) / len(results):.0f} 문자")
        if finetuned_model:
            print(f"   - 파인튜닝 모델 평균 요약 길이: {sum(len(r['finetuned_model_summary']) for r in results) / len(results):.0f} 문자")

    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()