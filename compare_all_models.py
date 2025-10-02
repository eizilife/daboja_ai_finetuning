#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 모델 성능 비교
1. 베이스라인 (원본 Llama 3.2 3B)
2. 1차 파인튜닝 (기본 LoRA)
3. 최종 모델 (데이터 합성 + 최적화)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import time
import os
from datetime import datetime
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

class ModelComparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.results = {}

    def load_baseline_model(self):
        """베이스라인 모델 로드 (1B)"""
        print("🔄 베이스라인 모델 로딩 (1B)...")

        model_name = "meta-llama/Llama-3.2-1B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)

        self.models["baseline"] = model
        self.tokenizers["baseline"] = tokenizer
        print("✅ 베이스라인 모델 로드 완료")

    def load_first_finetuned_model(self, adapter_path="./lora_adapter"):
        """1차 파인튜닝 모델 로드 (1B + LoRA)"""
        print("🔄 1차 파인튜닝 모델 로딩 (1B + LoRA)...")

        if not os.path.exists(adapter_path):
            print(f"⚠️ 1차 파인튜닝 모델을 찾을 수 없습니다: {adapter_path}")
            return False

        # 1B 베이스 모델 새로 로드
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)

        # LoRA 어댑터 적용
        model = PeftModel.from_pretrained(model, adapter_path)

        self.models["first_finetuned"] = model
        self.tokenizers["first_finetuned"] = tokenizer
        print("✅ 1차 파인튜닝 모델 로드 완료")
        return True

    def load_final_model(self, adapter_path="./summary-ai/models/summary-3b-final"):
        """최종 모델 로드"""
        print("🔄 최종 모델 로딩...")

        if not os.path.exists(adapter_path):
            print(f"⚠️ 최종 모델을 찾을 수 없습니다: {adapter_path}")
            return False

        # GPU 메모리 확보를 위해 이전 모델들 정리
        print("🧹 GPU 메모리 정리 중...")
        if "baseline" in self.models:
            del self.models["baseline"]
        if "first_finetuned" in self.models:
            del self.models["first_finetuned"]
        torch.cuda.empty_cache()

        # 새로운 베이스 모델 로드
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if self.device == "cuda":
            # 간단한 4비트 설정으로 변경
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="cuda:0",  # 명시적으로 GPU 지정
                    torch_dtype=torch.float16
                )
            except Exception as e:
                print(f"⚠️ 4비트 로딩 실패, CPU 모드로 전환: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)

        # 최종 LoRA 어댑터 적용
        model = PeftModel.from_pretrained(model, adapter_path)

        self.models["final"] = model
        self.tokenizers["final"] = tokenizer
        print("✅ 최종 모델 로드 완료")
        return True

    def create_prompt(self, text, model_type="final"):
        """모델 타입에 따른 프롬프트 생성"""

        if model_type == "baseline":
            # 베이스라인은 간단한 프롬프트
            return f"""다음 텍스트를 요약해주세요:

텍스트: {text}

요약:"""

        else:
            # 파인튜닝된 모델들은 고급 프롬프트
            instruction = """다음 텍스트를 읽고 핵심 내용을 2-3개의 완전한 문장으로 요약하세요.
요약은 반드시 완성된 문장으로 작성하고, 원문에 없는 내용은 추가하지 마세요."""

            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
텍스트: {text}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
요약: """

    def generate_summary(self, model_key, text, max_new_tokens=200):
        """요약 생성"""

        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]

        prompt = self.create_prompt(text, model_key)

        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 생성 시작 시간
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 생성 시간 계산
        generation_time = time.time() - start_time

        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 요약 부분만 추출
        if model_key == "baseline":
            if "요약:" in response:
                summary = response.split("요약:")[-1].strip()
            else:
                summary = response.split(prompt)[-1].strip()
        else:
            if "요약: " in response:
                summary = response.split("요약: ")[-1].strip()
            else:
                summary = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

        # 토큰 수 계산
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(outputs[0]) - input_tokens
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0

        return {
            "summary": summary,
            "generation_time": generation_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second
        }

    def evaluate_models(self, test_texts):
        """모든 모델 평가"""

        print("\n" + "="*80)
        print("🔥 3단계 모델 성능 비교 시작")
        print("="*80)

        model_names = {
            "baseline": "🔧 베이스라인 (Llama 3.2 1B 원본)",
            "first_finetuned": "🎯 1차 파인튜닝 (1B + 기본 LoRA)",
            "final": "🚀 최종 모델 (3B + 데이터 합성 + 최적화)"
        }

        comparison_results = []

        for i, text in enumerate(test_texts, 1):
            print(f"\n📄 테스트 {i}/{len(test_texts)} - 원문 길이: {len(text)}자")
            if len(text) > 500:
                print(f"원문: {text[:200]}... (긴 문서 - {len(text)}자)")
            else:
                print(f"원문: {text}")
            print("-" * 60)

            test_result = {
                "test_id": i,
                "original_text": text,
                "original_length": len(text)
            }

            # 각 모델로 요약 생성
            for model_key in ["baseline", "first_finetuned", "final"]:
                if model_key not in self.models:
                    print(f"⚠️ {model_names[model_key]} 모델이 로드되지 않았습니다.")
                    continue

                print(f"\n{model_names[model_key]} 처리 중...")

                try:
                    result = self.generate_summary(model_key, text)

                    test_result[f"{model_key}_summary"] = result["summary"]
                    test_result[f"{model_key}_length"] = len(result["summary"])
                    test_result[f"{model_key}_time"] = result["generation_time"]
                    test_result[f"{model_key}_tokens_per_sec"] = result["tokens_per_second"]
                    test_result[f"{model_key}_compression_ratio"] = len(result["summary"]) / len(text) * 100

                    print(f"✅ 완료 ({result['generation_time']:.2f}초, {result['tokens_per_second']:.1f} tok/s)")
                    print(f"📝 생성된 요약:")
                    print(f"   {result['summary']}")
                    print(f"📊 압축률: {len(result['summary'])}/{len(text)} = {len(result['summary'])/len(text)*100:.1f}%")

                except Exception as e:
                    print(f"❌ 오류: {e}")
                    test_result[f"{model_key}_summary"] = "오류 발생"
                    test_result[f"{model_key}_length"] = 0
                    test_result[f"{model_key}_time"] = 0
                    test_result[f"{model_key}_tokens_per_sec"] = 0
                    test_result[f"{model_key}_compression_ratio"] = 0

            # 이 테스트에 대한 요약 비교
            print(f"\n🔍 테스트 {i} 요약 품질 비교:")
            print("="*80)
            print(f"📄 원문 ({len(text)}자):")
            print(f"   {text}")
            print("\n📝 요약 결과:")

            for model_key in ["baseline", "first_finetuned", "final"]:
                if f"{model_key}_summary" in test_result:
                    model_name = {
                        "baseline": "🔧 베이스라인 (1B)",
                        "first_finetuned": "🎯 1차 파인튜닝 (1B+LoRA)",
                        "final": "🚀 최종 모델 (3B+LoRA)"
                    }[model_key]

                    summary = test_result[f"{model_key}_summary"]
                    length = test_result[f"{model_key}_length"]
                    time_taken = test_result[f"{model_key}_time"]
                    speed = test_result[f"{model_key}_tokens_per_sec"]
                    compression = test_result[f"{model_key}_compression_ratio"]

                    print(f"\n{model_name}:")
                    print(f"   📝 전체 요약:")
                    print(f"   {summary}")
                    print(f"   📊 통계: {length}자 | {time_taken:.2f}초 | {speed:.1f} tok/s | 압축률 {compression:.1f}%")

            print("\n" + "="*80)

            comparison_results.append(test_result)

        return comparison_results

    def analyze_results(self, results):
        """결과 분석 및 시각화"""

        print("\n" + "="*80)
        print("📊 성능 분석 결과")
        print("="*80)

        # 평균 성능 계산
        metrics = {}
        model_keys = ["baseline", "first_finetuned", "final"]
        model_names = {
            "baseline": "베이스라인",
            "first_finetuned": "1차 파인튜닝",
            "final": "최종 모델"
        }

        for model_key in model_keys:
            if f"{model_key}_time" not in results[0]:
                continue

            avg_time = sum(r[f"{model_key}_time"] for r in results) / len(results)
            avg_tokens_per_sec = sum(r[f"{model_key}_tokens_per_sec"] for r in results) / len(results)
            avg_length = sum(r[f"{model_key}_length"] for r in results) / len(results)
            avg_compression = sum(r[f"{model_key}_compression_ratio"] for r in results) / len(results)

            metrics[model_key] = {
                "name": model_names[model_key],
                "avg_generation_time": avg_time,
                "avg_tokens_per_sec": avg_tokens_per_sec,
                "avg_summary_length": avg_length,
                "avg_compression_ratio": avg_compression
            }

        # 결과 출력
        print("\n📈 평균 성능 비교:")
        print(f"{'모델':<15} {'생성시간(초)':<12} {'속도(tok/s)':<12} {'요약길이':<10} {'압축률(%)':<10}")
        print("-" * 70)

        for model_key, metric in metrics.items():
            print(f"{metric['name']:<15} "
                  f"{metric['avg_generation_time']:<12.2f} "
                  f"{metric['avg_tokens_per_sec']:<12.1f} "
                  f"{metric['avg_summary_length']:<10.0f} "
                  f"{metric['avg_compression_ratio']:<10.1f}")

        # 상세 개선율 계산
        if "baseline" in metrics and "final" in metrics:
            print("\n🚀 최종 모델 개선율 (vs 베이스라인):")
            speed_improvement = (metrics["final"]["avg_tokens_per_sec"] / metrics["baseline"]["avg_tokens_per_sec"] - 1) * 100
            time_improvement = (1 - metrics["final"]["avg_generation_time"] / metrics["baseline"]["avg_generation_time"]) * 100
            length_change = (metrics["final"]["avg_summary_length"] / metrics["baseline"]["avg_summary_length"] - 1) * 100
            compression_change = metrics["final"]["avg_compression_ratio"] - metrics["baseline"]["avg_compression_ratio"]

            print(f"⚡ 생성 속도: {speed_improvement:+.1f}%")
            print(f"⏱️ 생성 시간: {time_improvement:+.1f}%")
            print(f"📏 요약 길이: {length_change:+.1f}%")
            print(f"🗜️ 압축률 변화: {compression_change:+.1f}%p")

        if "first_finetuned" in metrics and "final" in metrics:
            print("\n📈 최종 vs 1차 파인튜닝:")
            speed_vs_first = (metrics["final"]["avg_tokens_per_sec"] / metrics["first_finetuned"]["avg_tokens_per_sec"] - 1) * 100
            time_vs_first = (1 - metrics["final"]["avg_generation_time"] / metrics["first_finetuned"]["avg_generation_time"]) * 100

            print(f"⚡ 속도 추가 개선: {speed_vs_first:+.1f}%")
            print(f"⏱️ 시간 추가 단축: {time_vs_first:+.1f}%")

        # 요약 품질 평가
        print("\n📝 요약 품질 분석:")
        for i, result in enumerate(results, 1):
            print(f"\n테스트 {i} 요약 길이 비교:")
            original_len = result["original_length"]

            for model_key in ["baseline", "first_finetuned", "final"]:
                if f"{model_key}_summary" in result:
                    model_name = {
                        "baseline": "베이스라인",
                        "first_finetuned": "1차 파인튜닝",
                        "final": "최종 모델"
                    }[model_key]

                    summary_len = result[f"{model_key}_length"]
                    compression = result[f"{model_key}_compression_ratio"]
                    print(f"  {model_name}: {original_len}자 → {summary_len}자 ({compression:.1f}%)")

        return metrics

    def save_results(self, results, metrics):
        """결과 저장"""

        # 상세 결과 저장
        with open("model_comparison_detailed.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "detailed_results": results,
                "summary_metrics": metrics,
                "device": self.device,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }, f, ensure_ascii=False, indent=2)

        # CSV로도 저장 (pandas 있는 경우만)
        if HAS_PANDAS:
            df = pd.DataFrame(results)
            df.to_csv("model_comparison_results.csv", index=False, encoding="utf-8")
            print("\n💾 결과 저장 완료:")
            print("- model_comparison_detailed.json (상세 결과)")
            print("- model_comparison_results.csv (CSV 형식)")
        else:
            print("\n💾 결과 저장 완료:")
            print("- model_comparison_detailed.json (상세 결과)")
            print("📝 pandas가 없어 CSV는 생략됩니다.")

def main():
    # 긴 문서 테스트 케이스 추가 (5000자 이상)
    long_document = """한국의 정보기술(IT) 산업은 2025년 들어 급격한 변화의 시대를 맞고 있다. 반도체, 디스플레이, 통신장비 등 전통적인 강점 분야에서의 글로벌 경쟁력 유지와 함께, 인공지능(AI), 메타버스, 블록체인 등 신기술 분야에서의 새로운 도약을 위한 노력이 지속되고 있다.

먼저 반도체 산업을 살펴보면, 삼성전자와 SK하이닉스를 중심으로 한 메모리 반도체 분야에서 여전히 세계 1위의 지위를 유지하고 있다. 특히 DDR5 DRAM과 HBM(High Bandwidth Memory) 시장에서의 기술 우위는 AI 시대의 핵심 경쟁력으로 작용하고 있다. 삼성전자는 올해 상반기에만 HBM3E 양산을 시작하며 엔비디아, AMD 등 글로벌 AI 칩 업체들과의 파트너십을 강화했다. SK하이닉스 역시 HBM4 개발에 박차를 가하고 있으며, 2026년 양산을 목표로 하고 있다.

시스템 반도체 분야에서는 삼성전자가 3나노 공정 기술에서 TSMC와의 격차를 줄이기 위해 대규모 투자를 지속하고 있다. 특히 GAA(Gate-All-Around) 기술을 활용한 3나노 공정의 수율 개선과 함께, 2나노 공정 개발에도 속도를 내고 있다. 이와 함께 파운드리 사업 확장을 위해 미국 텍사스와 평택에 신규 생산 라인 구축을 진행 중이다.

디스플레이 산업에서는 삼성디스플레이와 LG디스플레이가 각각 OLED와 QLED 기술을 중심으로 글로벌 시장을 선도하고 있다. 특히 폴더블 디스플레이 분야에서 삼성디스플레이는 압도적인 시장 점유율을 보이고 있으며, 갤럭시 폴드와 플립 시리즈의 성공과 함께 중국 스마트폰 제조사들로의 공급도 확대하고 있다. LG디스플레이는 대형 OLED TV 패널 시장에서의 독점적 지위를 바탕으로 투명 OLED, 롤러블 OLED 등 차세대 디스플레이 기술 개발에 집중하고 있다.

통신장비 분야에서는 삼성전자와 LG전자가 5G와 6G 기술 개발에 적극적으로 나서고 있다. 삼성전자는 미국과 일본 등지에서 5G 기지국 장비 공급을 확대하고 있으며, 특히 오픈랜(Open RAN) 기술 개발에 선도적 역할을 하고 있다. 6G 기술 개발을 위해서는 정부와 함께 'K-6G 프로젝트'를 추진하고 있으며, 2030년 상용화를 목표로 하고 있다.

소프트웨어 산업에서는 네이버, 카카오, 엔씨소프트 등이 AI 기술 개발과 글로벌 진출에 박차를 가하고 있다. 네이버는 하이퍼클로바X를 통해 생성형 AI 시장에 본격 진출했으며, 일본과 동남아시아 시장에서의 영향력 확대를 위해 현지 기업들과의 파트너십을 강화하고 있다. 카카오는 카카오브레인을 중심으로 한 AI 기술 개발과 함께, 모빌리티, 핀테크, 엔터테인먼트 등 다양한 분야에서의 디지털 전환을 이끌고 있다.

게임 산업에서는 엔씨소프트, 넷마블, 크래프톤 등이 글로벌 시장에서 한국 게임의 위상을 높이고 있다. 특히 'PUBG', '리니지', '블레이드 앤 소울' 등의 IP를 활용한 모바일 게임들이 전 세계적으로 큰 인기를 얻고 있으며, 메타버스 기술을 접목한 새로운 형태의 게임 개발에도 주력하고 있다.

핀테크 분야에서는 토스, 카카오페이, 네이버페이 등이 금융 서비스의 디지털 혁신을 이끌고 있다. 특히 토스는 간편송금 서비스를 시작으로 종합 금융 플랫폼으로 성장했으며, 최근에는 해외 진출과 함께 암호화폐 거래 서비스도 본격화하고 있다. 카카오페이와 네이버페이 역시 각각의 생태계를 바탕으로 결제, 투자, 대출 등 다양한 금융 서비스를 제공하고 있다.

E-커머스 분야에서는 쿠팡, 11번가, 지마켓 등이 치열한 경쟁을 벌이고 있다. 특히 쿠팡은 로켓배송을 통한 초고속 배송 서비스로 시장을 선도하고 있으며, 최근에는 일본과 대만 시장 진출을 통해 아시아 지역에서의 영향력 확대를 도모하고 있다. 또한 쿠팡이츠, 쿠팡플레이 등 다양한 서비스를 통해 종합 생활 플랫폼으로의 전환을 추진하고 있다.

바이오헬스케어 IT 분야에서는 의료진단 AI, 디지털 치료제, 헬스케어 플랫폼 등이 주목받고 있다. 뷰노, 루닛 등은 의료영상 분석 AI 기술을 바탕으로 글로벌 시장 진출에 성공했으며, 국내외 병원들과의 협력을 통해 기술 검증과 상용화를 진행하고 있다.

정부는 이러한 IT 산업의 글로벌 경쟁력 강화를 위해 'K-디지털 뉴딜'과 함께 '디지털 플랫폼 정부' 구축에 나서고 있다. 특히 AI, 빅데이터, 클라우드 등 핵심 기술 분야에 대한 투자 확대와 함께, 관련 인재 양성을 위한 교육 프로그램도 확대하고 있다. 또한 규제 샌드박스 제도를 통해 신기술의 상용화를 지원하고 있으며, 데이터 경제 활성화를 위한 법적 기반도 마련하고 있다.

그러나 몇 가지 도전 과제도 존재한다. 첫째, 중국과의 기술 격차 축소로 인한 경쟁 심화다. 특히 디스플레이와 배터리 분야에서 중국 기업들의 급성장이 한국 기업들의 시장 지위를 위협하고 있다. 둘째, 미중 기술 패권 경쟁 속에서 글로벌 공급망의 불안정성이 증가하고 있다. 셋째, AI와 자동화 기술의 발전으로 인한 일자리 대체 우려와 함께, 관련 인재의 부족 문제도 지속되고 있다.

이에 대응하기 위해 정부와 기업들은 다각적인 노력을 기울이고 있다. 정부는 'K-반도체 벨트' 조성을 통한 반도체 생태계 강화와 함께, AI 국가전략을 수립하여 관련 기술 개발과 인재 양성에 집중하고 있다. 기업들은 ESG 경영과 함께 지속가능한 기술 개발에 투자를 확대하고 있으며, 스타트업과의 협력을 통한 개방형 혁신에도 적극 나서고 있다.

결론적으로, 한국의 IT 산업은 전통적인 강점 분야에서의 기술 리더십을 유지하면서도 새로운 성장 동력 발굴을 위한 노력을 지속하고 있다. 글로벌 경쟁이 더욱 치열해지는 상황에서 기술 혁신과 인재 육성, 그리고 지속가능한 성장을 위한 전략적 접근이 향후 한국 IT 산업의 성공을 좌우할 것으로 전망된다."""

    # 테스트 텍스트들 (기존 짧은 텍스트 + 긴 문서)
    test_texts = [
        """정부는 내년부터 전기차 보조금을 현행 700만원에서 500만원으로 축소하기로 했다.
        전기차 시장이 빠르게 성장하면서 보조금 부담이 커진 것이 주요 이유다.
        하지만 전기차 업계는 보조금 축소로 판매에 타격을 받을 것이라고 우려하고 있다.""",

        """인공지능(AI) 기술의 발전으로 많은 산업 분야에서 자동화가 진행되고 있다.
        특히 제조업과 서비스업에서 AI 도입이 활발하다.
        하지만 AI로 인한 일자리 감소 우려도 커지고 있어 정부 차원의 대책 마련이 시급하다는 지적이다.""",

        """서울시는 대중교통 요금 인상을 검토하고 있다고 발표했다.
        지하철과 버스 요금이 동시에 오를 가능성이 높다.
        코로나19 이후 승객 수가 줄어들면서 적자가 지속되고 있는 것이 주요 원인이다.""",

        # 5000자 이상의 긴 문서 추가
        long_document
    ]

    print("순차적 모델 비교 시작 (메모리 절약 모드)")
    print("="*80)

    # 모델 비교기 초기화
    comparator = ModelComparator()
    all_results = []

    # 1단계: 베이스라인 테스트
    print("\n🔧 1단계: 베이스라인 모델 테스트")
    comparator.load_baseline_model()
    baseline_results = []
    for i, text in enumerate(test_texts, 1):
        print(f"\n📄 테스트 {i}: 베이스라인")
        result = comparator.generate_summary("baseline", text)
        baseline_results.append(result)
        print(f"✅ 완료: {result['summary'][:100]}...")

    # 2단계: 1차 파인튜닝 테스트
    print("\n🎯 2단계: 1차 파인튜닝 모델 테스트")
    comparator.load_first_finetuned_model()
    first_results = []
    for i, text in enumerate(test_texts, 1):
        print(f"\n📄 테스트 {i}: 1차 파인튜닝")
        result = comparator.generate_summary("first_finetuned", text)
        first_results.append(result)
        print(f"✅ 완료: {result['summary'][:100]}...")

    # 3단계: 최종 모델 테스트
    print("\n🚀 3단계: 최종 모델 테스트")
    comparator.load_final_model()
    final_results = []
    for i, text in enumerate(test_texts, 1):
        print(f"\n📄 테스트 {i}: 최종 모델")
        result = comparator.generate_summary("final", text)
        final_results.append(result)
        print(f"✅ 완료: {result['summary'][:100]}...")

    # 결과 종합
    print("\n" + "="*80)
    print("📊 최종 비교 결과")
    print("="*80)

    for i, text in enumerate(test_texts):
        print(f"\n📄 테스트 {i+1} ({len(text)}자):")
        print(f"원문: {text[:100]}...")
        print()

        models = [
            ("베이스라인 (1B)", baseline_results[i]),
            ("1차 파인튜닝 (1B+LoRA)", first_results[i]),
            ("최종 모델 (3B+LoRA)", final_results[i])
        ]

        for name, result in models:
            print(f"{name}:")
            print(f"  📝 요약: {result['summary']}")
            print(f"  📊 {len(result['summary'])}자 | {result['generation_time']:.2f}초 | {result['tokens_per_second']:.1f} tok/s")
            print()

    print("🎉 모든 비교 완료!")

if __name__ == "__main__":
    main()