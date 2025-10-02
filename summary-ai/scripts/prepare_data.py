#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIHub 문서 요약 데이터셋 전처리 스크립트
데이터셋: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=97
"""

import sys
import os

# Windows 환경에서 UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import os
import glob
from typing import List, Dict, Tuple

def extract_zip_files(data_dir: str):
    """압축 파일들 해제"""
    import zipfile

    print("📦 압축 파일 해제 중...")

    # .zip.part0 파일들 찾기
    zip_files = glob.glob(os.path.join(data_dir, "**/*.zip.part0"), recursive=True)

    for zip_file in zip_files:
        try:
            # .part0 제거하여 실제 zip 파일명 생성
            actual_zip = zip_file.replace('.part0', '')

            # .part0 파일을 실제 zip 파일로 이름 변경
            os.rename(zip_file, actual_zip)

            # 압축 해제
            extract_dir = os.path.dirname(actual_zip)
            with zipfile.ZipFile(actual_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                print(f"✅ 압축 해제 완료: {actual_zip}")

        except Exception as e:
            print(f"❌ 압축 해제 실패 {zip_file}: {str(e)}")

def load_aihub_summary_data(data_dir: str) -> List[Dict]:
    """
    AIHub 문서 요약 데이터 로드

    Args:
        data_dir: 다운로드받은 데이터 디렉토리

    Returns:
        처리된 데이터 리스트
    """
    print("📁 AIHub 데이터 로드 중...")

    # 먼저 압축 파일들 해제
    extract_zip_files(data_dir)

    all_data = []

    # JSON 파일들 찾기
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)

    if not json_files:
        print(f"❌ {data_dir}에서 JSON 파일을 찾을 수 없습니다.")
        print("데이터 구조를 확인하세요:")
        print("report-summarization/data/")
        print("  ├── Train_사설_data/")
        print("  ├── Train_신문기사_data/")
        print("  ├── Valid_사설_data/")
        print("  └── Valid_신문기사_data/")
        return []

    for json_file in json_files:
        print(f"📄 처리 중: {json_file}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # AIHub 데이터 구조에 따라 파싱
            if isinstance(data, dict):
                if 'documents' in data:
                    all_data.extend(data['documents'])
                elif 'data' in data:
                    all_data.extend(data['data'])
                else:
                    all_data.append(data)
            elif isinstance(data, list):
                all_data.extend(data)

        except Exception as e:
            print(f"❌ 파일 로드 실패 {json_file}: {str(e)}")
            continue

    print(f"✅ 총 {len(all_data)}개 데이터 로드 완료")
    return all_data

def combine_short_documents(data_list: List[Dict], min_length: int = 2000) -> List[Dict]:
    """
    짧은 문서들을 합쳐서 더 긴 문서 생성

    Args:
        data_list: 처리된 데이터 리스트
        min_length: 최소 문서 길이

    Returns:
        합쳐진 데이터 리스트
    """
    print("🔗 짧은 문서들 결합 중...")

    combined_data = []
    temp_docs = []
    temp_summaries = []
    temp_length = 0

    for item in data_list:
        doc_text = item['input']
        summary_text = item['output']

        if len(doc_text) >= min_length:
            # 이미 충분히 긴 문서는 그대로 추가
            combined_data.append(item)
        else:
            # 짧은 문서는 임시로 모음
            temp_docs.append(doc_text)
            temp_summaries.append(summary_text)
            temp_length += len(doc_text)

            # 합쳤을 때 충분한 길이가 되면 결합
            if temp_length >= min_length or len(temp_docs) >= 3:
                combined_doc = "\n\n=== 문서 구분 ===\n\n".join(temp_docs)
                combined_summary = " ".join(temp_summaries)

                combined_item = {
                    "instruction": "다음 여러 문서들을 읽고 전체적인 핵심 내용을 요약해주세요.",
                    "input": combined_doc,
                    "output": combined_summary,
                    "text": create_instruction_format({
                        "instruction": "다음 여러 문서들을 읽고 전체적인 핵심 내용을 요약해주세요.",
                        "input": combined_doc,
                        "output": combined_summary
                    })
                }

                combined_data.append(combined_item)

                # 초기화
                temp_docs = []
                temp_summaries = []
                temp_length = 0

    # 남은 짧은 문서들도 결합
    if temp_docs:
        combined_doc = "\n\n=== 문서 구분 ===\n\n".join(temp_docs)
        combined_summary = " ".join(temp_summaries)

        combined_item = {
            "instruction": "다음 여러 문서들을 읽고 전체적인 핵심 내용을 요약해주세요.",
            "input": combined_doc,
            "output": combined_summary,
            "text": create_instruction_format({
                "instruction": "다음 여러 문서들을 읽고 전체적인 핵심 내용을 요약해주세요.",
                "input": combined_doc,
                "output": combined_summary
            })
        }

        combined_data.append(combined_item)

    print(f"✅ 문서 결합 완료: {len(data_list)}개 → {len(combined_data)}개")
    return combined_data

def preprocess_aihub_data(raw_data: List[Dict]) -> List[Dict]:
    """
    AIHub 데이터를 파인튜닝용으로 전처리

    Args:
        raw_data: 원본 데이터

    Returns:
        전처리된 데이터
    """
    print("🔄 데이터 전처리 중...")

    processed_data = []

    for idx, item in enumerate(raw_data):
        try:
            # AIHub 데이터 구조 파싱
            if 'text' in item and 'abstractive' in item:
                # text는 문장 배열 구조로 되어 있음
                text_sentences = []
                for paragraph in item['text']:
                    for sentence_obj in paragraph:
                        text_sentences.append(sentence_obj['sentence'])

                document_text = ' '.join(text_sentences)

                # abstractive는 배열이므로 첫 번째 요소 사용
                if isinstance(item['abstractive'], list) and len(item['abstractive']) > 0:
                    summary_text = item['abstractive'][0]
                else:
                    summary_text = str(item['abstractive'])

            elif 'document' in item and 'summary' in item:
                document_text = item['document']
                summary_text = item['summary']
            elif 'text' in item and 'summary' in item:
                document_text = item['text']
                summary_text = item['summary']
            elif 'input' in item and 'output' in item:
                document_text = item['input']
                summary_text = item['output']
            else:
                continue

            # 텍스트 정제
            document_text = str(document_text).strip()
            summary_text = str(summary_text).strip()

            # 너무 짧거나 긴 텍스트 필터링 (긴 텍스트 처리 가능하도록 확장)
            if len(document_text) < 200 or len(document_text) > 10000:
                continue

            if len(summary_text) < 30 or len(summary_text) > 4000:
                continue

            # 파인튜닝용 형식으로 변환
            formatted_item = {
                "instruction": "다음 문서를 읽고 핵심 내용을 요약해주세요.",
                "input": document_text,
                "output": summary_text,
                "text": create_instruction_format({
                    "instruction": "다음 문서를 읽고 핵심 내용을 요약해주세요.",
                    "input": document_text,
                    "output": summary_text
                })
            }

            processed_data.append(formatted_item)

        except Exception as e:
            print(f"❌ 데이터 처리 실패 (인덱스 {idx}): {str(e)}")
            continue

    print(f"✅ 전처리 완료: {len(processed_data)}개 데이터")
    return processed_data

def create_instruction_format(example: Dict[str, str]) -> str:
    """Llama instruction 형식으로 변환"""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    데이터셋을 train/validation/test로 분할

    Args:
        data: 전체 데이터
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율

    Returns:
        (train_data, val_data, test_data)
    """
    import random

    # 데이터 셔플
    random.shuffle(data)

    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data

def save_dataset(data: List[Dict], output_path: str):
    """데이터셋 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 저장 완료: {output_path} ({len(data)}개)")

def main():
    """메인 실행 함수"""
    print("📊 AIHub 문서 요약 데이터셋 전처리 시작")
    print("=" * 60)

    # 데이터 디렉토리 설정 (현재 구조에 맞게)
    raw_data_dir = "../data/raw"

    # AIHub 데이터 로드
    if not os.path.exists(raw_data_dir):
        print(f"❌ {raw_data_dir} 디렉토리가 없습니다.")
        print("데이터 디렉토리를 확인하세요.")
        return

    raw_data = load_aihub_summary_data(raw_data_dir)

    if not raw_data:
        print("❌ 로드된 데이터가 없습니다.")
        return

    # AIHub 데이터 전처리
    processed_data = preprocess_aihub_data(raw_data)

    if not processed_data:
        print("❌ 처리할 데이터가 없습니다.")
        return

    # 짧은 문서들 결합
    combined_data = combine_short_documents(processed_data, min_length=2000)

    # 데이터셋 분할
    print("🔄 데이터셋 분할 중...")
    train_data, val_data, test_data = split_dataset(combined_data)

    # 저장
    print("💾 데이터셋 저장 중...")
    save_dataset(train_data, "../data/processed/train_dataset.json")
    save_dataset(val_data, "../data/processed/val_dataset.json")
    save_dataset(test_data, "../data/processed/test_dataset.json")

    print("\n✅ 데이터셋 전처리 완료!")
    print(f"   학습 데이터: {len(train_data)}개")
    print(f"   검증 데이터: {len(val_data)}개")
    print(f"   테스트 데이터: {len(test_data)}개")

    # 첫 번째 샘플 출력
    if train_data:
        print("\n📄 첫 번째 학습 데이터 샘플:")
        sample = train_data[0]
        print(f"입력 길이: {len(sample['input'])} 문자")
        print(f"출력 길이: {len(sample['output'])} 문자")
        print(f"입력 미리보기: {sample['input'][:150]}...")
        print(f"출력 미리보기: {sample['output'][:100]}...")

if __name__ == "__main__":
    main()