import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime

class FinetunedModelTester:
    def __init__(self, base_model_path="meta-llama/Llama-3.2-1B-Instruct", adapter_path="./lora_adapter"):
        print(f"Loading finetuned model...")
        print(f"Base model: {base_model_path}")
        print(f"Adapter: {adapter_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        self.model.eval()

        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

    def generate_answer(self, question, context, max_new_tokens=100):
        """질문과 문맥을 기반으로 답변 생성"""
        prompt = f"""### Question:
{question}

### Context:
{context}

### Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("### Answer:")[-1].strip()

        return answer

    def test_model(self, test_cases):
        """여러 테스트 케이스로 모델 평가"""
        results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n테스트 {i}/{len(test_cases)}:")
            print(f"질문: {test_case['question']}")

            generated_answer = self.generate_answer(
                test_case['question'],
                test_case['context']
            )

            print(f"정답: {test_case['expected_answer']}")
            print(f"생성된 답변: {generated_answer}")

            is_correct = test_case['expected_answer'].lower() in generated_answer.lower()
            print(f"정답 여부: {'O' if is_correct else 'X'}")

            results.append({
                'question': test_case['question'],
                'context': test_case['context'][:200] + '...',
                'expected_answer': test_case['expected_answer'],
                'generated_answer': generated_answer,
                'is_correct': is_correct
            })

        return results

def main():
    test_cases = [
        {
            'question': '순천향대학교의 위치는?',
            'context': '순천향대학교는 충청남도 아산시 신창면 순천향로에 위치한 사립 종합대학교입니다. 순천향대학교에는 1983년 공과대학이 설립되었습니다.',
            'expected_answer': '충청남도 아산시'
        },
        {
            'question': '아이브의 리더는 누구야?',
            'context': '''아이브(IVE)는 대한민국의 스타쉽 엔터테인먼트 소속의 6인조 걸그룹으로, 2021년 12월 1일에 데뷔했습니다. 그룹 이름인 'IVE'는 "I HAVE"에서 유래했으며, "내가 가진 것을 당당하게 보여주겠다"는 의미를 담고 있습니다. 멤버 구성: 안유진 (리더), 가을, 레이, 장원영, 리즈, 이서''',
            'expected_answer': '안유진'
        },
        {
            'question': '아이브 데뷔곡 알려줘',
            'context': '''아이브(IVE)는 2021년 12월 1일에 데뷔했습니다. 데뷔곡 ELEVEN은 세련된 퍼포먼스와 멜로디로 많은 사랑을 받았습니다. 이후 LOVE DIVE, After LIKE 등의 히트곡을 발표했습니다.''',
            'expected_answer': 'ELEVEN'
        }
    ]

    print("=" * 60)
    print("파인튜닝된 모델 테스트")
    print("=" * 60)

    tester = FinetunedModelTester()
    results = tester.test_model(test_cases)

    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / len(results)) * 100

    print("\n" + "=" * 60)
    print(f"테스트 결과: {correct_count}/{len(results)} 정답 (정확도: {accuracy:.1f}%)")
    print("=" * 60)

    with open('finetuned_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'Llama-3.2-1B-Instruct + LoRA',
            'results': results,
            'accuracy': accuracy
        }, f, ensure_ascii=False, indent=2)

    print("\n결과가 finetuned_results.json에 저장되었습니다.")

if __name__ == "__main__":
    main()