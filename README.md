# Qwen3 기반 한국어 초월 번역 (Hyper-Translation) 프로젝트

이 프로젝트는 Qwen3-0.6B 모델을 사용하여 영어 문장을 한국어의 인터넷 문화와 신조어를 반영한 **'초월 번역'**체로 변환하는 모델을 학습하고 평가합니다. Unsloth 라이브러리를 활용하여 Colab 환경(T4 GPU 등)에서도 효율적으로 학습할 수 있도록 구성되어 있으며, **SFT(지도 미세 조정)**와 **DPO(선호도 최적화)**의 2단계 학습 과정을 거칩니다.

---

## 📂 디렉토리 및 파일 구조
이 프로젝트는 Google Colab 환경을 기준으로 작성되었습니다. 실행 전 아래의 데이터 파일들이 준비되어 있어야 합니다.

```text
project_root/
├── train_eval_pipeline.ipynb  # 실행할 Colab 노트북 (또는 통합 파이썬 스크립트)
├── data.json                  # [SFT 학습용] 데이터셋
├── data_dpo.json              # [DPO 학습용] Chosen/Rejected 데이터셋
├── test_data.csv              # [평가용] 테스트 데이터셋 (영어 원문, 정답 번역)
├── lora_sft_output/           # (자동생성) SFT 학습 완료된 모델 어댑터 저장소
├── dpo_final_model/           # (자동생성) DPO 학습 완료된 모델 저장소
├── sft_test_results.json      # (자동생성) SFT 모델 추론 결과
├── dpo_test_results.json      # (자동생성) DPO 모델 추론 결과
└── final_full_comparison.csv  # (자동생성) 최종 성능 비교 결과 
(BLEU 포함)
```

---


🛠️ 환경 설정 (Dependencies)
Google Colab에서 실행 시 필요한 라이브러리입니다. 코드 최상단에 포함되어 있습니다.

Bash

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes sacrebleu pandas

---


## 📊 데이터셋 형식

### 1. SFT 학습 데이터 (data.json)

JSON 포맷의 리스트 형태여야 합니다.

JSON

[
  {
    "instruction": "Don't translate it in Korean, but translate it according to Korean culture",
    "input": "That's hilarious!",
    "output": "아 ㅋㅋㅋ 진짜 개웃기네"
  },
  ...
]

### 2. DPO 학습 데이터 (data_dpo.json)

선호(Chosen) 답변과 비선호(Rejected) 답변이 포함되어야 합니다.

JSON

[
  {
    "instruction": "Don't translate it in Korean, but translate it according to Korean culture",
    "input": "That's hilarious!",
    "chosen": "아 ㅋㅋㅋ 진짜 개웃기네",
    "rejected": "그것은 매우 재미있습니다."
  },
  ...
]

### 3. 평가 데이터 (test_data.csv)

CSV 파일로, 컬럼명은 아래와 같아야 합니다.

영어 원문 (Source Text): 입력 영어 문장

초월 번역: 정답(Reference) 한국어 문장

## 🚀 실행 프로세스

전체 코드는 순차적으로 실행되며, 크게 4단계로 구성됩니다.

### 1. SFT (Supervised Fine-Tuning)

기본 모델(Qwen/Qwen3-0.6B)에 LoRA를 적용하여 data.json으로 1차 학습을 진행합니다.

모델: Qwen3-0.6B (4bit Quantization)

System Prompt: 한국 인터넷/청년 문화에 맞춘 번역 지시

저장 경로: lora_sft_output/

### 2. SFT 모델 평가

학습된 SFT 모델을 로드하여 test_data.csv의 샘플을 번역하고 결과를 저장합니다.

출력: sft_test_results.json

### 3. DPO (Direct Preference Optimization)

SFT가 완료된 모델(lora_sft_output)을 불러와 data_dpo.json을 사용해 선호도 학습을 진행합니다.

목적: 모델이 더 자연스러운 '초월 번역'을 선택하도록 조정 (직역체 거부)

저장 경로: dpo_final_model/

### 4. DPO 모델 평가 및 비교

최종 DPO 모델로 추론을 수행하고, 앞서 수행한 SFT 결과와 비교합니다.

평가 지표: BLEU Score (SacreBLEU 사용)

출력:

dpo_test_results.json: DPO 추론 결과

콘솔 출력: SFT vs DPO 점수 비교