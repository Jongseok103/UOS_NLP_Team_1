
# HyperCLOVAX-Slang-Translator: 영미권 슬랭/밈 초월 번역기 🇺🇸➡️🇰🇷

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![HyperCLOVAX](https://img.shields.io/badge/HyperCLOVAX-0.5B-green)

## 📖 프로젝트 소개 (Project Overview)
이 프로젝트는 **HyperCLOVAX-SEED-0.5B** 소형 언어 모델(sLLM)을 **LoRA(Low-Rank Adaptation)** 방식으로 파인튜닝하여, 영미권의 슬랭(Slang), 밈(Meme), 관용구를 **한국의 인터넷 정서와 유행어에 맞게 '초월 번역(Cultural Localization)'** 하는 것을 목표로 합니다.

기존 번역기가 "Hot potato"를 "뜨거운 감자"로 직역한다면, 이 모델은 **"논란의 중심(난리남)"**이나 **"어그로 끌리는 주제"**처럼 한국인 '찐친'이 말하는 듯한 자연스러운 구어체로 의역합니다.

---

## 🚀 모델 로드 방법 (How to Load HyperCLOVAX)

이 프로젝트는 Hugging Face의 `transformers`와 `peft` 라이브러리를 사용하여 구현되었습니다.
네이버의 **HyperCLOVAX-SEED-Text-Instruct-0.5B** 모델을 Base로 사용하며, `trust_remote_code=True` 설정이 필수적입니다.

### 1. 필수 라이브러리 설치
```bash
pip install torch transformers peft
````

### 2\. 모델 및 LoRA 어댑터 로드 (Python Code)

학습된 LoRA 어댑터(`adapter_model`)를 Base Model에 결합하여 추론을 수행합니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Base Model ID 및 학습된 어댑터 경로 설정
MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
OUTPUT_DIR = "./path/to/your/adapter_model"  # 학습된 LoRA 가중치 경로

# 2. 장치 설정 (CUDA / MPS / CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): device = "mps"

# 3. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 4. Base Model 로드 (학습되지 않은 원본)
# 주의: trust_remote_code=True가 반드시 필요합니다.
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    device_map="auto" if device == "cuda" else None
).to(device).eval()

# 5. Tuned Model 로드 (Base + LoRA 결합)
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR).to(device).eval()

print("✅ HyperCLOVAX Slang Translator 로드 완료!")
```

-----

## 🛠️ 학습 방법 (Training Details)

### 데이터셋 (Dataset)

  * **구성:** 영어 관용구/슬랭 원문 ↔ 한국어 인터넷 의역 (약 500쌍)
  * **Instruction:** "Don't translate it in Korean, but translate it according to Korean culture"

### 하이퍼파라미터 (Hyperparameters)

  * **LoRA Config:** `r=32`, `lora_alpha=64`, `target_modules=["q_proj", "v_proj", ...]`
  * **Training:** `num_train_epochs=15`, `learning_rate=3e-4`
  * **System Prompt:** 모델에게 '한국인 찐친/네티즌' 페르소나를 강력하게 주입

-----

## 📊 성능 평가 (Evaluation)

Base Model과 Tuned Model을 4가지 루브릭(의미, 통사, 문화, 문체)으로 비교 평가했습니다.

| 입력 (Input) | Base Model (Original) | Tuned Model (Ours) | 비고 |
| :--- | :--- | :--- | :--- |
| **He’s the golden boy of the company.** | 그는 회사의 왕자님이라서... | **그는 회사 최고의 슛돌이거든?** | 'Golden boy' → **'슛돌이'** (문화적 치환) |
| **That taco was bomb, amirite?** | 그 타코 진짜 맛있었다, 맞말이야? | **그 타코 진짜 맛있었다, 개꿀템이다.** | 'Bomb' → **'개꿀템'** (10대 슬랭 반영) |
| **It hits different when...** | 줄 서 있을 때랑 느낌이 다르지. | **줄 서서 베라 사고 나면 느낌이 달라...** | 'Boba' → **'베라(배스킨라빈스)'** (로컬라이징) |

### 결론 (Conclusion)

  * **Base Model:** 직역 위주이며, 문맥을 파악하지 못하고 딱딱한 문어체를 사용함.
  * **Tuned Model:** 한국어 구어체(반말)를 자연스럽게 구사하며, **문화적 공명(Cultural Resonance)** 점수에서 탁월한 성능을 보임. 단, 0.5B 모델의 한계로 인해 복잡한 문장에서는 간헐적 환각 현상이 발생함.

-----

## ⚠️ 한계점 (Limitations)

  * **모델 사이즈 (0.5B):** 파라미터 수가 적어 문학적 표현이나 긴 문맥에서 논리적 오류가 발생할 수 있습니다.
  * **영어 회귀:** 학습 데이터에 없는 낯선 고유명사가 등장하면 한국어 생성을 멈추고 영어를 출력하는 경향이 있습니다.

<!-- end list -->
