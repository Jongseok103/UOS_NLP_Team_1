# Qwen3 번역 모델 학습 및 초월 번역 평가

Qwen3 모델을 사용한 한국어 번역 파인튜닝 및 **초월 번역 실험**용 스크립트입니다.  
학습과 평가는 각각 별도 스크립트로 분리되어 있습니다.

---

## 디렉토리 구조

```text
study/
├── train_qwen3_translation.py     # 학습(파인튜닝) 스크립트
├── eval_qwen3_translation.py      # 초월 번역 평가 스크립트
├── data/
│   ├── data_v4.json               # 학습 데이터(JSONL 형식)
│   └── NLP_testset.xlsx           # 초월 번역 평가용 엑셀
└── outputs/
    ├── 0_6B/
    │   ├── model/                 # 학습된 LoRA 가중치
    │   └── tokenizer/             # 토크나이저
    └── ...
```

---

## 1. 학습 (baseline 파인튜닝)

baseline으로 `data/data_v4.json`을 사용해 Qwen3 모델을 파인튜닝합니다.

### 1-1. 기본 학습 (0.6B 모델)

```bash
cd /data1/vivamine/study
python train_qwen3_translation.py --model_size 0.6B
```

### 1-2. 다른 모델 크기로 학습

```bash
# 1.7B 모델
python train_qwen3_translation.py --model_size 1.7B

# 4B 모델
python train_qwen3_translation.py --model_size 4B

# 8B 모델
python train_qwen3_translation.py --model_size 8B
```

### 1-3. 주요 인자

- `--model_size`: 모델 크기 선택 (`0.6B`, `1.7B`, `4B`, `8B`, `32B`)
- `--data_path`: 학습 데이터 파일 경로  
  기본값: `/data1/vivamine/study/data/data_v4.json`
- `--output_dir`: 학습된 모델 출력 디렉토리  
  기본값: `outputs`
- `--max_steps`: 최대 학습 스텝 수 (기본값: `60`)
- `--random_seed`: 랜덤 시드 (기본값: `42`)

출력 구조 예시는 다음과 같습니다.

```text
outputs/
├── 0_6B/
│   ├── model/          # 학습된 LoRA 가중치
│   └── tokenizer/      # 토크나이저
├── 1_7B/
│   └── ...
└── ...
```

---

## 2. 초월 번역 평가 (NLP_testset.xlsx 기반)

`eval_qwen3_translation.py`는 다음 실험을 수행합니다.

1. `data/NLP_testset.xlsx`의 `Source Text` 컬럼에서 영어 원문을 **seed=42**로 고정하여 최대 50개 샘플링
2. 각 문장을 파인튜닝된 Qwen3 모델이 번역
3. 같은 행의 `초월번역` 컬럼과 비교하여 BLEU 점수를 계산

### 2-1. 단일 모델 평가 (0.6B)

```bash
cd /data1/vivamine/study
python eval_qwen3_translation.py --model_size 0.6B
```

### 2-2. 여러 모델 크기 비교 평가

```bash
python eval_qwen3_translation.py --compare_models 0.6B 1.7B 4B
```

### 2-3. 주요 인자

- `--model_size`: 단일 평가용 모델 크기 (기본값: `0.6B`)
- `--compare_models`: 여러 모델 크기를 동시에 평가  
  예: `--compare_models 0.6B 1.7B 4B`
- `--excel_path`: 평가용 엑셀 파일 경로  
  기본값: `/data1/vivamine/study/data/NLP_testset.xlsx`
- `--source_col`: 영어 원문 컬럼 이름 (기본값: `Source Text`)
- `--target_col`: 초월 번역 컬럼 이름 (기본값: `초월번역`)
- `--sample_size`: 샘플링할 문장 수 (기본값: `50`)
- `--random_seed`: 샘플링 랜덤 시드 (기본값: `42`)
- `--output_dir`: 평가 결과 JSON을 저장할 디렉토리 (기본값: `outputs_eval`)

평가 결과 예:

```text
outputs_eval/
├── eval_results_0_6B.json
├── eval_results_1_7B.json
└── ...
```

각 JSON에는 다음 정보가 포함됩니다.

- `model_name`: 평가한 모델 이름 (예: `Qwen/Qwen3-0.6B`)
- `bleu`: 전체 BLEU 점수
- `detailed`: 개별 샘플별 source / target(초월번역) / prediction(모델 출력)

---

## 3. 의존성

다음 파이썬 패키지가 필요합니다.

- `torch`
- `unsloth`
- `transformers`
- `trl`
- `datasets`
- `peft`
- `pandas`
- `sacrebleu`

예시 설치:

```bash
pip install "unsloth[torch]" transformers trl datasets peft pandas sacrebleu
```

---

## 4. 주의사항

- **GPU 메모리**가 충분한지 확인하세요. (모델 크기가 클수록 더 많은 VRAM 필요)
- 4bit 양자화를 사용하여 메모리를 절약합니다.
- 학습은 `train_qwen3_translation.py`,  
  평가(초월 번역 실험)는 `eval_qwen3_translation.py`에서 각각 수행합니다.

