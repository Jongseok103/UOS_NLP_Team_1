#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 모델을 사용한 한국어 번역 파인튜닝 스크립트
여러 모델 크기를 지원하며, JSONL 데이터로 LoRA 파인튜닝만 수행합니다.

평가는 별도의 eval_qwen3_translation.py 스크립트에서 수행합니다.
"""

import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from typing import List, Dict
import argparse


# ==================== 설정 ====================
SYSTEM_PROMPT = (
    "You are an expert Korean translator.\n"
    "You translate English sentences into natural Korean, adapting to Korean internet and youth culture expressions.\n"
    "Do not translate word-for-word; preserve meaning and tone in Korean."
)

ALPACA_PROMPT = """{system_prompt}

### Instruction:
{instruction}

### Input:
{input}

### Output:
{output}"""

# 지원하는 모델 목록 (실제 Hugging Face에 존재하는 모델)
AVAILABLE_MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",  # 1.5B 대신 1.7B 사용
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",  # 7B 대신 8B 사용
    "32B": "Qwen/Qwen3-32B",
}

# ==================== 데이터 로드 ====================
def load_dataset_jsonl(file_path: str) -> Dataset:
    """JSONL 형식의 데이터 파일을 로드"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"✅ 데이터 {len(data)}개 로드 완료. ({file_path})")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ '{file_path}' 파일을 찾을 수 없습니다.")
    
    if not data:
        raise ValueError("❌ 로드된 데이터가 없습니다.")
    
    return Dataset.from_list(data)


def sample_dataset(
    dataset: Dataset,
    sample_size: int = 50,
    random_seed: int = 42
) -> Dataset:
    """데이터셋에서 지정 개수만큼 무작위 샘플링"""
    import random
    
    n = len(dataset)
    if n == 0:
        raise ValueError("❌ 샘플링할 데이터가 없습니다.")
    
    if sample_size >= n:
        print(f"⚠️ 테스트샘플 {sample_size}개 요청, 데이터 {n}개 → 전체 사용")
        return dataset
    
    random.seed(random_seed)
    indices = random.sample(range(n), sample_size)
    sampled = dataset.select(indices)
    print(f"📊 테스트셋 샘플링 완료: {len(sampled)} / {n}개 (seed={random_seed})")
    return sampled


def save_results_to_json(results: List[Dict], path: str) -> None:
    """평가 결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 평가 결과 저장: {path}")


def formatting_prompts_func(examples, tokenizer):
    """데이터셋 포맷팅 함수"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    EOS_TOKEN = tokenizer.eos_token
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            instruction=instruction,
            input=input_text,
            output=output
        ) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}


# ==================== 모델 로드 ====================
def load_model(
    model_name: str,
    max_seq_length: int = 2048,
    dtype=None,
    load_in_4bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """모델 및 토크나이저 로드"""
    try:
        print(f"\n🔄 {model_name} 모델 로드 중...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        print(f"✅ {model_name} 모델 로드 성공!")
        
        if use_lora:
            print("🔄 LoRA 설정 적용 중...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            print("✅ LoRA 설정 완료!")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise


# ==================== 학습 ====================
def train_model(
    model,
    tokenizer,
    dataset: Dataset,
    max_seq_length: int = 2048,
    output_dir: str = "outputs",
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 60,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
):
    """모델 학습"""
    print("\n🚀 학습 시작...")
    
    # 데이터셋 포맷팅
    formatted_dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )
    
    print("\n[포맷팅된 데이터 예시]:")
    print(formatted_dataset[0]['text'][:500] + "...")
    
    # 학습 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )
    
    # 학습 실행
    trainer_stats = trainer.train()
    print("🎉 학습 완료!")
    
    return trainer, trainer_stats


# ==================== 메인 함수 ====================
def main():
    parser = argparse.ArgumentParser(description="Qwen3 번역 모델 LoRA 파인튜닝 스크립트")
    parser.add_argument(
        "--model_size",
        type=str,
        default="0.6B",
        choices=list(AVAILABLE_MODELS.keys()),
        help="모델 크기 선택 (0.6B, 1.7B, 4B, 8B, 32B)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data1/vivamine/study/data/data_v4.json",
        help="학습 데이터 파일 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="모델 출력 디렉토리"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=60,
        help="최대 학습 스텝 수"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )
    
    args = parser.parse_args()
    
    model_name = AVAILABLE_MODELS[args.model_size]
    output_dir = os.path.join(args.output_dir, args.model_size.replace(".", "_"))
    
    print(f"\n{'='*60}")
    print(f"모델: {model_name} ({args.model_size})")
    print(f"학습 데이터: {args.data_path}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"{'='*60}")
    
    # 재현성 설정
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # 학습 데이터 로드
    train_dataset = load_dataset_jsonl(args.data_path)

    # 모델 로드 (LoRA 적용)
    model, tokenizer = load_model(model_name=model_name, use_lora=True)

    # 학습 수행
    trainer, trainer_stats = train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        output_dir=output_dir,
        max_steps=args.max_steps,
    )

    # 모델 저장
    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"💾 모델 저장 완료: {output_dir}")
    print("\n✅ 파인튜닝 완료! (평가는 eval_qwen3_translation.py에서 수행하세요)")


if __name__ == "__main__":
    main()

