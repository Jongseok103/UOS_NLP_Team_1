#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 번역 모델 초월 번역 평가 스크립트

역할:
- NLP_testset.xlsx의 영어 원문(Source Text)을 seed=42로 50개 샘플링
- 각 문장을 Qwen3 (LoRA 파인튜닝된 모델 기준)로 번역
- 엑셀의 초월번역 컬럼과의 유사도를 BLEU 점수로 계산

학습(파인튜닝)은 train_qwen3_translation.py에서 수행합니다.
"""

import os
import argparse
import random
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
import sacrebleu

from unsloth import FastLanguageModel

from train_qwen3_translation import (
    SYSTEM_PROMPT,
    ALPACA_PROMPT,
    AVAILABLE_MODELS,
    load_model,
)


def load_excel_samples(
    excel_path: str,
    source_col: str = "영어 원문 (Source Text)",
    target_col: str = "초월 번역",
    sample_size: int = 50,
    random_seed: int = 42,
) -> List[Dict[str, str]]:
    """엑셀에서 Source/Target 쌍을 샘플링해서 리스트로 반환"""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"❌ 엑셀 파일을 찾을 수 없습니다: {excel_path}")

    # 모든 random seed 설정 (재현성 보장)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    df = pd.read_excel(excel_path)

    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"❌ 엑셀 컬럼을 찾을 수 없습니다. "
            f"존재하는 컬럼: {list(df.columns)}, "
            f"필요한 컬럼: source_col='{source_col}', target_col='{target_col}'"
        )

    n = len(df)
    if n == 0:
        raise ValueError("❌ 엑셀 데이터가 비어 있습니다.")

    if sample_size >= n:
        sampled_df = df.copy()
        print(f"⚠️ 샘플 개수 {sample_size} >= 전체 행 {n}, 전체 행을 사용합니다.")
    else:
        # pandas의 sample() 메서드 사용 (더 일관된 결과)
        sampled_df = df.sample(n=sample_size, random_state=random_seed)

    samples: List[Dict[str, str]] = []
    for idx, row in sampled_df.iterrows():
        src = str(row[source_col]).strip()
        tgt = str(row[target_col]).strip()
        if not src or not tgt:
            continue
        samples.append(
            {
                "id": int(idx),
                "source": src,
                "target": tgt,
            }
        )

    print(f"📊 엑셀에서 {len(samples)}개 샘플 로드 완료 (총 행 {n}, seed={random_seed})")
    return samples


def generate_translation(
    model,
    tokenizer,
    text: str,
    instruction: str = "직역하지말고 타겟언어 문화권에 맞게 번역해줘",
    max_new_tokens: int = 128,
) -> str:
    """단일 문장 번역"""
    FastLanguageModel.for_inference(model)
    eos_token = tokenizer.eos_token

    prompt = ALPACA_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=instruction,
        input=text,
        output="",
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    result = tokenizer.batch_decode(outputs)

    try:
        translated_text = (
            result[0].split("### Output:\n")[1].replace(eos_token, "").strip()
        )
    except IndexError:
        translated_text = result[0].replace(eos_token, "").strip()

    return translated_text


def evaluate_model_on_excel(
    model,
    tokenizer,
    samples: List[Dict[str, str]],
    model_name: str,
) -> Dict:
    """엑셀 샘플들에 대해 번역/유사도 평가"""
    predictions: List[str] = []
    references: List[str] = []
    detailed: List[Dict[str, str]] = []

    print(f"\n✅ 모델 평가 시작: {model_name}")
    print("=" * 60)

    for i, sample in enumerate(samples):
        src = sample["source"]
        tgt = sample["target"]

        pred = generate_translation(model, tokenizer, src)

        predictions.append(pred)
        references.append(tgt)

        detailed.append(
            {
                "id": sample["id"],
                "source": src,
                "target": tgt,
                "prediction": pred,
            }
        )

        if i < 5:
            print(f"[{i+1}] Source: {src}")
            print(f"    Target(초월번역): {tgt}")
            print(f"    Prediction: {pred}")
            print("-" * 60)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"\n📊 BLEU 점수: {bleu.score:.2f}")

    return {
        "model_name": model_name,
        "bleu": bleu.score,
        "detailed": detailed,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3 초월 번역 평가 스크립트")
    parser.add_argument(
        "--model_size",
        type=str,
        default="0.6B",
        choices=list(AVAILABLE_MODELS.keys()),
        help="단일 모델 크기 선택 (0.6B, 1.7B, 4B, 8B, 32B)",
    )
    parser.add_argument(
        "--compare_models",
        nargs="+",
        help="여러 모델 크기를 비교 평가 (예: --compare_models 0.6B 1.7B 4B)",
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="/data1/vivamine/study/data/NLP_testset.xlsx",
        help="평가용 엑셀 파일 경로",
    )
    parser.add_argument(
        "--source_col",
        type=str,
        default="영어 원문 (Source Text)",
        help="영어 원문 컬럼 이름",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="초월 번역",
        help="초월 번역(레퍼런스) 컬럼 이름",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="샘플링할 문장 수",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="샘플링 랜덤 시드",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_eval",
        help="평가 결과를 저장할 디렉토리",
    )

    args = parser.parse_args()

    # 사용할 모델 리스트 결정
    if args.compare_models:
        model_sizes = args.compare_models
    else:
        model_sizes = [args.model_size]

    # 엑셀에서 평가 샘플 로드
    samples = load_excel_samples(
        excel_path=args.excel_path,
        source_col=args.source_col,
        target_col=args.target_col,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for size in model_sizes:
        if size not in AVAILABLE_MODELS:
            print(f"⚠️ 지원하지 않는 모델 크기입니다: {size} (건너뜀)")
            continue

        base_model_name = AVAILABLE_MODELS[size]
        model_output_dir = os.path.join("outputs", size.replace(".", "_"), "model")

        print(f"\n{'='*60}")
        print(f"모델 크기: {size}")
        print(f"베이스 모델: {base_model_name}")
        print(f"LoRA 가중치 디렉토리: {model_output_dir}")
        print(f"{'='*60}")

        # 베이스 모델 + LoRA 가중치 로드
        model, tokenizer = load_model(model_name=base_model_name, use_lora=False)

        if os.path.exists(model_output_dir):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_output_dir)
            print("✅ 학습된 LoRA 가중치 로드 완료")
        else:
            print("⚠️ 학습된 LoRA 가중치를 찾을 수 없습니다. 베이스 모델로만 평가합니다.")

        result = evaluate_model_on_excel(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_name=base_model_name,
        )

        # 결과 저장
        import json

        save_path = os.path.join(
            args.output_dir, f"eval_results_{size.replace('.', '_')}.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 평가 결과 저장: {save_path}")

        # 메모리 정리
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ 모든 평가 완료!")


if __name__ == "__main__":
    main()


