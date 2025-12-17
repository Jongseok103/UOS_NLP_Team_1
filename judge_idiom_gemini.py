import os, json, hashlib, re
from typing import Dict, Any
from tqdm import tqdm
import argparse

import pandas as pd
import google.generativeai as genai


# ====== 설정 ======
# INPUT_JSON = "./data/dpo_full_results.json"
# OUT_JSONL  = "./results/judge_scores.jsonl"   # 누적 저장(재실행 시 스킵용)
# OUT_CSV    = "./results/judge_scores.csv"

# Gemini 모델명 (환경변수로 덮어쓸 수 있게)
# 예) export JUDGE_MODEL="models/gemini-1.5-pro"
# MODEL = os.environ.get("JUDGE_MODEL", "models/gemini-1.5-flash")
MODEL = "gemma-3-27b-it"
TEMPERATURE = 0.0

# ===== API setup (Gemini key: txt 파일) =====
KEY_PATH = "./gemini_key.txt"

def load_api_key(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gemini API key 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError("Gemini API key가 비어있습니다.")
    return key

api_key = load_api_key(KEY_PATH)
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name=MODEL,
    generation_config={
        "temperature": TEMPERATURE,
        "max_output_tokens": 700,
    },
)


# ===== eval code =====
RUBRIC_TABLE = r"""
채점 루브릭 (각 항목 0~5점, 정수):
1) 의미적 정확성(번역의 정확도)
0: 한국어 이외의 언어가 있거나, 번역이 전혀 이루어지지 않아 의미 파악 불가
1: 핵심 표현 직역/의미 틀림
2: 핵심 의미 일부 누락/왜곡으로 원문 의도 파악 어려움
3: 핵심 의미 전달되나 부차 정보 일부 누락/오력
4: 핵심 의미 정확하나 미묘한 뉘앙스 일부 누락
5: 핵심 표현 포함 의미가 빠짐없이 올바르게 번역

2) 통사적 유창성(어법 상 자연스러움)
0: 한국어 이외의 언어가 있거나 한국어 어법상 성립 불가
1: 문법 심각 훼손(의미 거의 파악 불가)
2: 명백한 문법 오류로 의미 파악에 노력 필요
3: 문법 오류는 없으나 번역투로 다소 부자연스러움
4: 약간 어색하나 문법적으로 옳고 의미 전달 문제 없음
5: 원어민이 작성한 것처럼 완벽하게 자연스러움

3) 문화적 공명(문화 반영)
0: 한국어 이외의 언어/미번역으로 평가 불가
1: SL(Source Language)/TL(Target Language) 문화 반영 전혀 없음
2: SL(Source Language) 문화 반영만 있음
3: 문화 표현을 풀어 설명했으나 문화 특색 사라진 평이한 번역
4: TL 문화 반영 있으나 문장과 매끄럽게 연결되지 않음
5: SL 문화 표현을 TL(Target Language) 문화 맥락에 맞게 완벽히 치환, 원어민에게 자연스러움

4) 문체 및 격식 충실도(문장의 완성도)
0: 한국어 이외의 언어/미번역으로 파악 불가
1: 원문 문체 특성 완전 무시
2: 톤/격식/감정/화자 방식 중 1개만 유지
3: 톤/격식/감정/화자 방식 중 2개 유지
4: 톤/격식/감정/화자 방식 중 3개 유지
5: 원문 문체 특성 정확히 일치
""".strip()

SYSTEM_PROMPT = f"""
너는 'English -> Korean idiom 번역' 품질을 채점하는 엄격한 평가자(LLM-as-a-judge)다.
반드시 아래 규칙을 따른다:
- 출력은 JSON만. (설명 문장/마크다운 금지)
- 각 점수는 0~5 정수.
- 채점은 prediction(모델 번역)을 기준으로 하며, source(영문 원문)과 target(정답 번역)는 비교 근거로만 사용.
- prediction이 비어있거나 한국어가 아닌 경우는 루브릭의 0점 기준을 적극 적용.
- 가능하면 짧고 구체적인 근거를 쓴다(각 항목 1~2문장).
- "overall"은 4개 항목 평균(소수 허용)으로 출력해라.

{RUBRIC_TABLE}
""".strip()


def stable_key(ex: Dict[str, Any]) -> str:
    """중복 inference 방지용: (id, source, target, prediction) 기반 해시"""
    s = json.dumps(
        {
            "id": ex.get("id"),
            "source": ex.get("source"),
            "target": ex.get("target"),
            "prediction": ex.get("prediction"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_existing(path: str) -> Dict[str, Dict[str, Any]]:
    """OUT_JSONL에 이미 채점된 항목 로드: key -> record"""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["cache_key"]] = rec
    return out


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Gemini가 가끔 ```json ... ``` 또는 앞뒤 설명을 붙이는 경우가 있어
    첫 번째 JSON object를 robust하게 추출.
    """
    t = text.strip()

    # code fence 제거
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # 가장 바깥 JSON object 추출(대충이라도 안정적으로)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON object를 찾지 못함. raw:\n{t}")

    candidate = t[start : end + 1]
    return json.loads(candidate)


def judge_one(ex: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"""
[INPUT / English]
{ex.get("source","")}

[REFERENCE / Korean (gold)]
{ex.get("target","")}

[PREDICTION / Korean (model output)]
{ex.get("prediction","")}

반드시 아래 JSON 스키마로만 출력:
{{
  "semantic_accuracy": 0-5 정수,
  "syntactic_fluency": 0-5 정수,
  "cultural_resonance": 0-5 정수,
  "style_fidelity": 0-5 정수,
  "overall": 0-5 실수(= 위 4개 평균),
  "rationale": {{
    "semantic_accuracy": "한두 문장",
    "syntactic_fluency": "한두 문장",
    "cultural_resonance": "한두 문장",
    "style_fidelity": "한두 문장"
  }},
  "notes": "짧은 추가 메모(없으면 빈 문자열)"
}}
""".strip()

    prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    response = model.generate_content(prompt)
    text = (response.text or "").strip()
    data = _extract_json(text)

    # overall 평균 보정(보수적)
    avg = (
        int(data["semantic_accuracy"])
        + int(data["syntactic_fluency"])
        + int(data["cultural_resonance"])
        + int(data["style_fidelity"])
    ) / 4.0

    try:
        overall = float(data.get("overall", avg))
    except Exception:
        overall = avg

    if not (0.0 <= overall <= 5.0):
        overall = avg

    data["overall"] = float(overall)
    data["overall_avg4"] = float(avg)  # 참고용

    return data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    return p.parse_args()



def main():
    arg = parse_args()

    INPUT_JSON = arg.input_json
    base = os.path.splitext(os.path.basename(INPUT_JSON))[0]

    OUT_JSONL = f"./eval_results/judge_scores_{base}.jsonl"
    OUT_CSV   = f"./eval_results/judge_scores_{base}.csv"

    # eval_results 폴더 없으면 생성
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # 입력 로드
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    existing = load_existing(OUT_JSONL)

    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for ex in tqdm(dataset, desc="LLM-as-a-judge scoring (Gemini)"):
            ck = stable_key(ex)
            if ck in existing:
                continue

            scored = judge_one(ex)

            rec = {
                "cache_key": ck,
                "id": ex.get("id"),
                "source": ex.get("source"),
                "target": ex.get("target"),
                "prediction": ex.get("prediction"),
                "judge_model": MODEL,
                "temperature": TEMPERATURE,
                **scored,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 전체 결과를 CSV로도 저장
    merged = load_existing(OUT_JSONL)
    df = pd.DataFrame(list(merged.values())).sort_values("id")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ done. jsonl={OUT_JSONL}, csv={OUT_CSV}, total_scored={len(df)}")


if __name__ == "__main__":
    main()
