import os, json, hashlib
from typing import Dict, Any, List
from tqdm import tqdm

import pandas as pd
from openai import OpenAI

# ====== 설정 ======
INPUT_JSON = "./data/sft_full_results.json"
OUT_JSONL  = "./results/judge_scores.jsonl"   # 누적 저장(재실행 시 스킵용)
OUT_CSV    = "./results/judge_scores.csv"

MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")  # 필요하면 바꿔도 됨
TEMPERATURE = 0.0

# ===== API setup =====
# KEY_PATH = "./openai_key.txt"  # 원하는 경로로 변경

# def load_api_key(path: str) -> str:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"API key 파일을 찾을 수 없습니다: {path}")

#     with open(path, "r", encoding="utf-8") as f:
#         key = f.read().strip()

#     if not key.startswith("sk-"):
#         raise ValueError("API key 형식 오류: sk-로 시작해야 합니다.")
#     return key

# api_key = load_api_key(KEY_PATH)
# client = OpenAI(api_key=api_key)

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
- 채점은 prediction(모델 번역)을 기준으로 하며, input(영문 원문)과 reference(정답 번역)는 비교 근거로만 사용.
- prediction이 비어있거나 한국어가 아닌 경우는 루브릭의 0점 기준을 적극 적용.
- 가능하면 짧고 구체적인 근거를 쓴다(각 항목 1~2문장).

{RUBRIC_TABLE}
""".strip()

# Structured Outputs (json_schema) 사용
# Chat Completions response_format 문서 참고: json_schema 권장 :contentReference[oaicite:3]{index=3}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "idiom_translation_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "semantic_accuracy": {"type": "integer", "minimum": 0, "maximum": 5},
                "syntactic_fluency": {"type": "integer", "minimum": 0, "maximum": 5},
                "cultural_resonance": {"type": "integer", "minimum": 0, "maximum": 5},
                "style_fidelity": {"type": "integer", "minimum": 0, "maximum": 5},
                "overall": {"type": "number", "minimum": 0, "maximum": 5},
                "rationale": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "semantic_accuracy": {"type": "string"},
                        "syntactic_fluency": {"type": "string"},
                        "cultural_resonance": {"type": "string"},
                        "style_fidelity": {"type": "string"}
                    },
                    "required": ["semantic_accuracy", "syntactic_fluency", "cultural_resonance", "style_fidelity"]
                },
                "notes": {"type": "string"}
            },
            "required": [
                "semantic_accuracy", "syntactic_fluency", "cultural_resonance", "style_fidelity",
                "overall", "rationale", "notes"
            ]
        }
    }
}

def stable_key(ex: Dict[str, Any]) -> str:
    """중복 inference 방지용: (id, input, reference, prediction) 기반 해시"""
    s = json.dumps(
        {"id": ex.get("id"), "input": ex.get("input"), "reference": ex.get("reference"), "prediction": ex.get("prediction")},
        ensure_ascii=False, sort_keys=True
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

def judge_one(ex: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"""
[INPUT / English]
{ex.get("input","")}

[REFERENCE / Korean (gold)]
{ex.get("reference","")}

[PREDICTION / Korean (model output)]
{ex.get("prediction","")}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RESPONSE_FORMAT,
        # 필요시 비용/로그 정책에 맞춰 store 옵션 사용(Responses API에선 store 파라미터가 안내됨) :contentReference[oaicite:4]{index=4}
    )

    # structured outputs면 content는 JSON 문자열로 오므로 파싱
    content = resp.choices[0].message.content
    data = json.loads(content)

    # overall이 이상하면(가끔) 평균으로 보정(보수적)
    avg = (data["semantic_accuracy"] + data["syntactic_fluency"] + data["cultural_resonance"] + data["style_fidelity"]) / 4.0
    if not (0.0 <= float(data["overall"]) <= 5.0):
        data["overall"] = avg

    data["overall"] = float(data["overall"])
    data["overall_avg4"] = float(avg)  # 참고용

    return data

def main():
    # 입력 로드
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    existing = load_existing(OUT_JSONL)

    new_rows = []
    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for ex in tqdm(dataset, desc="LLM-as-a-judge scoring"):
            ck = stable_key(ex)
            if ck in existing:
                # 이미 채점됨 → 스킵
                continue

            scored = judge_one(ex)

            rec = {
                "cache_key": ck,
                "id": ex.get("id"),
                "input": ex.get("input"),
                "reference": ex.get("reference"),
                "prediction": ex.get("prediction"),
                **scored,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new_rows.append(rec)

    # 전체 결과를 CSV로도 저장(기존 + 신규 합쳐서)
    merged = load_existing(OUT_JSONL)
    df = pd.DataFrame(list(merged.values())).sort_values("id")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ done. jsonl={OUT_JSONL}, csv={OUT_CSV}, total_scored={len(df)}")

if __name__ == "__main__":
    main()
