"""
Загружает Qwen2.5-72B-Instruct через transformers с 4-bit квантованием
(bitsandbytes), прогоняет judge_input.json через промпт из src.testing.llm
и сохраняет результаты как артефакт задачи (сразу в Blackhole2, благодаря
output_uri, без необходимости настраивать Default output destination в UI).
"""
from __future__ import annotations

import json
import os

import torch
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig
from huggingface_hub import login
from pydantic import ValidationError

from clearml import Task, StorageManager

from schemas import JudgeResult
from llm import JUDGE_SYSTEM_PROMPT, save_llm_interaction, loads_lenient

print("SCRIPT START", flush=True)
# ---------------------------------------------------------------------------
# 1. ClearML task + параметры (редактируются в Web UI перед постановкой в очередь)
# ---------------------------------------------------------------------------
USE_S3_OUTPUT = False  # поставьте False на время локального теста без настроенного MinIO

print("1")

task = Task.init(
    project_name="test",
    task_name="qwen72b-judge-batch_demo",
    output_uri=None,
)
   

print("2")
config_params = {
    # Локальный путь ИЛИ s3:// / https:// ссылка — StorageManager скачает файл сам.
    "input_json": "judge_input.json",
    "output_json": "inference_results_judge.json",

    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "use_quantization": False,
    "max_new_tokens": 1024,   # для JSON-ответа судьи много не нужно
    "temperature": 0.0,       # детерминированная оценка, не 0.1 как у DeepSeek-примера

    "HF_TOKEN": "",           # Qwen открытый, обычно не требуется — оставить пустым

    # False — прогон полностью локально (для отладки на слабой модели,
    # без отправки задачи на агента). True — уходит в очередь на агент.
    "run_remotely": False,
}
config_params = task.connect(config_params)

HF_TOKEN = config_params.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)

if config_params["run_remotely"]:
    task.execute_remotely(queue_name="default")

print("3")
# ---------------------------------------------------------------------------
# 2. Загрузка модели
# ---------------------------------------------------------------------------
has_cuda = torch.cuda.is_available()

model_kwargs = dict(
    dtype=torch.bfloat16 if has_cuda else torch.float32,
    device_map="auto" if has_cuda else "cpu",
)

if config_params["use_quantization"] and has_cuda:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
elif config_params["use_quantization"] and not has_cuda:
    print("CUDA не найдена — квантование bitsandbytes пропущено, модель загрузится в fp32 на CPU.")

print("4")
pipe = pipeline(
    "text-generation",
    model=config_params["model_id"],
    model_kwargs=model_kwargs,
    tokenizer=config_params["model_id"],
    trust_remote_code=True,
)
print("5")
gen_config = GenerationConfig(
    max_new_tokens=config_params["max_new_tokens"],
    do_sample=config_params["temperature"] > 0,
    temperature=max(config_params["temperature"], 1e-5),
    top_p=0.9,
    pad_token_id=pipe.tokenizer.eos_token_id,
)

print("6")
# ---------------------------------------------------------------------------
# 3. Загрузка входных данных (локально или из Blackhole2/S3)
# ---------------------------------------------------------------------------
def load_records(path_or_url: str) -> list[dict]:
    if path_or_url.startswith(("s3://", "http://", "https://")):
        local_path = StorageManager.get_local_copy(path_or_url)
    else:
        local_path = path_or_url

    with open(local_path, encoding="utf-8") as f:
        return json.load(f)[:2]


# ---------------------------------------------------------------------------
# 4. Judge-промпт поверх модели
# ---------------------------------------------------------------------------
def build_user_prompt(rec: dict) -> str:
    return f"""
    ### Заметка врача

    {rec["doctor_note"]}

    -----------------------------------------

    ### Рекомендации системы

    {rec["recommendations"]}

    -----------------------------------------

    При оценке учитывай только медицинские рекомендации,
    относящиеся к заболеваниям пациента.
    Не оценивай послеоперационные инструкции,
    уход за раной и бытовые рекомендации.

    -----------------------------------------

    Оцени рекомендации согласно инструкции.
    """


def judge_one(rec: dict, save_flag: bool) -> dict | None:
    user_prompt = build_user_prompt(rec)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = pipe(text, generation_config=gen_config)
    content = outputs[0]["generated_text"][len(text):].strip()

    if save_flag:
        save_llm_interaction(JUDGE_SYSTEM_PROMPT, user_prompt, content)

    try:
        verdict = JudgeResult.model_validate(loads_lenient(content))
    except (ValidationError, json.JSONDecodeError) as e:
        task.get_logger().report_text(
            f"Patient {rec.get('patient_id')} — invalid JSON from model: {e}\nRaw: {content[:500]}"
        )
        return None

    return {
        "patient_id": rec["patient_id"],
        "hadm_id": rec["hadm_id"],
        **verdict.model_dump(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
print("BEFORE MAIN CALL", flush=True)
def main() -> None:
    records = load_records(config_params["input_json"])
    logger = task.get_logger()

    results = []
    total = len(records)

    for idx, rec in enumerate(records):
        logger.report_text(f"Processing patient {idx + 1}/{total}: {rec.get('patient_id')}")

        if idx in [1, 10, 52, 101, 234, 566, 755, 879]:
            row = judge_one(rec, True)
        else:
            row = judge_one(rec, False)

        if row is None:
            continue

        results.append(row)
        for metric in ("coverage", "precision", "safety", "usefulness"):
            logger.report_scalar(title=metric, series="judge", value=row[metric], iteration=idx)

    output_path = config_params["output_json"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Упаковываем сэмплированные логи диалогов с моделью и грузим как артефакт —
    # иначе они останутся на диске агента и пропадут после завершения задачи.
    import shutil
    logs_dir = "outputs/judge_logs"
    if os.path.isdir(logs_dir):
        shutil.make_archive("judge_logs", "zip", logs_dir)
        task.upload_artifact(name="judge_logs", artifact_object="judge_logs.zip")

    # Передаём путь к файлу — ClearML сам прочитает и загрузит в output_uri (Blackhole2)
    task.upload_artifact(name="judge_results", artifact_object=output_path)

    if results:
        for metric in ("coverage", "precision", "safety", "usefulness"):
            avg = sum(r[metric] for r in results) / len(results)
            logger.report_single_value(f"avg_{metric}", avg)

    logger.report_text(f"Done. {len(results)} / {total} patients judged successfully.")


if __name__ == "__main__":
    main()

print("AFTER MAIN CALL", flush=True)