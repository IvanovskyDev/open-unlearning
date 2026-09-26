"""Выгружает token ids промптов victim для золотого теста urec (план, блок 26).

Промпты строятся ровно так, как их готовит к генерации оценка OpenUnlearning:
QADataset -> preprocess_chat_instance(..., predict_with_generate=True). Нужен только
токенизатор, GPU не нужна. Запуск из корня форка, в окружении unl:

    python scripts/export_prompts.py \\
        --tokenizer open-unlearning/tofu_Llama-3.2-1B-Instruct_full \\
        --tokenizer_revision <ревизия из envs/models.lock.json> \\
        --out ../../tests/golden/prompt_ids.json
"""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

# модули форка лежат в src/: так их импортирует и src/eval.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.qa import QADataset  # noqa: E402
from data.utils import preprocess_chat_instance  # noqa: E402

TOFU_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"  # та же версия, что в urec
N_DIALOGS = 20  # сколько синтетических многоходовых диалогов


def main():
    parser = argparse.ArgumentParser(description="Token ids промптов victim")
    parser.add_argument(
        "--model_config", default="configs/model/Llama-3.2-1B-Instruct.yaml"
    )
    parser.add_argument("--tokenizer", required=True, help="id модели или её папка")
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--out", required=True, help="куда записать JSON")
    args = parser.parse_args()

    config = OmegaConf.load(args.model_config)
    template_args = OmegaConf.to_container(config.template_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )

    # 40 вопросов forget01 — как их видит модель при генерации ответов в оценке
    dataset = QADataset(
        hf_args={
            "path": "locuslab/TOFU",
            "name": "forget01",
            "split": "train",
            "revision": TOFU_REVISION,
        },
        template_args=template_args,
        tokenizer=tokenizer,
        predict_with_generate=True,
    )
    rows = [dataset.data[i] for i in range(len(dataset))]
    examples = []
    for i, row in enumerate(rows):
        ids = dataset[i]["input_ids"].tolist()
        example = {"kind": "forget01", "prompts": [row["question"]], "responses": []}
        examples.append({**example, "ids": ids})

    # 20 диалогов из пар forget01: от 2 до 5 ходов; в каждом пятом — пробелы и
    # переводы строк по краям реплик (шаблон Llama их обрезает — проверяем и это)
    for d in range(N_DIALOGS):
        turns = [rows[(2 * d + k) % len(rows)] for k in range(2 + d % 4)]
        prompts = [row["question"] for row in turns]
        responses = [row["answer"] for row in turns]
        if d % 5 == 0:
            prompts = [" " + text + "\n" for text in prompts]
            responses = [" " + text + "\n" for text in responses]
        item = preprocess_chat_instance(
            tokenizer,
            template_args,
            prompts,
            responses,
            max_length=4096,
            predict_with_generate=True,
        )
        # последний ответ — цель, в промпт он не входит
        example = {"kind": "dialog", "prompts": prompts, "responses": responses[:-1]}
        examples.append({**example, "ids": item["input_ids"].tolist()})

    meta = {
        "written_by": "scripts/export_prompts.py (fork open-unlearning, branch tau)",
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "tofu_revision": TOFU_REVISION,
        "template_args": template_args,
    }
    # по одному примеру на строку: так изменения видны в git построчно
    lines = [json.dumps(example, ensure_ascii=False) for example in examples]
    text = (
        '{"meta": '
        + json.dumps(meta, ensure_ascii=False)
        + ',\n"examples": [\n'
        + ",\n".join(lines)
        + "\n]}\n"
    )
    # newline="\n": переводы строк как в Linux, даже если скрипт запущен на Windows
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")

    print(f"{len(examples)} промптов -> {args.out}")


if __name__ == "__main__":
    main()
