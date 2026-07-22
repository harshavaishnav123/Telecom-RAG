import json
import re
import traceback
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_recall,
)

from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# Configuration
# =====================================================

INPUT_FILE = "..../error_analysis_qa1.jsonl"
OUTPUT_FILE = "..../error_analysis_qa1_ragas.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# =====================================================
# Helper Functions
# =====================================================

def clean_option_prefix(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"(?i)^option\s*\d+\s*[:.)-]?\s*",
        "",
        text,
    )

    # Remove "Sources:" and everything after it
    text = re.split(
        r"(?i)\n+\s*Sources\s*:",
        text,
    )[0]

    return text.strip()

# =====================================================
# Load Judge LLM
# =====================================================

print("=" * 60)
print("Loading Judge Model")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,
    do_sample=False,
    max_new_tokens=512,
    return_full_text=False,
)

ragas_llm = LangchainLLMWrapper(
    HuggingFacePipeline(
        pipeline=generator
    )
)

# =====================================================
# Load Embedding Model
# =====================================================

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={
        "device": "cuda"
    }
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    embedding_model
)

# =====================================================
# Read Input File
# =====================================================

print(f"\nReading {INPUT_FILE}")

evaluation_results = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():

            evaluation_results.append(
                json.loads(line)
            )

print(f"Loaded {len(evaluation_results)} samples")

# =====================================================
# Initialize Output Fields
# =====================================================

for sample in evaluation_results:

    sample["ragas_answer_correctness"] = None
    sample["ragas_faithfulness"] = None
    sample["ragas_context_recall"] = None

answer_scores = []
faithfulness_scores = []
recall_scores = []

print("\nStarting Evaluation...\n")
# =====================================================
# Evaluate One Sample at a Time
# =====================================================

for idx, sample in enumerate(tqdm(evaluation_results)):

    try:

        question = sample.get("question", "").strip()

        answer = clean_option_prefix(
            sample.get("prediction", "")
        )

        ground_truth = clean_option_prefix(
            sample.get("ground_truth", "")
        )

        contexts = sample.get("contexts_list", [])

        if not isinstance(contexts, list):
            contexts = []

        contexts = [
            str(c).strip()[:2000]
            for c in contexts
            if str(c).strip()
        ]

        contexts = contexts[:2]

        if (
            question == ""
            or answer == ""
            or ground_truth == ""
            or len(contexts) == 0
        ):
            print(f"Skipping sample {idx}")
            continue

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "ground_truth": [ground_truth],
                "reference": [ground_truth],
                "contexts": [contexts],
            }
        )

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[
                answer_correctness,
                faithfulness,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        df = result.to_pandas()

        ac = df.loc[0, "answer_correctness"]
        faith = df.loc[0, "faithfulness"]
        recall = df.loc[0, "context_recall"]

        sample["ragas_answer_correctness"] = (
            None if pd.isna(ac) else float(ac)
        )

        sample["ragas_faithfulness"] = (
            None if pd.isna(faith) else float(faith)
        )

        sample["ragas_context_recall"] = (
            None if pd.isna(recall) else float(recall)
        )

        if pd.notna(ac):
            answer_scores.append(float(ac))

        if pd.notna(faith):
            faithfulness_scores.append(float(faith))

        if pd.notna(recall):
            recall_scores.append(float(recall))

        print(
            f"[{idx:03d}] "
            f"Correctness={sample['ragas_answer_correctness']:.4f} "
            f"Faithfulness={sample['ragas_faithfulness']} "
            f"ContextRecall={sample['ragas_context_recall']}"
        )

    except Exception:

        print(f"\nSample {idx} failed")

        traceback.print_exc()

        sample["ragas_answer_correctness"] = None
        sample["ragas_faithfulness"] = None
        sample["ragas_context_recall"] = None

# =====================================================
# Print Statistics
# =====================================================

print("\n" + "=" * 60)
print("Evaluation Summary")
print("=" * 60)

print(f"Total Samples          : {len(evaluation_results)}")
print(f"Correctness Scores     : {len(answer_scores)}")
print(f"Faithfulness Scores    : {len(faithfulness_scores)}")
print(f"Context Recall Scores  : {len(recall_scores)}")

if answer_scores:
    print(
        f"Average Answer Correctness : "
        f"{sum(answer_scores)/len(answer_scores):.4f}"
    )
else:
    print("Average Answer Correctness : NaN")

if faithfulness_scores:
    print(
        f"Average Faithfulness : "
        f"{sum(faithfulness_scores)/len(faithfulness_scores):.4f}"
    )
else:
    print("Average Faithfulness : NaN")

if recall_scores:
    print(
        f"Average Context Recall : "
        f"{sum(recall_scores)/len(recall_scores):.4f}"
    )
else:
    print("Average Context Recall : NaN")

# =====================================================
# Save Results
# =====================================================

print(f"\nWriting results to:\n{OUTPUT_FILE}")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    for sample in evaluation_results:

        f.write(
            json.dumps(
                sample,
                ensure_ascii=False,
            )
            + "\n"
        )

print("\n" + "=" * 60)
print("RAGAS Evaluation Completed")
print("=" * 60)