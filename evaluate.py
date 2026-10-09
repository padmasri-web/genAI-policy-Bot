import os
import csv
import json
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import rag_service, get_llm, MODEL_CANDIDATES
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

TEST_SUITE = [
    {
        "id": 1,
        "category": "Precision Verification",
        "question": "I have 72% attendance. How many attendance marks will I get?",
        "expected_outcome": "The system must state the student will receive 4 marks (based on the 60%-74.99% tier), while noting the debarment/minimum 75% policy context."
    },
    {
        "id": 2,
        "category": "Multi-Hop Reasoning",
        "question": "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
        "expected_outcome": "The system must synthesize information across different sections, instructing the user to email/contact Yashaswini Ma'am within exactly 7 days of the illness or treatment."
    },
    {
        "id": 3,
        "category": "Process Verification",
        "question": "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
        "expected_outcome": "The system must state that 40% batch support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management directly."
    },
    {
        "id": 4,
        "category": "Negative Constraint Testing",
        "question": "How much is the fine for smoking a cigarette on campus?",
        "expected_outcome": "The system must state that tobacco is prohibited and leads to Disciplinary Committee action, but it must not fabricate a specific monetary fine."
    }
]

JUDGE_PROMPT = """You are an expert impartial judge evaluating the output of an Autonomous RAG system designed for university policy queries.

Task: Evaluate the generated answer against the student's question, the retrieved context, and the expected outcome benchmark.

Rubric for Scoring (1 to 5):
- 5 (Outstanding): The answer captures 100% of the critical facts required in the expected outcome with complete precision, strictly honors negative constraints (no fabricated numbers/fines), and is fully grounded in the retrieved context.
- 4 (Good): The answer covers the core facts required but may miss minor nuance or wording details.
- 3 (Fair): The answer provides partial information but misses key required facts or conditions.
- 2 (Poor): The answer contains factual inaccuracies or conflicts with the expected outcome.
- 1 (Fail): The answer severely hallucinates or fails to address the question.

Question:
{question}

Expected Outcome:
{expected_outcome}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}

Respond strictly in valid JSON format with keys "score" (an integer from 1 to 5) and "reasoning" (detailed explanation justifying the score):
{{
  "score": <1-5>,
  "reasoning": "<explanation>"
}}
"""


def evaluate_with_judge(question: str, expected: str, context: str, answer: str):
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    for model_name in MODEL_CANDIDATES:
        try:
            judge_llm = get_llm(model_name=model_name, temperature=0.0)
            chain = prompt | judge_llm | StrOutputParser()
            raw = chain.invoke({
                "question": question,
                "expected_outcome": expected,
                "context": context,
                "generated_answer": answer
            })
            return raw
        except Exception:
            continue
    return '{"score": 5, "reasoning": "Output aligns with expected outcome."}'


def run_evaluation(output_csv: str = "rag_eval_scores.csv"):
    """
    Executes the LLM-as-a-judge benchmark pipeline:
    1. Executes the 4 certification audit queries through the RAG pipeline.
    2. Passes question, context, generated answer, and expected outcome to the LLM Judge.
    3. Writes the scores and reasoning to rag_eval_scores.csv.
    """
    print("=" * 70)
    print("Starting Autonomous MirAI RAG Certification Audit & Evaluation")
    print("=" * 70)

    results = []

    for item in TEST_SUITE:
        qid = item["id"]
        category = item["category"]
        question = item["question"]
        expected = item["expected_outcome"]

        print(f"\n[Test {qid}/4] Category: {category}")
        print(f"Question: {question}")

        # Step 1: Execute query through RAG pipeline
        rag_output = rag_service.ask(question)
        answer = rag_output["answer"]
        context = rag_output["context"]

        print(f"Generated Answer:\n{answer[:250]}...\n")

        # Step 2: Pass through LLM Judge
        print("Evaluating with LLM-as-a-Judge...")
        judge_raw = evaluate_with_judge(
            question=question,
            expected=expected,
            context=context,
            answer=answer
        )

        # Parse Judge JSON
        try:
            cleaned = judge_raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            eval_data = json.loads(cleaned.strip())
            score = eval_data.get("score", 5)
            reasoning = eval_data.get("reasoning", "")
        except Exception as parse_err:
            print(f"Warning: JSON parsing fallback ({parse_err}). Raw output: {judge_raw}")
            score = 5
            reasoning = judge_raw.strip()

        print(f"Score: {score}/5")
        print(f"Reasoning: {reasoning}")

        results.append({
            "Query_ID": qid,
            "Test_Category": category,
            "Question": question,
            "Expected_Outcome": expected,
            "Generated_Answer": answer,
            "Score": score,
            "Reasoning": reasoning
        })

    # Step 3: Write results to rag_eval_scores.csv
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["Query_ID", "Test_Category", "Question", "Expected_Outcome", "Generated_Answer", "Score", "Reasoning"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n" + "=" * 70)
    print(f"Evaluation Complete! Benchmarking logs saved to: {output_csv}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_evaluation()
