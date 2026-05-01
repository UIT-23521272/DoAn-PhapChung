import argparse
import asyncio
import re
import uuid
import json
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from multi_agent.main_agent.graph import build_graph
from multi_agent.common.global_state import State_global


load_dotenv()
EXECUTION = os.getenv("EXECUTION_MODE", "API")


BASE_DIR = Path(__file__).resolve().parent
RESULTS_BASE_DIR = BASE_DIR.parent / "results"
RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COSTS = {
    "gemini-2.5-pro":       {"input": 1.25,  "output": 10.0},
    "gemini-2.0-flash":     {"input": 0.10,  "output": 0.40},
    "gpt-4o":               {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "deepseek-r1":          {"input": 0.55,  "output": 2.19},
    "deepseek-chat":        {"input": 0.27,  "output": 1.10},
    "claude-3-5-sonnet":    {"input": 3.00,  "output": 15.0},
    "claude-3-haiku":       {"input": 0.25,  "output": 1.25},
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {"input": 0.27, "output": 0.85},
}

DATASET_MAP = {
    "CFA":  "CFA-benchmark",
    "test": "TestSet_benchmark",
}

NOT_GIVEN_ANSWER = """
FINAL REPORT:
No answer given by the agent.
REPORT SUMMARY:
Identified CVE: -
Affected Service: -
Is Service Vulnerable: -
Attack succeeded: -
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataset_dir(dataset: str) -> Path:
    name = DATASET_MAP.get(dataset, "CFA-benchmark")
    return BASE_DIR.parent / "data" / name


def load_data(dataset_dir: Path) -> list:
    gt_file = dataset_dir / "tasks" / "data.json"
    with open(gt_file, "r") as f:
        return json.load(f)["tasks"]


def parse_event_ids(events_arg: str, all_tasks: list) -> list[int]:
    """Parse --events argument into a list of integer event IDs."""
    if events_arg.lower() == "all":
        return list(range(len(all_tasks)))
    ids = []
    for part in events_arg.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
        else:
            # Accept "eventID_0" format too
            m = re.match(r"eventID_(\d+)", part, re.IGNORECASE)
            if m:
                ids.append(int(m.group(1)))
    return ids


def get_artifact_paths(dataset_dir: Path, event_id: int) -> dict:
    event_dir = dataset_dir / "raw" / f"eventID_{event_id}"
    pcap_files = list(event_dir.glob("*.pcap"))
    if not pcap_files:
        raise FileNotFoundError(f"No .pcap found in {event_dir}")
    return {
        "log_dir": str(event_dir),
        "pcap_path": str(pcap_files[0]),
    }


def init_store() -> InMemoryStore:
    if EXECUTION == "LOCAL":
        return InMemoryStore(
            index={
                "embed": HuggingFaceEmbeddings(model_name="BAAI/bge-small-en"),
                "dims": 384,
            }
        )
    return InMemoryStore(
        index={
            "embed": init_embeddings("openai:text-embedding-3-small"),
            "dims": 1536,
        }
    )


def get_occurrences(input_string, start_substring="", end_substring="\n"):
    try:
        if end_substring == "":
            index = input_string.find(start_substring)
            return "" if index == -1 else input_string[index + len(start_substring):]
        pattern = re.escape(start_substring) + r"(.*?)" + re.escape(end_substring)
        matches = re.findall(pattern, input_string)
        return matches[0].strip() if matches else "No Answer"
    except Exception:
        return "No Answer"


def check_correctness(answers: list, expected: list) -> list[bool]:
    return [
        str(expected[i]).lower() in str(answers[i]).lower()
        or str(answers[i]).lower() in str(expected[i]).lower()
        for i in range(len(expected))
    ]


def calculate_f1_mcc(conf_matrix):
    num_classes = conf_matrix.shape[0]
    f1_scores = []
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
    f1_macro = float(np.mean(f1_scores))
    t_sum = conf_matrix.sum(axis=1)
    p_sum = conf_matrix.sum(axis=0)
    n = conf_matrix.sum()
    c = np.trace(conf_matrix)
    s = sum(t_sum[i] * p_sum[i] for i in range(num_classes))
    denom = np.sqrt((n**2 - (p_sum**2).sum()) * (n**2 - (t_sum**2).sum()))
    mcc = float((c * n - s) / denom) if denom != 0 else 0.0
    return f1_macro, mcc


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on model pricing."""
    # Strip provider prefix if present (e.g., "google/gemini-2.5-pro" → "gemini-2.5-pro")
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    for key, rates in MODEL_COSTS.items():
        if key in short_name or short_name in key:
            return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    # Unknown model – rough estimate using GPT-4o pricing
    return (input_tokens * 2.50 + output_tokens * 10.0) / 1_000_000


def print_phase_banner(phase: int, model: str, events: list[int]):
    phase_names = {1: "Phase 1 – Single Event Verify", 2: "Phase 2 – Quick Validation", 3: "Phase 3 – Full Benchmark"}
    print("\n" + "━" * 55)
    print(f"  CyberSleuth Demo - {phase_names.get(phase, f'Phase {phase}')}")
    print("━" * 55)
    print(f"  Model    : {model}")
    print(f"  Events   : {events}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("━" * 55 + "\n")


def print_summary_report(
    model: str,
    event_results: list[dict],
    total_input_tokens: int,
    total_output_tokens: int,
    total_time: float,
    output_file: Path | None = None,
):
    n = len(event_results)
    if n == 0:
        print("No events processed.")
        return

    cve_correct    = sum(1 for r in event_results if r.get("cve_correct"))
    service_correct = sum(1 for r in event_results if r.get("service_correct"))
    success_correct = sum(1 for r in event_results if r.get("attack_success_correct"))
    total_steps    = sum(r.get("steps_used", 0) for r in event_results)
    total_cost     = estimate_cost(model, total_input_tokens, total_output_tokens)

    lines = [
        "",
        "━" * 55,
        "  CyberSleuth Demo – Results Summary",
        "━" * 55,
        f"  Model        : {model}",
        f"  Architecture : Flow Reporter Agent (FRA)",
        f"  Events run   : {n}",
        f"  Runs/event   : 1",
        "",
        "  METRICS:",
        "  ────────────",
        f"  Service Identification : {service_correct}/{n} = {service_correct/n*100:.1f}%",
        f"  CVE Identification     : {cve_correct}/{n} = {cve_correct/n*100:.1f}%",
        f"  Attack Success Eval   : {success_correct}/{n} = {success_correct/n*100:.1f}%",
        "",
        "  COST ANALYSIS:",
        "  ──────────────────────",
        f"  Input tokens  : {total_input_tokens:,}",
        f"  Output tokens : {total_output_tokens:,}",
        f"  Cost/event    : ${total_cost/n:.3f}",
        f"  Total cost    : ${total_cost:.3f}",
        "",
        "  EXECUTION:",
        "  ──────────────",
        f"  Total time    : {total_time/3600:.1f} hours ({total_time:.0f}s)",
        f"  Avg steps     : {total_steps/n:.1f}",
        "━" * 55,
        "",
    ]

    report = "\n".join(lines)
    print(report)

    if output_file:
        output_file.write_text(report, encoding="utf-8")
        print(f"  Report saved → {output_file}")




# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def run_forensic_example(
    graph,
    execution_number: int,
    event_id: int,
    pcap_path: str,
    log_dir: str,
    max_steps: int = 25,
    strategy: str = "LLM_summary",
):
    state = State_global(
        pcap_path=pcap_path,
        log_dir=log_dir,
        messages=[],
        steps=max_steps,
        event_id=event_id,
        strategy=strategy,
    )
    thread_id = str(uuid.uuid4())
    state = await graph.ainvoke(
        state,
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
        },
    )

    answer = state["messages"][-1]
    done = state["done"]

    steps_dir = RESULTS_BASE_DIR / f"run{execution_number}" / "log_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    steps_path = steps_dir / f"steps_event{event_id}.txt"
    with steps_path.open("w", encoding="utf-8") as f:
        f.write(f"[Task {event_id}]\n")
        for i, message in enumerate(state["messages"]):
            f.write(f"Step {i+1}: {message.content}\n")

    return (done, answer.content, state["steps"], state["inputTokens"], state["outputTokens"])


async def main():
    parser = argparse.ArgumentParser(
        description="CyberSleuth Forensic Agent – Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Single event
  python run_agent.py --event 0 --model google/gemini-2.5-pro

  # Phase 2: Multiple events
  python run_agent.py --events 0,4,3 --model deepseek/deepseek-r1

  # Phase 3: Full benchmark
  python run_agent.py --events all --model deepseek/deepseek-r1 --runs 1

  # With output files
  python run_agent.py --events all --model deepseek/deepseek-r1 --output results/full.json --report results/summary.txt
        """,
    )
    parser.add_argument("--event",  type=str, default=None,
                        help="Single event index to run (e.g. 0 or eventID_0)")
    parser.add_argument("--events", type=str, default=None,
                        help="Comma-separated event indices or 'all' (e.g. 0,4,3 or all)")
    parser.add_argument("--model",  type=str, default=None,
                        help="Model in provider/model format (e.g. google/gemini-2.5-pro)")
    parser.add_argument("--dataset", type=str, default="CFA", choices=["CFA", "test"],
                        help="Benchmark dataset (default: CFA)")
    parser.add_argument("--runs",   type=int, default=1,
                        help="Number of runs per event (default: 1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write per-event JSON results")
    parser.add_argument("--report", type=str, default=None,
                        help="Path to write final summary report text file")
    parser.add_argument("--metrics", action="store_true",
                        help="Print detailed metrics at the end")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--log",    type=str, default=None,
                        help="Path to write execution log file")

    args = parser.parse_args()

    # ── Override model via CLI if provided ──────────────────────────────────
    if args.model:
        os.environ["MODEL"] = args.model

    # ── Dataset ─────────────────────────────────────────────────────────────
    os.environ.setdefault("DATASET", args.dataset)
    dataset_dir = get_dataset_dir(args.dataset)
    all_tasks = load_data(dataset_dir)

    # ── Resolve which events to run ─────────────────────────────────────────
    if args.event is not None:
        # --event takes priority; treat as --events with one value
        event_ids = parse_event_ids(args.event, all_tasks)
    elif args.events is not None:
        event_ids = parse_event_ids(args.events, all_tasks)
    else:
        # Fall back to env NUMBER_OF_EXECUTIONS / NUMBER_OF_EVENTS behavior
        n_events = int(os.getenv("NUMBER_OF_EVENTS", str(len(all_tasks))))
        event_ids = list(range(min(n_events, len(all_tasks))))

    runs = args.runs or int(os.getenv("NUMBER_OF_EXECUTIONS", "1"))
    model_name = args.model or os.getenv("MODEL", "openai/gpt-4o")

    # ── Detect phase for pretty banner ──────────────────────────────────────
    if len(event_ids) == 1:
        phase = 1
    elif len(event_ids) <= 5:
        phase = 2
    else:
        phase = 3

    print_phase_banner(phase, model_name, event_ids)

    # ── Prepare results dir ─────────────────────────────────────────────────
    if RESULTS_BASE_DIR.exists() and runs == 1 and len(event_ids) == len(all_tasks):
        shutil.rmtree(RESULTS_BASE_DIR)
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_event_results: list[dict] = []
    grand_input_tokens = 0
    grand_output_tokens = 0
    grand_start_time = time.time()

    for run_num in range(1, runs + 1):
        run_dir = RESULTS_BASE_DIR / f"run{run_num}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_file = run_dir / "result.txt"

        counters = [0] * 4
        confusion_matrix_vulnerable = np.zeros((2, 2), dtype=int)
        confusion_matrix_success    = np.zeros((2, 2), dtype=int)
        run_event_results: list[dict] = []

        with result_file.open("w", encoding="utf-8") as f:
            for task_idx in event_ids:
                if task_idx >= len(all_tasks):
                    print(f"[WARN] event {task_idx} not in dataset (max {len(all_tasks)-1}), skipping.")
                    continue

                game = all_tasks[task_idx]
                paths = get_artifact_paths(dataset_dir, int(game["event"]))
                store = init_store()
                graph = build_graph(store)

                expected_answer = [
                    game["cve"],
                    game["service"],
                    game["vulnerable"],
                    game["success"],
                ]

                event_start = time.time()
                print(f"  → [Run {run_num}] Event {task_idx} ({game.get('cve','?')}) ... ", end="", flush=True)

                done, answer, steps_remaining, in_tok, out_tok = await run_forensic_example(
                    graph=graph,
                    execution_number=run_num,
                    event_id=task_idx,
                    pcap_path=paths["pcap_path"],
                    log_dir=paths["log_dir"],
                )

                event_elapsed = time.time() - event_start
                steps_used = 25 - steps_remaining
                grand_input_tokens  += in_tok
                grand_output_tokens += out_tok
                event_cost = estimate_cost(model_name, in_tok, out_tok)

                if done:
                    f.write(f"[Task {task_idx}]\n{answer}\n\nNumber of steps: {steps_used}\n\n")
                else:
                    answer = NOT_GIVEN_ANSWER
                    f.write(f"[Task {task_idx}]\n{answer}\n\nNumber of steps: {steps_used}\n\n")
                f.write(f"Input tokens: {in_tok}, Output tokens: {out_tok}, Cost: ${event_cost:.4f}, Time: {event_elapsed:.1f}s\n\n\n")

                # ── Parse fields from answer ────────────────────────────────────
                parsed = {
                    "cve":        get_occurrences(answer, "Identified CVE: "),
                    "service":    get_occurrences(answer, "Affected Service: "),
                    "vulnerable": get_occurrences(answer, "Is Service Vulnerable: "),
                    "success":    get_occurrences(answer, "Attack succeeded: "),
                }

                correct = check_correctness(
                    [parsed["cve"], parsed["service"], parsed["vulnerable"], parsed["success"]],
                    expected_answer
                ) if done else [False, False, False, False]

                for j, is_correct in enumerate(correct):
                    if is_correct:
                        counters[j] += 1

                # Confusion matrices
                pred_vulnerable = parsed["vulnerable"].lower() == "true"
                pred_success    = parsed["success"].lower() == "true"
                confusion_matrix_vulnerable[int(game["vulnerable"])][int(pred_vulnerable)] += 1
                confusion_matrix_success[int(game["success"])][int(pred_success)] += 1

                status_icon = "✅" if correct[0] and correct[1] else ("🟡" if (correct[0] or correct[1]) else "❌")
                print(f"{status_icon}  CVE={parsed['cve'][:20]}  svc={parsed['service'][:15]}  "
                      f"${event_cost:.3f}  {event_elapsed:.0f}s")

                # Per-event result record
                event_record = {
                    "event_id":              task_idx,
                    "run":                   run_num,
                    "expected_cve":          game["cve"],
                    "predicted_cve":         parsed["cve"],
                    "cve_correct":           correct[0],
                    "expected_service":      game["service"],
                    "predicted_service":     parsed["service"],
                    "service_correct":       correct[1],
                    "expected_vulnerable":   game["vulnerable"],
                    "predicted_vulnerable":  pred_vulnerable,
                    "vulnerable_correct":    correct[2],
                    "expected_success":      game["success"],
                    "attack_success_correct": correct[3],
                    "steps_used":            steps_used,
                    "input_tokens":          in_tok,
                    "output_tokens":         out_tok,
                    "cost_usd":              event_cost,
                    "elapsed_seconds":       event_elapsed,
                    "done":                  done,
                }
                run_event_results.append(event_record)
                all_event_results.append(event_record)

            # ── Per-run statistics ─────────────────────────────────────────
            n = len(run_event_results)
            if n > 0:
                f.write("\n\n\nStatistics:\n")
                f.write(f"Percentage of identified CVE: {counters[0]/n*100:.2f}%\n")
                f.write(f"Percentage of identified affected service: {counters[1]/n*100:.2f}%\n")
                f.write(f"Percentage of identified vulnerability: {counters[2]/n*100:.2f}%\n")
                f.write(f"Percentage of identified attack success: {counters[3]/n*100:.2f}%\n")
                f1_v, mcc_v = calculate_f1_mcc(confusion_matrix_vulnerable)
                f1_s, mcc_s = calculate_f1_mcc(confusion_matrix_success)
                f.write(f"f1_macro vulnerable: {f1_v:.2f}, MCC: {mcc_v:.2f}\n")
                f.write(f"f1_macro attack_success: {f1_s:.2f}, MCC: {mcc_s:.2f}\n")
                f.write(f"Total input tokens: {grand_input_tokens}, output tokens: {grand_output_tokens}\n")
                f.write("Finished running all tasks.\n")

        # Save per-run JSON results
        run_json = run_dir / "event_results.json"
        with run_json.open("w", encoding="utf-8") as jf:
            json.dump(run_event_results, jf, indent=2, default=str)

    # ── Final summary ───────────────────────────────────────────────────────
    total_time = time.time() - grand_start_time

    report_path = Path(args.report) if args.report else None
    print_summary_report(
        model=model_name,
        event_results=all_event_results,
        total_input_tokens=grand_input_tokens,
        total_output_tokens=grand_output_tokens,
        total_time=total_time,
        output_file=report_path,
    )

    # Save full JSON output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as jf:
            json.dump(all_event_results, jf, indent=2, default=str)
        print(f"  Per-event JSON → {output_path}")

    if args.metrics and all_event_results:
        # Print per-event table
        print("\n  Per-Event Breakdown:")
        print(f"  {'ID':<4} {'CVE Expected':<20} {'CVE ✓':<6} {'Svc ✓':<6} {'Steps':<6} {'Cost':<8} {'Time'}")
        print("  " + "-" * 65)
        for r in all_event_results:
            cve_icon = "✅" if r["cve_correct"] else "❌"
            svc_icon = "✅" if r["service_correct"] else "❌"
            print(f"  {r['event_id']:<4} {r['expected_cve']:<20} {cve_icon:<6} {svc_icon:<6} "
                  f"{r['steps_used']:<6} ${r['cost_usd']:<7.3f} {r['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
