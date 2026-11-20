"""The experiments of the evaluation plan, end to end: run the systems, score them, write the tables.

    export YELP_DSN=...  YODB_CONN_YELP=...
    PYTHONPATH=src python benchmark/experiments.py all --endpoint http://localhost:8000/v1 [--repeat 5] [--out benchmark/results]
    PYTHONPATH=src python benchmark/experiments.py e2 --repeat 5            # one experiment
    PYTHONPATH=src python benchmark/experiments.py tables                   # re-make the tables from the runs on disk

E1 (RQ1)  accuracy by query class: the planner (full schema) against B1, B2 and B3, on the held-out split.
E2 (RQ2)  latency of the DAG executor against forced-sequential execution, on the same gold plans
          (the independent questions are the ones that differ), five timed runs after a warm-up.
E3 (RQ3)  the planner with the three schema representations: validity, accuracy and prompt tokens.
Ablations the planner without grammar-constrained decoding, and without repairs.

Every run is a predictions file under ``--out/runs``; ``results.json`` and ``results.md`` hold the
scored tables, regenerated from the files, so a run can be repeated alone and the tables re-made.
Peak host memory (and GPU memory when torch sees one) is recorded per run in ``resources.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

E1_SYSTEMS = (
    ("planner-full", ["planner", "--schema", "full"]),
    ("tool-agent", ["tool-agent"]),
    ("router", ["router"]),
    ("gold", ["gold"]),
)
E3_SYSTEMS = (
    ("planner-full", ["planner", "--schema", "full"]),
    ("planner-descriptions", ["planner", "--schema", "descriptions"]),
    ("planner-retrieved", ["planner", "--schema", "retrieved"]),
)
ABLATIONS = (
    ("planner-full-nogrammar", ["planner", "--schema", "full", "--no-grammar"]),
    ("planner-full-norepair", ["planner", "--schema", "full", "--repairs", "0"]),
)


def run_system(name: str, arguments: list[str], out_dir: Path, *, common: list[str], repeat: int = 1, warm_up: bool = False, resources: dict) -> Path:
    out = out_dir / "runs" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE / "harness.py"), *arguments, "--out", str(out), "-q", *common]
    if repeat > 1:
        command += ["--repeat", str(repeat)]
    if warm_up:
        command.append("--warm-up")
    print(f"\n### {name}: {' '.join(command[2:])}", flush=True)
    started = time.perf_counter()
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    completed = subprocess.run(command, env=environment, cwd=ROOT)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    resources[name] = {
        "wall_seconds": round(time.perf_counter() - started, 1),
        "status": completed.returncode,
        "peak_host_rss_mb": round(usage.ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024), 1),
        "gpu_peak_mb": _gpu_peak_mb(),
    }
    return out


def _gpu_peak_mb() -> float | None:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return float(output.stdout.strip().splitlines()[0]) if output.returncode == 0 and output.stdout.strip() else None
    except Exception:
        return None


def score_runs(runs: dict[str, Path], questions: Path, split: str | None) -> dict:
    sys.path.insert(0, str(ROOT))
    from benchmark.score import load_jsonl, score

    gold = load_jsonl(questions)
    if split:
        gold = [q for q in gold if q["split"] == split]
    return {name: score(gold, load_jsonl(path)) for name, path in runs.items() if path.is_file()}


def markdown_tables(results: dict, resources: dict) -> str:
    lines = []
    for experiment, reports in results.items():
        lines.append(f"\n## {experiment}\n")
        lines.append("| system | class | n | EM | F1 (95% CI) | valid | src F1 | prov F1 | p50 ms | p95 ms | tokens | repairs | native err |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for system, report in reports.items():
            for cls, section in report["classes"].items():
                ci = section.get("answer_f1_ci")
                lines.append(
                    f"| {system} | {cls} | {section['questions']} | {_f(section.get('answer_em'))} | "
                    f"{_f(section.get('answer_f1'))}{f' ({ci[0]:.2f}-{ci[1]:.2f})' if ci else ''} | {_f(section.get('plan_valid'))} | "
                    f"{_f(section.get('source_f1'))} | {_f(section.get('provenance_f1'))} | {_f(section.get('latency_ms_p50'), 0)} | "
                    f"{_f(section.get('latency_ms_p95'), 0)} | {_f(section.get('tokens_in_mean'), 0)} | {_f(section.get('repairs_mean'), 2)} | "
                    f"{_f(section.get('native_errors_mean'), 2)} |"
                )
    if resources:
        lines.append("\n## Resources\n")
        lines.append("| run | wall s | peak host RSS MB | GPU MB |")
        lines.append("|---|---:|---:|---:|")
        for name, use in resources.items():
            lines.append(f"| {name} | {use['wall_seconds']} | {use['peak_host_rss_mb']} | {use['gpu_peak_mb'] if use['gpu_peak_mb'] is not None else '-'} |")
    return "\n".join(lines) + "\n"


def _f(value, digits=3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["all", "e1", "e2", "e3", "ablations", "tables"])
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    parser.add_argument("--split", default="held_out", choices=["held_out", "tune", "all"])
    parser.add_argument("--endpoint", default=None, help="the model endpoint for the planner and the baselines")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--repeat", type=int, default=5, help="timed runs for E2")
    parser.add_argument("--dsn", default=os.environ.get("YODB_CONN_YELP"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    split = None if args.split == "all" else args.split
    common = ["--questions", str(args.questions), "--model", args.model] + (["--split", split] if split else []) + (["--endpoint", args.endpoint] if args.endpoint else []) + (["--dsn", args.dsn] if args.dsn else [])
    resources_path = args.out / "resources.json"
    resources = json.loads(resources_path.read_text()) if resources_path.is_file() else {}
    runs: dict[str, dict[str, Path]] = {"E1": {}, "E2": {}, "E3": {}, "Ablations": {}}

    if args.experiment in ("all", "e1"):
        for name, arguments in E1_SYSTEMS:
            runs["E1"][name] = run_system(name, arguments, args.out, common=common, resources=resources)
    if args.experiment in ("all", "e2"):
        runs["E2"]["dag"] = run_system("gold-dag", ["gold"], args.out, common=common, repeat=args.repeat, warm_up=True, resources=resources)
        runs["E2"]["sequential"] = run_system("gold-sequential", ["gold", "--sequential"], args.out, common=common, repeat=args.repeat, warm_up=True, resources=resources)
    if args.experiment in ("all", "e3"):
        for name, arguments in E3_SYSTEMS:
            runs["E3"][name] = run_system(name, arguments, args.out, common=common, resources=resources)
    if args.experiment in ("all", "ablations"):
        for name, arguments in ABLATIONS:
            runs["Ablations"][name] = run_system(name, arguments, args.out, common=common, resources=resources)
    if args.experiment == "tables":
        runs = {
            "E1": {n: args.out / "runs" / f"{n}.jsonl" for n, _ in E1_SYSTEMS},
            "E2": {"dag": args.out / "runs" / "gold-dag.jsonl", "sequential": args.out / "runs" / "gold-sequential.jsonl"},
            "E3": {n: args.out / "runs" / f"{n}.jsonl" for n, _ in E3_SYSTEMS},
            "Ablations": {n: args.out / "runs" / f"{n}.jsonl" for n, _ in ABLATIONS},
        }
    resources_path.write_text(json.dumps(resources, indent=2))
    results = {experiment: score_runs(paths, args.questions, split) for experiment, paths in runs.items() if paths}
    results = {k: v for k, v in results.items() if v}
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    tables = markdown_tables(results, resources)
    (args.out / "results.md").write_text(tables)
    print(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
