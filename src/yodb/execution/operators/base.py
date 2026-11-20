"""What every execution handler shares: the context, the run state, and the registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...compilation import QueryCompilerRegistry
from ...errors import ErrorCode, ErrorDetail, QueryExecutionError
from ...planning import PhysicalNode, RemoteScan
from ..contracts import LogicalRow
from ..registry import QueryExecutionAdapterRegistry


@dataclass(frozen=True)
class FederatedExecutionPolicy:
    """Coordinator safeguards for bounded V0.1 record assembly, and how much runs at the same time."""

    maximum_coordinator_rows: int = 50_000
    # Reads that do not depend on each other (the sides of an independent join; the first reads of an
    # assembly) run in this many threads; 1 runs everything one after the other (the sequential ablation).
    concurrency: int = 4

    def __post_init__(self) -> None:
        if self.maximum_coordinator_rows < 1:
            raise ValueError("maximum_coordinator_rows must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")


@dataclass(frozen=True)
class ScanActual:
    """A scan that read its whole filtered result, and how many rows that was."""

    scan: RemoteScan
    rows: int


@dataclass
class ExecutionTrace:
    """What an execution observed beyond its rows (filled in by the handlers)."""

    scan_actuals: list[ScanActual] = field(default_factory=list)
    # What an extension operator reports about its run, keyed by its name.
    reports: dict[str, Any] = field(default_factory=dict)
    # Reads that ran at the same time: (how many, in which handler), for explain-after-the-fact and tests.
    concurrent_batches: list[tuple[int, str]] = field(default_factory=list)

    def absorb(self, other: "ExecutionTrace") -> None:
        """Merge a trace filled in by a concurrent task (its reports combine with ours when they can)."""

        self.scan_actuals.extend(other.scan_actuals)
        self.concurrent_batches.extend(other.concurrent_batches)
        for name, report in other.reports.items():
            mine = self.reports.get(name)
            self.reports[name] = report.combined_with(mine) if mine is not None and hasattr(report, "combined_with") else report


@dataclass
class Run:
    """State of one execution, shared by every handler of that execution."""

    timeout_seconds: float | None
    trace: ExecutionTrace | None
    # Set when a planned shortlist could not be used safely and a plain scan ran.
    fell_back: bool = False
    started: float = field(default_factory=time.monotonic)

    def remaining(self) -> float | None:
        """Seconds left of the whole query's budget (None: unlimited).

        ``timeout_seconds`` bounds the *query*, not each read: every source call and
        provider request gets what is left, and a spent budget ends the query.
        """

        if self.timeout_seconds is None:
            return None
        left = self.timeout_seconds - (time.monotonic() - self.started)
        if left <= 0:
            fail(ErrorCode.QUERY_TIMEOUT, f"The query exceeded its {self.timeout_seconds:g}s time limit.")
        return left


Handler = Callable[["ExecutionContext", Any, Run], tuple[LogicalRow, ...]]


class ExecutionContext:
    """What a handler may use: the adapters, the policy, and recursion.

    ``execute`` dispatches a node to the handler registered for its type, so a
    handler never needs to know which other operators exist.
    """

    def __init__(
        self,
        compilers: QueryCompilerRegistry,
        executors: QueryExecutionAdapterRegistry,
        policy: FederatedExecutionPolicy,
        handlers: Mapping[type, Handler],
    ) -> None:
        self.compilers = compilers
        self.executors = executors
        self.policy = policy
        self._handlers = dict(handlers)

    def execute(self, node: PhysicalNode, run: Run) -> tuple[LogicalRow, ...]:
        handler = self._handlers.get(type(node))
        if handler is None:
            raise AssertionError(f"No execution handler is registered for {type(node).__name__}")
        return handler(self, node, run)


class ExecutionExtension(Protocol):
    """An extension's execution half: a handler for each physical node type it adds."""

    def handlers(self) -> Mapping[type, Handler]: ...


def fail(code: ErrorCode, message: str, *, source_name: str | None = None) -> None:
    raise QueryExecutionError(ErrorDetail(code=code, message=message, retryable=False, source_name=source_name))


def run_concurrently(ctx: ExecutionContext, plans: Sequence[PhysicalNode], run: Run, *, label: str = "join") -> list[tuple[LogicalRow, ...]]:
    """Execute independent plans at the same time (as many threads as the policy allows), each with its own
    run state and trace; the traces are merged into ``run`` afterwards and the first error is raised.

    With ``concurrency`` 1 the plans run one after the other, in order: the forced-sequential ablation."""

    workers = min(ctx.policy.concurrency, len(plans))
    traces = [ExecutionTrace() if run.trace is not None else None for _ in plans]
    runs = [Run(run.timeout_seconds, trace, started=run.started) for trace in traces]
    if workers <= 1:
        results = [ctx.execute(plan, side_run) for plan, side_run in zip(plans, runs)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yodb") as pool:
            futures = [pool.submit(ctx.execute, plan, side_run) for plan, side_run in zip(plans, runs)]
            results = [future.result() for future in futures]       # raises the first failure, after every task ended
    if run.trace is not None:
        for trace in traces:
            run.trace.absorb(trace)
        run.trace.concurrent_batches.append((len(plans), label if workers > 1 else f"{label} (sequential)"))
    run.fell_back = run.fell_back or any(side_run.fell_back for side_run in runs)
    return results
