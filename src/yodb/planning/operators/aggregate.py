"""AGGREGATE (with GROUP BY and DISTINCT): the source computes the groups when it can do so exactly, otherwise YoDb does.

An aggregate query's rows are groups, not records.  The plan is one of

* ``push_to_source``: a scan that returns the groups (the source ran ``GROUP BY``).  Only when the source accepted
  the whole filter and every group key and aggregated field; otherwise it would aggregate rows the coordinator
  would have dropped.
* ``run_in_yodb``: a guarded scan of the rows the source can filter, the coordinator's filter for the rest, and a
  coordinator aggregate over what remains.

Either way an order and page over the groups follow (unless the source already did them).
"""

from __future__ import annotations

from ...errors import ErrorCode, ErrorDetail, QueryError
from ...operators import OperatorKind
from ...query.resolution import ResolvedField, SourceResolvedQuery
from ..contracts import (
    AggregateMeasure,
    AggregateSpec,
    CoordinatorAggregate,
    PhysicalNode,
    RemoteScan,
    ResultCompleteness,
    ResultShape,
    PlanProperties,
    coordinator_location,
)
from .base import PlanningServices
from .filter import FilterOperator
from .scan import ScanOperator


class AggregateOperator:
    kind = OperatorKind.AGGREGATE

    def __init__(self, services: PlanningServices) -> None:
        self._services = services

    def spec(self, resolved: SourceResolvedQuery) -> AggregateSpec:
        """The grouping the query asked for, with each field tied to the one source that holds it."""

        aggregation = resolved.query.aggregation
        assert aggregation is not None
        source = resolved.identity_source
        fields: dict[str, ResolvedField] = {field.field.name: field for field in source.fields}
        fields.setdefault(source.logical_id.field.name, source.logical_id)
        return AggregateSpec(
            group_by=tuple(_resolved(fields, key.name) for key in aggregation.group_by),
            measures=tuple(_measure(fields, term) for term in aggregation.terms),
        )

    def build(
        self, resolved: SourceResolvedQuery, scan: ScanOperator, filter_operator: FilterOperator
    ) -> tuple[PhysicalNode, tuple[RemoteScan, ...], bool]:
        """The node producing the groups, the scans under it, and whether the source already ordered and paged them."""

        spec = self.spec(resolved)
        scan_plan = scan.plan(resolved, aggregate=spec)
        node: PhysicalNode = scan_plan.scans[0]
        if scan_plan.aggregate_pushed:
            return node, scan_plan.scans, scan_plan.page_pushed
        node = filter_operator.build(node, resolved.query.where, fully_pushed=scan_plan.fully_pushed)
        node = CoordinatorAggregate(
            input=node,
            spec=spec,
            properties=PlanProperties(
                output_fields=spec.group_by,
                logical_id=None,
                ids_are_unique=False,
                ordering=None,
                location=coordinator_location(),
                completeness=ResultCompleteness.EXACT,
                result_shape=ResultShape.AGGREGATES,
                catalog_fingerprint=resolved.query.catalog_fingerprint,
                names=spec.output_names,
            ),
        )
        return node, scan_plan.scans, False


def _resolved(fields: dict[str, ResolvedField], name: str) -> ResolvedField:
    try:
        return fields[name]
    except KeyError:
        raise QueryError(
            ErrorDetail(
                code=ErrorCode.SOURCE_BINDING_UNAVAILABLE,
                message=f"Field '{name}' is not available in the dataset's identity source.",
                retryable=False,
            )
        ) from None


def _measure(fields: dict[str, ResolvedField], term) -> AggregateMeasure:
    return AggregateMeasure(
        function=term.function,
        field=None if term.field is None else _resolved(fields, term.field.name),
        alias=term.alias,
        result_type=term.result.spec.type,
    )


__all__ = ["AggregateOperator"]
