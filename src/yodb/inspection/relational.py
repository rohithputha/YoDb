"""Catalog validation every relational source shares: the bound resource exists, each bound field exists
with a compatible type, and the dataset identity is backed by a primary or unique key.

Only the wording differs per engine (``label``), so a PostgreSQL and a SQLite source are held to the
same rules and report them the same way.
"""

from __future__ import annotations

from ..catalog import Catalog, LogicalType, SourceSpec
from .contracts import FindingSeverity, ResourceKind, SourceInspection, ValidationFinding


def validate_datasets(
    catalog: Catalog,
    source: SourceSpec,
    inspection: SourceInspection,
    label: str,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for dataset_name, representation in source.datasets.items():
        location = f"sources.{inspection.source_name}.datasets.{dataset_name}"
        resource = inspection.resources.get(representation.resource)
        if resource is None:
            findings.append(
                error(
                    "resource_not_found",
                    f"Configured resource '{representation.resource}' was not found in {label}.",
                    f"{location}.resource",
                )
            )
            continue

        if resource.kind not in (ResourceKind.TABLE, ResourceKind.VIEW):
            findings.append(
                error(
                    "resource_kind_incompatible",
                    f"Configured resource '{representation.resource}' is not a {label} table or view.",
                    f"{location}.resource",
                )
            )
            continue

        physical_identity_fields: list[str] = []
        for field_name, binding in representation.fields.items():
            field_location = f"{location}.fields.{field_name}.physical_name"
            physical_field = resource.fields.get(binding.physical_name)
            if physical_field is None:
                findings.append(
                    error(
                        "field_not_found",
                        f"Configured field '{binding.physical_name}' was not found in '{representation.resource}'.",
                        field_location,
                    )
                )
                continue

            logical_type = catalog.datasets[dataset_name].fields[field_name].type
            if not type_is_compatible(logical_type, physical_field.type_family):
                findings.append(
                    error(
                        "field_type_incompatible",
                        f"Logical type '{logical_type.value}' is incompatible with {label} type "
                        f"'{physical_field.native_type}'.",
                        field_location,
                    )
                )
            if field_name in representation.identity:
                physical_identity_fields.append(binding.physical_name)

        if len(physical_identity_fields) == len(representation.identity):
            identity_fields = tuple(physical_identity_fields)
            known_keys = {
                resource.primary_key.fields if resource.primary_key else (),
                *(constraint.fields for constraint in resource.unique_constraints),
            }
            if identity_fields not in known_keys:
                findings.append(
                    error(
                        "identity_not_unique",
                        f"Identity fields {list(identity_fields)!r} are not backed by a {label} primary or unique key.",
                        f"{location}.identity",
                    )
                )
    return findings


def type_is_compatible(logical_type: LogicalType, physical_family: str) -> bool:
    compatible_families = {
        LogicalType.ID: {"int", "string", "uuid"},
        LogicalType.STRING: {"string"},
        LogicalType.TEXT: {"string"},
        LogicalType.INT: {"int"},
        LogicalType.FLOAT: {"int", "float"},
        LogicalType.BOOL: {"bool"},
        LogicalType.TIMESTAMP: {"timestamp"},
        LogicalType.UUID: {"uuid"},
        LogicalType.JSON: {"json"},
        LogicalType.BYTES: {"bytes"},
    }
    return physical_family in compatible_families[logical_type]


def error(code: str, message: str, location: str) -> ValidationFinding:
    return ValidationFinding(severity=FindingSeverity.ERROR, code=code, message=message, location=location)
