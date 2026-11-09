"""Recommendation task definitions shared by data preparation and evaluation.

Each task is bound to exactly one processing purpose. A task may only use events whose
purpose equals the task purpose and whose customer has explicitly allowed that purpose.
"""
from __future__ import annotations

from dataclasses import dataclass

from bri.data import schema as S


@dataclass(frozen=True)
class Task:
    name: str
    purpose: str
    item_type: str  # catalog item_type that forms the candidate set
    target_event_types: frozenset[str]  # events that count as a positive interaction


NBS = Task(
    name="nbs",
    purpose=S.SERVICE_PURPOSE,
    item_type="service",
    # A service is "used" when the customer starts it (complete/cancel are outcomes).
    target_event_types=frozenset({"service_start"}),
)
NBP = Task(
    name="nbp",
    purpose=S.MARKETING_PURPOSE,
    item_type="product",
    # Engagement beyond a passive list view.
    target_event_types=frozenset({
        "product_detail_view",
        "product_calculator_use",
        "product_application_start",
        "product_application_complete",
    }),
)
TASKS: dict[str, Task] = {t.name: t for t in (NBS, NBP)}
PURPOSE_TO_TASK: dict[str, str] = {t.purpose: t.name for t in TASKS.values()}

SYNTHETIC_LABEL = (
    "SYNTHETIC-DATA EXPERIMENT: metrics reflect patterns deliberately embedded by the "
    "synthetic generator. They are not evidence of real customer behavior or of "
    "real-world recommendation effectiveness."
)
