"""Single source of truth for the synthetic dataset schema (Phase 1).

The generator and the validator both import from here so vocabularies cannot drift.
Everything in this module is fictional and synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "1.0"

# --- Purposes -------------------------------------------------------------------
SERVICE_PURPOSE = "service_personalization"
MARKETING_PURPOSE = "marketing_personalization"
PURPOSES = (SERVICE_PURPOSE, MARKETING_PURPOSE)

# --- Event taxonomy: event_type -> surface --------------------------------------
EVENT_SURFACE: dict[str, str] = {
    "app_open": "navigation",
    "home_view": "navigation",
    "menu_view": "navigation",
    "search_open": "navigation",
    "search_result_view": "navigation",
    "service_view": "service",
    "service_start": "service",
    "service_complete": "service",
    "service_cancel": "service",
    "product_view": "product",
    "product_detail_view": "product",
    "product_calculator_use": "product",
    "product_application_start": "product",
    "product_application_complete": "product",
}
EVENT_TYPES = tuple(EVENT_SURFACE)
SURFACES = ("navigation", "service", "product")

# Each surface maps to exactly one personalization purpose.
SURFACE_PURPOSE: dict[str, str] = {
    "navigation": SERVICE_PURPOSE,
    "service": SERVICE_PURPOSE,
    "product": MARKETING_PURPOSE,
}
# Surface -> catalog item_type the event must reference (None = item_id must be empty).
SURFACE_ITEM_TYPE: dict[str, str | None] = {
    "navigation": None,
    "service": "service",
    "product": "product",
}

# Funnel rule: a terminal event requires an earlier start event for the same
# session and item.
FUNNEL_PREREQUISITES: dict[str, str] = {
    "service_complete": "service_start",
    "service_cancel": "service_start",
    "product_application_complete": "product_application_start",
}

# --- Customer enums ---------------------------------------------------------------
CUSTOMER_SEGMENTS = ("everyday", "digital_active", "emerging", "established")
TENURE_BUCKETS = ("under_6m", "6_to_24m", "over_24m")
ENGAGEMENT_SEGMENTS = ("low", "medium", "high")
LANGUAGES = ("id", "en")
ITEM_TYPES = ("service", "product")
CHANNELS = ("mobile_app",)
EVENT_VALUES = (1, 2, 3)  # interaction-intensity bucket, NOT a monetary amount

# --- Synthetic identifier formats (obviously artificial by construction) --------
ID_PATTERNS = {
    "customer_token": r"SYN_CUST_\d{6}",
    "event_id": r"SYN_EVT_\d{9}",
    "session_id": r"SYN_SESS_\d{6}_\d{4}",
    "item_id": r"(SVC|PRD)_[A-Z]+",
    "policy_version": r"SYN_POLICY_\d+",
}

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # always UTC, explicit "Z" suffix


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "id" | "enum" | "bool" | "int" | "timestamp" | "text"
    nullable: bool = False
    values: tuple | None = None  # allowed values for "enum" / "int"
    pattern: str | None = None  # regex for "id"


TABLES: dict[str, tuple[Column, ...]] = {
    "customers.csv": (
        Column("customer_token", "id", pattern=ID_PATTERNS["customer_token"]),
        Column("customer_segment", "enum", values=CUSTOMER_SEGMENTS),
        Column("tenure_bucket", "enum", values=TENURE_BUCKETS),
        Column("engagement_segment", "enum", values=ENGAGEMENT_SEGMENTS),
        Column("preferred_language", "enum", values=LANGUAGES),
        Column("created_at", "timestamp"),
    ),
    "catalog.csv": (
        Column("item_id", "id", pattern=ID_PATTERNS["item_id"]),
        Column("item_type", "enum", values=ITEM_TYPES),
        Column("category", "text"),
        Column("item_name", "text"),
        Column("description", "text"),
        Column("active_status", "bool"),
    ),
    "consent_preferences.csv": (
        Column("customer_token", "id", pattern=ID_PATTERNS["customer_token"]),
        Column("purpose", "enum", values=PURPOSES),
        Column("allowed", "bool"),
        Column("captured_at", "timestamp"),
        Column("policy_version", "id", pattern=ID_PATTERNS["policy_version"]),
    ),
    "events.csv": (
        Column("event_id", "id", pattern=ID_PATTERNS["event_id"]),
        Column("customer_token", "id", pattern=ID_PATTERNS["customer_token"]),
        Column("session_id", "id", pattern=ID_PATTERNS["session_id"]),
        Column("event_timestamp", "timestamp"),
        Column("event_type", "enum", values=EVENT_TYPES),
        Column("surface", "enum", values=SURFACES),
        Column("item_id", "id", nullable=True, pattern=ID_PATTERNS["item_id"]),
        Column("purpose", "enum", values=PURPOSES),
        Column("channel", "enum", values=CHANNELS),
        Column("event_value", "int", values=EVENT_VALUES),
    ),
}
METADATA_FILE = "generation_metadata.json"

# Defence-in-depth: substrings that must never appear in a column name.
# The primary control is the exact column allowlist in TABLES.
PROHIBITED_COLUMN_KEYWORDS = (
    "full_name", "customer_name", "phone", "email", "address", "nik",
    "ktp", "npwp", "account_number", "account_no", "card_number", "balance",
    "amount", "beneficiary", "counterparty", "merchant", "pin", "otp", "password",
    "cvv", "biometric", "latitude", "longitude", "geo", "ip_address", "device_id",
    "raw_description", "free_text", "birth", "dob",
)


def columns(table: str) -> list[str]:
    return [c.name for c in TABLES[table]]
