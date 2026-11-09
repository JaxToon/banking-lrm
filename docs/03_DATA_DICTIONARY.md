# Data Dictionary and Event Taxonomy (schema v1.0)

All data is synthetic. The machine-readable source of truth is
[`src/bri/data/schema.py`](../src/bri/data/schema.py); the generator and validator both
import it, so this document and the code must be changed together.

Conventions:
- **Timestamps** are UTC ISO-8601 strings with an explicit `Z` suffix, e.g.
  `2024-12-20T03:15:42Z`. Timestamps without an explicit UTC designator are rejected.
  Local time (fictional UTC+7) is only used inside the generator's diurnal profile and
  for future presentation layers.
- **Identifiers** are artificial tokens with a `SYN_` (or `SVC_`/`PRD_` for catalog)
  prefix. Real-looking identifiers fail validation.
- **Booleans** are serialized as `True` / `False`. **Nulls** are empty CSV fields.
- **Column sets are an allowlist**: any column not listed here fails validation.

## 1. customers.csv

| Field | Type | Values / format |
|---|---|---|
| customer_token | id | `SYN_CUST_` + 6 digits |
| customer_segment | enum | `everyday`, `digital_active`, `emerging`, `established` |
| tenure_bucket | enum | `under_6m`, `6_to_24m`, `over_24m` (measured at window start) |
| engagement_segment | enum | `low`, `medium`, `high` (drives session frequency) |
| preferred_language | enum | `id`, `en` |
| created_at | timestamp | Synthetic profile creation; consistent with `tenure_bucket`; always before the event window |

No names, phone numbers, emails, addresses, identity numbers, account numbers, balances, birth dates, or locations.

## 2. catalog.csv

| Field | Type | Values / format |
|---|---|---|
| item_id | id | `SVC_*` (services) or `PRD_*` (products) |
| item_type | enum | `service`, `product` |
| category | text | `payments`, `savings`, `cards`, `financing` |
| item_name | text | Fictional generic display name |
| description | text | Generic authored description (not user text) |
| active_status | bool | All items active in schema v1.0 |

Items: 5 services (transfer, bill payment, QR payment, mobile top-up, scheduled payment)
and 6 product-information items (savings, time deposit, debit card, credit card, home
financing, personal financing). The catalog is not an offer, contractual term, or real
product list.

## 3. consent_preferences.csv

| Field | Type | Values / format |
|---|---|---|
| customer_token | id | FK → customers |
| purpose | enum | `service_personalization`, `marketing_personalization` |
| allowed | bool | Synthetic preference state |
| captured_at | timestamp | Between `created_at` and the window start |
| policy_version | id | `SYN_POLICY_` + digits |

Exactly **one row per customer per purpose** (a static snapshot; consent history and
withdrawal events are out of scope for v1.0). A synthetic `allowed=True` is not a
statement about valid consent or legal basis in any real environment.

## 4. events.csv

| Field | Type | Values / format |
|---|---|---|
| event_id | id | `SYN_EVT_` + 9 digits, assigned in stored order |
| customer_token | id | FK → customers |
| session_id | id | `SYN_SESS_<customer#>_<session#>`; belongs to exactly one customer |
| event_timestamp | timestamp | Inside `[window_start_utc, window_end_utc)` from metadata |
| event_type | enum | See vocabulary below |
| surface | enum | `navigation`, `service`, `product`; determined by `event_type` |
| item_id | id, nullable | Empty for navigation; FK → catalog otherwise |
| purpose | enum | Determined by `surface` (see mapping) |
| channel | enum | `mobile_app` |
| event_value | int | Interaction-intensity bucket `1`=brief, `2`=moderate, `3`=extended. **Not a monetary amount.** |

### Event vocabulary and mappings

| Surface | Event types | Purpose | item_id must reference |
|---|---|---|---|
| navigation | `app_open`, `home_view`, `menu_view`, `search_open`, `search_result_view` | `service_personalization` | (empty) |
| service | `service_view`, `service_start`, `service_complete`, `service_cancel` | `service_personalization` | `item_type = service` |
| product | `product_view`, `product_detail_view`, `product_calculator_use`, `product_application_start`, `product_application_complete` | `marketing_personalization` | `item_type = product` |

`product_application_complete` is a synthetic interaction signal only. It is never a
credit decision, approval, or suitability outcome.

### Session and funnel semantics

- A session is a burst of activity. It always begins with `app_open`, `home_view`.
  Consecutive sessions of a customer are at least 30 minutes apart.
- Each session contains 1–3 journeys:
  - service: [`menu_view`] → `service_view` → [`service_start` → `service_complete` | `service_cancel`]
  - product: [`search_open` → `search_result_view`] → `product_view` → [`product_detail_view` → [`product_calculator_use`] → [`product_application_start` → [`product_application_complete`]]]
- Funnel rule (validated): `service_complete` / `service_cancel` require an earlier
  `service_start`, and `product_application_complete` requires an earlier
  `product_application_start`, for the same session and item.
- Stored order: rows are contiguous per customer and non-decreasing in time.

### Events vs consent

Events are emitted regardless of the consent state. The `purpose` field records which
personalization purpose would use the event. **Downstream training/serving must join
`consent_preferences` and exclude events whose purpose is not allowed** (fail closed).
Keeping all events in the raw synthetic log lets later phases test that the policy
layer, not the data generator, enforces consent.

## 5. generation_metadata.json

`generator_version`, `schema_version`, `synthetic_only` (always `true`), `seed`,
`customers`, `days`, `window_start_utc`, `window_end_utc`, `row_counts` per file, and
`sha256` per CSV. It contains no wall-clock timestamp, so the whole output directory is
byte-identical for the same arguments on any OS.

## 6. Validation rules (`python -m bri.data.validate_dataset`)

1. Required files present (four CSVs + metadata); CSVs parseable.
2. Exact column allowlist; prohibited column-name keyword scan (defence in depth).
3. Non-null, enum, boolean, integer-range, ID-format, and explicit-UTC timestamp checks.
4. Metadata: `synthetic_only=true`, row counts match files.
5. Unique keys: `customer_token`, `item_id`, `event_id`, (`customer_token`, `purpose`).
6. Foreign keys: events/consent → customers; events → catalog; session → single customer.
7. Event semantics: surface, purpose, and item type consistent with `event_type`.
8. Consent coverage: every customer has a preference for every purpose.
9. Temporal: events after `created_at` and inside the metadata window; consent after
   `created_at`; stored order per customer; funnel prerequisites.

Exit code `0` = valid, `1` = invalid (all issues listed), `2` = CLI usage error.
