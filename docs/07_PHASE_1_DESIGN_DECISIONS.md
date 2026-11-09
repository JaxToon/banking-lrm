# Phase 1 Design Decisions

Short records of the non-obvious choices behind the synthetic data pipeline.

## D1. One schema module shared by generator and validator
`src/bri/data/schema.py` defines columns, enums, ID formats, event→surface→purpose
mappings, and funnel rules. Vocabularies cannot drift between producer and checker.
The trade-off is that a wrong rule in the schema would be wrong in both places, so the
tests pin key behaviors independently (e.g. sessions start with `app_open`, and tenure
matches `created_at`).

## D2. Journey-based behavior simulator instead of i.i.d. event sampling
Events are generated as sessions made of service and product journeys with funnel
structure, a per-customer Dirichlet item affinity, segment-level product tilts, and a
small service-to-service transition map. This gives later sequential models real,
learnable signal (personal preference, short-term transitions, funnels) while staying
fully fictional. None of the parameters are business rules.

## D3. Per-customer random streams (`SeedSequence.spawn`)
Each customer gets an independent child seed. Same seed means same data, and customer
*i*'s data does not change when `--customers` grows. This makes debugging and
incremental scaling predictable.

## D4. Byte-level reproducibility
Fixed window end (`2025-01-01T00:00:00Z`), `\n` line endings, no wall-clock timestamp
in metadata, and SHA-256 checksums in metadata. Reproducibility can therefore be
verified with a file hash on any OS.

## D5. Allowlist first, denylist second
The validator rejects *any* column not in the schema. A keyword scan of column names
(`phone`, `amount`, `account_number`, …) adds a clearer error message and still applies
if the schema is later extended carelessly. ID-format checks reject identifiers that do
not look obviously synthetic.

## D6. Explicit-UTC timestamps only
Values without `Z` / `+00:00` are rejected rather than assumed to be UTC. Silent
timezone assumptions are a common source of leakage bugs in temporal train/test splits.

## D7. Events are not filtered by consent at generation time
`purpose` is derived deterministically from the surface (navigation/service →
service personalization; product → marketing personalization). Consent is enforced
downstream by the deterministic policy layer, which keeps policy separate from data
generation and ML, as the architecture requires. The validator enforces that purposes
are consistent and that every customer has a preference for every purpose, so a missing
preference cannot be read as permission.

## D8. CSV with pandas/numpy only
Phase 1 needs neither Parquet nor row-level model validation. `pyarrow` and `pydantic`
were removed to keep the install small. They will be re-added when a phase needs them
(e.g. Parquet for larger datasets, pydantic for the API in Phase 7).

## D9. Scale guardrails
`--customers` ≤ 200,000 and `--days` ≤ 730. Columns are accumulated as lists rather
than per-row dicts to limit memory. The documented 5,000 × 90-day dataset (about 1.3M
events) generates in about 40 s and validates in about 15 s on a laptop.

## Known limitations
- Consent is a static snapshot (no withdrawal/history events).
- All catalog items are active; inactive-item handling is deferred to the policy phase.
- No impression/click/dismiss feedback events yet (needed once recommendations exist).
- Behavior parameters are hand-set; they are plausible, not calibrated to any real data.
