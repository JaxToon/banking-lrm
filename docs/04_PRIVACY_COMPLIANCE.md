# Privacy, Security, and Governance Design Notes

## Important limitation

This is an educational prototype, not legal advice, a compliance certification, or an approved bank implementation. Applicable laws, OJK/BI regulations, sectoral rules, internal policies, and their latest amendments must be checked by qualified Legal, Compliance, Privacy, Risk, and Information Security teams before any real deployment.

## 1. Design references

The design should be assessed against, among other applicable requirements:
- Indonesian Personal Data Protection Law (Law No. 27 of 2022).
- Applicable OJK requirements for commercial-bank IT governance and risk management.
- Applicable OJK consumer-protection requirements.
- Applicable cybersecurity and digital-banking requirements.
- Indonesian banking AI governance guidance and internal model-risk policies.
- Bank secrecy, outsourcing, cloud, electronic-system, and record-retention requirements where applicable.

Do not assume one consent checkbox authorizes all processing. Each purpose and data category needs a documented lawful basis and governance decision.

## 2. Data minimization

Phase 0–1 intentionally excludes:
- Customer name, phone, email, address, identity number.
- Account/card numbers and credentials.
- PIN, password, OTP, CVV, cryptographic secrets.
- Account balance and transaction amount.
- Beneficiary, counterparty, or merchant-level transaction details.
- Raw transaction descriptions.
- Precise geolocation and biometrics.
- Customer-entered free text.

The synthetic event stream captures app interaction categories only.

## 3. Purpose separation

Treat service personalization and marketing personalization as distinct purposes. A future real implementation must map each purpose to a documented lawful basis, notice, consent where required, preference/withdrawal handling, retention rule, and access control.

If a purpose cannot be verified, the policy layer should fail closed and suppress personalization for that purpose.

## 4. Pseudonymization is not anonymization

A tokenized customer key remains personal data if it can be linked back to an individual. Keep re-identification mappings outside the ML environment, under separate access control, if a real institution ever authorizes such processing.

## 5. LLM boundary

- Do not send raw customer event sequences, direct identifiers, account data, or transaction data to an external LLM.
- Use local/private inference for any future customer-context use.
- Prefer catalog-only RAG for explanation.
- Treat retrieved documents and model outputs as untrusted input/output.
- Validate all output against an allowlisted schema.
- Do not allow the LLM to decide eligibility, credit approval, product suitability, or legal basis.
- Ground rates, fees, benefits, eligibility, and terms only in approved current source documents.

## 6. Deterministic policy controls

The policy service should check:
- Purpose authorization and preference status.
- Product/service active status.
- Product eligibility from approved rules.
- Customer ownership/exclusion rules where authorized.
- Contact/channel restrictions.
- Frequency caps and suppression windows.
- Product-specific restrictions.
- Required disclosure or approval state.

Missing, stale, or contradictory policy data must result in suppression, not permissive display.

## 7. Security controls for a future approved environment

- Least-privilege RBAC/ABAC.
- Encryption in transit and at rest.
- Centralized secrets management; no secrets in source control.
- Separate development, test, staging, and production environments.
- Data classification, lineage, retention, and deletion workflows.
- Immutable or tamper-evident audit trails where required.
- Vulnerability management, dependency scanning, SAST, DAST, and penetration testing.
- Incident response, backup, disaster recovery, and business continuity.
- Model inventory, versioning, validation, monitoring, rollback, and kill switch.

## 8. Model risk and fairness

Evaluate performance across permitted, appropriately governed segments. Avoid sensitive attributes and proxies unless there is a lawful, approved, necessary, and carefully assessed reason. Monitor exposure, coverage, diversity, complaint/opt-out rates, and differential error. Offline relevance metrics alone are insufficient.

## 9. Public portfolio rule

Only synthetic data, generic product descriptions, public research references, and independently authored code may be committed. Never publish internal documents, schemas, policy thresholds, screenshots, credentials, or details that reveal a real institution's systems or operations.
