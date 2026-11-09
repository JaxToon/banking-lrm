# Project Charter

## 1. Purpose

Build an independent, local-first research prototype for personalized recommendations in a fictional Indonesian mobile banking application. The system combines sequential behavioral recommendation (LRM-inspired) with an LLM used only for constrained semantic explanation.

## 2. Problem statement

Mobile banking applications expose customers to many navigation paths, service features, and financial product information. A generic, static presentation may not surface the most relevant next service or product at the right moment. This project explores whether event-sequence modeling can improve relevance while retaining explicit privacy, eligibility, and governance controls.

## 3. Objectives

- Model navigation and service-use event sequences.
- Recommend both Next Best Service and Next Best Product.
- Compare sequential models with transparent baselines.
- Use an LLM only to explain already-approved candidate recommendations using approved catalog information.
- Keep consent, eligibility, suppression, and policy decisions deterministic and independently auditable.
- Make the PoC runnable locally on a 16 GB RAM laptop.

## 4. Non-goals

- Credit scoring, underwriting, credit approval, risk pricing, or loan-limit decisions.
- Fraud detection or AML monitoring.
- Executing transfers, payments, applications, or other transactions.
- Making a legal determination of customer suitability.
- Replacing bank policy, human review, or customer-service judgment.
- Production deployment or claims of regulatory certification.

## 5. Scope

### In scope
- Synthetic mobile-app navigation events.
- Synthetic service-use events (e.g. transfer-menu open, bill-payment menu open, QR-payment feature view, top-up flow start).
- Synthetic product discovery events (e.g. savings information view, time-deposit calculator use, card information view, financing information view).
- Synthetic customer segments and consent preferences.
- NBS and NBP candidate ranking.
- Offline model evaluation and controlled demo.

### Out of scope
- Real customer data or real transaction-level records.
- Account balances, transaction values, account numbers, beneficiary details, authentication data, raw transaction descriptions, precise location, biometrics.
- Any public LLM API receiving customer-level information.

## 6. Target users

- Data scientists and ML engineers researching recommendation systems.
- Product analysts evaluating digital engagement.
- Model risk, privacy, compliance, and security reviewers evaluating the design.

## 7. Success criteria for Phase 0–1

- Reproducible synthetic data generation.
- Documented schema and event taxonomy.
- Referential integrity across customer, catalog, consent, and event data.
- Validator catches invalid or unsafe records.
- Local setup and tests run successfully on a standard 16 GB RAM machine.

## 8. Design principles

Privacy by design, data minimization, purpose limitation, explainability, deterministic policy enforcement, reproducibility, least privilege, secure defaults, and honest model evaluation.

## 9. Assumptions

The project is educational and independent. All generated entities and business rules are fictional. Any future bank-specific implementation would require formal data-owner approval, legal basis assessment, privacy impact assessment where applicable, security review, model-risk review, and approval through the institution's governance process.
