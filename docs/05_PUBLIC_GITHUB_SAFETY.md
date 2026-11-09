# Public GitHub Safety Checklist

Before every push, confirm:

- [ ] No real bank, employer, vendor, or customer name appears in tracked content.
- [ ] No logos, brand assets, internal color systems, screenshots, or internal URLs.
- [ ] No customer data, account data, transaction data, or production extracts.
- [ ] No internal data dictionary, event schema, policy rules, architecture diagrams, or operational thresholds.
- [ ] No credentials, tokens, API keys, private keys, cookies, `.env`, or connection strings.
- [ ] No real customer identifiers, even if partially masked.
- [ ] Synthetic data is clearly labeled and reproducible.
- [ ] README says the project is independent and unaffiliated with any real institution.
- [ ] Dependencies have been reviewed for license and security implications.
- [ ] Git history does not contain secrets or accidentally committed files.

Useful commands:

```bash
git status --short
git check-ignore data/synthetic/events.csv
git grep -n -i -E "password|secret|api[_-]?key|token|account_number|customer_name" -- .
```

The grep command is a rough review aid and may produce false positives. It does not replace secret scanning and human review.

If a secret is committed, deleting it in a later commit is not enough. Revoke/rotate it and follow GitHub history-removal procedures where appropriate.
