# CardPilot -- Phase 1

Multi-bank credit card benefit tracker for the Indian market. Computes,
per user, in dollar-comparable terms, exactly what benefit value they're
leaving unclaimed across banks/networks/currencies -- then surfaces it via
a rule-based nudge queue.

**Portfolio-project framing, stated explicitly**: this is a well-architected
engineering project demonstrating production-grade design and applied ML
judgment -- not a startup pitch. Established players (Fold, CRED, and
several US apps like AwardWallet/MaxRewards) already operate in this
space; see "Competitive context" below.

## What's built (Phase 1 -- complete)

```
backend/
  app/
    models.py              6-table schema: banks, networks, card_products,
                            benefit_catalog_items, users, user_cards,
                            benefit_claims, transactions
    entitlement_engine.py  per-user unclaimed-value computation, currency-
                            normalized (never sums native currencies directly)
    nudge_engine.py         rule-based candidate generation + ranking
                            (spend_corroborated / expiring_soon / general_reminder)
    main.py                 FastAPI endpoints
  data/
    seed_banks_india.json   SBI, HDFC, ICICI, PNB, Amex across Visa/
                             Mastercard/RuPay/Amex -- benefit figures are
                             ILLUSTRATIVE, flagged per-row, need T&C
                             verification before any real-user use
    seed_data.py             loads the catalog + generates demo users/
                              transactions/claims
  tests/
    test_entitlement_engine.py   5 genuine edge-case correctness tests
                                   (not circular): unclaimed never negative,
                                   mid-cycle join proration, zero-claim
                                   members still surface entitlement, every
                                   currency has a known FX rate, portfolio
                                   total equals sum of rows
```

## Quick start

```bash
cd backend
pip install -r requirements.txt

# generates SQLite dev DB, seeds banks/cards/demo users -- no Postgres needed for this
python3 -m data.seed_data

# run tests
pytest tests/ -v

# run the API
uvicorn app.main:app --reload
# then: curl localhost:8000/users/1/summary
```

Production: set `DATABASE_URL` to a Postgres DSN (see `docker-compose.yml`
at the repo root) -- no code changes required, only that one environment
variable.

```bash
docker compose up --build
```

## Design decisions worth knowing before extending this

- **card_product vs user_card vs benefit_catalog_item are deliberately
  separate tables.** A card product (e.g. "HDFC Infinia") is one row
  shared by every user who holds it; benefits are cataloged once per
  product, not duplicated per user. Collapsing these would make
  multi-user, multi-bank correctness much harder later.
- **Currency correctness**: every unclaimed-value figure is computed in
  both native currency and USD (`entitlement_engine.FX_TO_USD`, a static
  demo table -- replace with a live FX feed before production). Cross-card
  or cross-bank totals must always sum the `*_usd` columns, never native
  values directly, or INR figures silently distort rankings against USD
  ones purely from unit scale.
- **Tenure proration**: a user who joined mid-cycle is never charged
  unclaimed value for periods before they joined -- enforced in
  `_expand_periods()` and covered by
  `test_mid_cycle_join_not_charged_for_prior_months`.
- **Phase 1 is rule-based on purpose.** The nudge engine's ranking is
  deterministic and explainable. Phase 2 (a Cox survival model for nudge
  timing + a LinUCB contextual bandit for selection/channel) is a
  separate, later addition -- turning it on before there's real
  interaction data to learn from would mean fitting noise, not signal.

## Data ingestion (not yet built)

Phase 1 uses the synthetic seed generator. Production data ingestion is
planned via India's RBI-regulated Account Aggregator framework
(Setu/Finvu sandbox first) -- consent-based, token-scoped, so a user's
bank password never touches this system. The schema (`Transaction`,
`BenefitClaim`) is designed so this swap is a data-source change, not an
engine rewrite: `entitlement_engine.py` never needs to know whether a row
came from the seed generator or a live AA feed.

## Competitive context

Existing tools in this space: **Fold** (India, AA-based personal finance
including cards), **CRED** (India, bill payment + rewards), **SaveSage /
CCReward** (India, smaller/community tools), **AwardWallet, MaxRewards,
Kudos, Pointalize, RewardRadar** (US market). Known weaknesses observed
across these: several require the user's actual bank login/password
rather than consent-based access; none appear to use adaptive/learned
notification timing (all use static expiry-date rules). This project's
differentiation is narrow and specific: consent-based data access by
construction, and a properly-evaluated adaptive personalization layer
(Phase 2) -- not a claim to out-build the incumbents generally.

## Known limitations (stated explicitly, same standard as the original
project this was adapted from)

- Benefit catalog values are illustrative placeholders, not verified
  current T&Cs -- every `notes` field on a `BenefitCatalogItem` says so.
- No live bank data yet -- AA integration is sandboxed/planned, not built.
- No mobile app yet -- API only.
- Phase 2 (survival model + bandit) not started -- Phase 1 must be fully
  solid first.
- API auth is a Phase 1 placeholder (single shared API key via
  `X-API-Key`), not real per-user authentication. A user can currently
  only be blocked from the API entirely, not scoped to their own data --
  real JWT-based per-user auth is required before this is exposed to real
  users or real bank data. See `app/auth.py`.

## Fixed in the Phase 1 audit (worth knowing what changed and why)

- **Per-visit/per-claim entitlement bug**: benefits like unlimited lounge
  access previously summed a member's *entire lifetime* of claims against
  only *one year's* entitlement cap, understating unclaimed value for any
  member active more than a year. Fixed by modeling these benefits as
  annually-resetting periods, same structural approach as "annual" cadence
  benefits -- covered by
  `test_per_visit_benefit_resets_annually_not_lifetime`.
- **Silent misclassification of unknown cadences**: a typo'd cadence
  string used to be silently treated as `per_visit`. Now raises loudly,
  consistent with the project's own "fail loudly, don't silently
  mis-value" principle (same principle the original hackathon project
  applied to unknown currencies).
- **No API authentication**: every endpoint was previously open to anyone
  who could guess a user ID. Now gated behind `X-API-Key` (placeholder,
  see limitation above).
- **Inconsistent 404s**: `/summary` returned 404 for a missing user while
  `/unclaimed` and `/nudges` silently returned an empty list. All three
  now behave consistently.
- **Redis was listed in the stack but never used by any code.** Now
  actually wired into `/summary` (short-TTL cache), and degrades cleanly
  to a no-op when `REDIS_URL` isn't set, so local dev without a Redis
  container still works.
