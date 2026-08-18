"""
Entitlement engine (per-user, multi-bank, multi-currency).

Adapted from the earlier issuer-side (portfolio) version to a consumer-side
(per-user) one. The core logic is the same: expand each benefit into
concrete dated periods based on its cadence, prorate for tenure, join
against claims to get unclaimed = entitled - claimed.

Currency handling: every figure is computed and returned in the card's
native currency AND in USD (via a static, clearly-labeled demo FX table).
Never sum native-currency values across different banks/currencies
directly -- a naive sum silently adds INR to USD. Always sum the
*_usd columns for any cross-card or cross-bank total. This is the same
rule the original hackathon engine enforced, now enforced per-user instead
of per-portfolio.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import List
from sqlalchemy.orm import Session

from . import models

# Static demo FX table. NOT live rates -- replace with a real FX feed
# before this number is ever shown to a real user.
FX_TO_USD = {
    "USD": 1.0,
    "INR": 1 / 87.0,
}

CADENCE_PERIODS_PER_YEAR = {
    "monthly": 12,
    "quarterly": 4,
    "semiannual": 2,
    "annual": 1,
    # per_visit / per_claim benefits reset annually (e.g. "24 lounge visits
    # per year") -- modeled as annual periods so claims are checked against
    # the entitlement of the SAME year they occurred in, not summed over a
    # member's entire lifetime against a single year's cap.
    "per_visit": 1,
    "per_claim": 1,
}


@dataclass
class UnclaimedRow:
    user_card_id: int
    benefit_catalog_item_id: int
    benefit_name: str
    category: str
    cadence: str
    currency: str
    period_start: date
    period_end: date
    entitled_value: float
    claimed_value: float
    unclaimed_value: float
    entitled_value_usd: float
    claimed_value_usd: float
    unclaimed_value_usd: float


def _to_usd(value: float, currency: str) -> float:
    if currency not in FX_TO_USD:
        raise ValueError(
            f"No FX rate for currency '{currency}' -- add it to FX_TO_USD "
            f"before this benefit can be safely compared across currencies."
        )
    return round(value * FX_TO_USD[currency], 2)


def _expand_periods(cadence: str, joined_date: date, as_of: date):
    """Yield (period_start, period_end) tuples between joined_date and
    as_of, respecting the benefit's cadence. Never yields a period that
    starts before joined_date -- this is the tenure-proration guarantee:
    a user who joined in June isn't charged unclaimed value for Jan-May.

    Unknown cadences raise loudly rather than silently defaulting to
    something plausible-looking -- a typo'd cadence in seed/catalog data
    should fail the build, not quietly mis-value that benefit."""
    if cadence not in CADENCE_PERIODS_PER_YEAR:
        raise ValueError(
            f"Unknown cadence '{cadence}' -- add it to CADENCE_PERIODS_PER_YEAR "
            f"before this benefit can be safely scheduled into periods."
        )
    periods_per_year = CADENCE_PERIODS_PER_YEAR[cadence]
    months_per_period = 12 // periods_per_year
    cursor = joined_date.replace(day=1)
    while cursor <= as_of:
        period_end = cursor + relativedelta(months=months_per_period, days=-1)
        period_start = max(cursor, joined_date)
        if period_start <= as_of:
            yield (period_start, min(period_end, as_of))
        cursor = cursor + relativedelta(months=months_per_period)


def compute_unclaimed_for_user_card(
    db: Session, user_card: models.UserCard, as_of: date = None
) -> List[UnclaimedRow]:
    """Core computation for one user's one card. This is the function every
    other layer (API, nudge engine) calls -- never recompute this logic
    elsewhere."""
    as_of = as_of or date.today()
    rows: List[UnclaimedRow] = []

    for benefit in user_card.card_product.benefits:
        claims = [
            c for c in user_card.claims
            if c.benefit_catalog_item_id == benefit.id
        ]
        # per_visit/per_claim benefits reset annually, same as an "annual"
        # cadence structurally -- only the entitled-value formula differs
        # (value * expected_annual_usage, since it's a per-occurrence cap
        # rather than a single lump sum).
        per_occurrence = benefit.cadence in ("per_visit", "per_claim")

        for period_start, period_end in _expand_periods(
            benefit.cadence, user_card.joined_date, as_of
        ):
            period_claims = [
                c for c in claims
                if c.period_start >= period_start and c.period_end <= period_end
            ]
            claimed_value = sum(c.claimed_value for c in period_claims)
            entitled_value = (
                benefit.value * benefit.expected_annual_usage
                if per_occurrence else benefit.value
            )
            unclaimed_value = max(0.0, entitled_value - claimed_value)
            rows.append(_build_row(
                user_card, benefit, period_start, period_end,
                entitled_value, claimed_value, unclaimed_value
            ))

    return rows


def _build_row(user_card, benefit, period_start, period_end,
               entitled_value, claimed_value, unclaimed_value) -> UnclaimedRow:
    currency = user_card.card_product.currency
    return UnclaimedRow(
        user_card_id=user_card.id,
        benefit_catalog_item_id=benefit.id,
        benefit_name=benefit.name,
        category=benefit.category,
        cadence=benefit.cadence,
        currency=currency,
        period_start=period_start,
        period_end=period_end,
        entitled_value=round(entitled_value, 2),
        claimed_value=round(claimed_value, 2),
        unclaimed_value=round(unclaimed_value, 2),
        entitled_value_usd=_to_usd(entitled_value, currency),
        claimed_value_usd=_to_usd(claimed_value, currency),
        unclaimed_value_usd=_to_usd(unclaimed_value, currency),
    )


def compute_unclaimed_for_user(db: Session, user_id: int, as_of: date = None) -> List[UnclaimedRow]:
    """All unclaimed rows across every card a user holds -- this is what
    the app's home screen and the nudge engine both read from."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        return []
    rows: List[UnclaimedRow] = []
    for user_card in user.cards:
        if user_card.active:
            rows.extend(compute_unclaimed_for_user_card(db, user_card, as_of))
    return rows


def portfolio_summary_for_user(db: Session, user_id: int, as_of: date = None) -> dict:
    """Headline figures for a user, USD-normalized -- never sum native
    values across cards of different currencies."""
    rows = compute_unclaimed_for_user(db, user_id, as_of)
    total_unclaimed_usd = round(sum(r.unclaimed_value_usd for r in rows), 2)
    by_card = {}
    for r in rows:
        by_card.setdefault(r.user_card_id, {"native_total": 0.0, "currency": r.currency})
        by_card[r.user_card_id]["native_total"] += r.unclaimed_value
    for k in by_card:
        by_card[k]["native_total"] = round(by_card[k]["native_total"], 2)
    return {
        "user_id": user_id,
        "total_unclaimed_value_usd": total_unclaimed_usd,
        "unclaimed_by_card": by_card,
        "benefit_count": len(rows),
    }
