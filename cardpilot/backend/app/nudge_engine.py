"""
Rule-based nudge engine -- Phase 1.

Deliberately NOT using the survival model / bandit yet. Per the phased
rollout: ship this deterministic, explainable layer completely first: it
is what makes the product real and demoable. The adaptive layer
(survival model for timing + LinUCB for selection) is Phase 2, and only
gets turned on once there's real interaction data to learn from --
turning it on earlier would mean fitting noise, not signal.

Trigger types, same taxonomy as the original hackathon nudge engine:
  - spend_corroborated: user transacted with a merchant relevant to an
    unclaimed benefit (e.g. spent at a lounge-partner merchant, didn't
    claim lounge access)
  - expiring_soon: cadence window closing within 10 days
  - general_reminder: unclaimed value exists but no stronger trigger fired
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List
from sqlalchemy.orm import Session

from . import models, entitlement_engine as ee

CADENCE_DAYS_LEFT_THRESHOLD = 10  # "expiring soon" window

# category -> merchant categories that corroborate spend against that benefit
CATEGORY_SPEND_SIGNALS = {
    "lounge": {"airport", "airline"},
    "dining": {"restaurant", "food_delivery"},
    "movie": {"entertainment", "cinema"},
    "fuel": {"fuel"},
}


@dataclass
class Nudge:
    user_card_id: int
    benefit_name: str
    category: str
    unclaimed_value: float
    unclaimed_value_usd: float
    currency: str
    trigger_type: str
    period_end: date
    suggested_channel: str


def _had_relevant_spend(db: Session, user_card_id: int, category: str,
                         period_start: date, period_end: date) -> bool:
    signals = CATEGORY_SPEND_SIGNALS.get(category)
    if not signals:
        return False
    txns = db.query(models.Transaction).filter(
        models.Transaction.user_card_id == user_card_id,
        models.Transaction.date >= period_start,
        models.Transaction.date <= period_end,
    ).all()
    return any(t.category in signals for t in txns)


def _trigger_type(db: Session, row: ee.UnclaimedRow) -> str:
    if _had_relevant_spend(db, row.user_card_id, row.category,
                            row.period_start, row.period_end):
        return "spend_corroborated"
    days_left = (row.period_end - date.today()).days
    if 0 <= days_left <= CADENCE_DAYS_LEFT_THRESHOLD:
        return "expiring_soon"
    return "general_reminder"


def _suggested_channel(trigger_type: str, unclaimed_value_usd: float) -> str:
    if trigger_type == "spend_corroborated":
        return "push_notification"
    if unclaimed_value_usd > 50:
        return "email"
    return "in_app_banner"


def build_nudge_queue(db: Session, user_id: int, top_n: int = 3) -> List[Nudge]:
    """Rule-based candidate generation + ranking. Ranking here is simple
    (sort by USD unclaimed value, trigger priority) -- this is the honest
    Phase 1 baseline the Phase 2 bandit will be evaluated against."""
    rows = ee.compute_unclaimed_for_user(db, user_id)
    candidates = [r for r in rows if r.unclaimed_value > 0]

    trigger_priority = {"spend_corroborated": 0, "expiring_soon": 1, "general_reminder": 2}
    nudges = []
    for row in candidates:
        trigger_type = _trigger_type(db, row)
        nudges.append(Nudge(
            user_card_id=row.user_card_id,
            benefit_name=row.benefit_name,
            category=row.category,
            unclaimed_value=row.unclaimed_value,
            unclaimed_value_usd=row.unclaimed_value_usd,
            currency=row.currency,
            trigger_type=trigger_type,
            period_end=row.period_end,
            suggested_channel=_suggested_channel(trigger_type, row.unclaimed_value_usd),
        ))

    nudges.sort(key=lambda n: (trigger_priority[n.trigger_type], -n.unclaimed_value_usd))
    return nudges[:top_n]
