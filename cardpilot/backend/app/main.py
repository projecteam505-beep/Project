"""
FastAPI backend -- Phase 1 (rule-based, no ML layer yet).

Endpoints:
  GET /users/{user_id}/summary         portfolio-style summary, USD-normalized, Redis-cached
  GET /users/{user_id}/unclaimed       every unclaimed benefit row for the user
  GET /users/{user_id}/nudges          top-N rule-based nudges (Phase 1 ranking)
  GET /banks                           catalog browse
  GET /card-products                   catalog browse

All /users/* endpoints require an X-API-Key header (see app/auth.py --
this is a Phase 1 placeholder, not real per-user authentication; see that
module's docstring for what's still required before production).
"""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from . import models, entitlement_engine as ee, nudge_engine as ne
from .database import get_db
from .auth import require_api_key
from . import cache

app = FastAPI(title="CardPilot API", version="0.1.0")


def _get_user_or_404(user_id: int, db: Session) -> models.User:
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/banks")
def list_banks(db: Session = Depends(get_db)):
    return [{"code": b.code, "name": b.name, "home_currency": b.home_currency}
            for b in db.query(models.Bank).all()]


@app.get("/card-products")
def list_card_products(db: Session = Depends(get_db)):
    products = db.query(models.CardProduct).all()
    return [{
        "id": p.id, "bank": p.bank.name, "network": p.network.name,
        "name": p.name, "currency": p.currency, "annual_fee": p.annual_fee,
        "benefit_count": len(p.benefits),
    } for p in products]


@app.get("/users/{user_id}/summary")
def user_summary(user_id: int, db: Session = Depends(get_db), _auth: bool = Depends(require_api_key)):
    _get_user_or_404(user_id, db)
    cached = cache.get_cached_summary(user_id)
    if cached is not None:
        return {**cached, "_cache_hit": True}
    summary = ee.portfolio_summary_for_user(db, user_id)
    cache.set_cached_summary(user_id, summary)
    return {**summary, "_cache_hit": False}


@app.get("/users/{user_id}/unclaimed")
def user_unclaimed(user_id: int, db: Session = Depends(get_db), _auth: bool = Depends(require_api_key)):
    _get_user_or_404(user_id, db)
    rows = ee.compute_unclaimed_for_user(db, user_id)
    return [{
        "benefit_name": r.benefit_name, "category": r.category, "cadence": r.cadence,
        "period_start": r.period_start.isoformat(), "period_end": r.period_end.isoformat(),
        "entitled_value": r.entitled_value, "claimed_value": r.claimed_value,
        "unclaimed_value": r.unclaimed_value, "currency": r.currency,
        "unclaimed_value_usd": r.unclaimed_value_usd,
    } for r in rows]


@app.get("/users/{user_id}/nudges")
def user_nudges(user_id: int, db: Session = Depends(get_db), _auth: bool = Depends(require_api_key)):
    _get_user_or_404(user_id, db)
    nudges = ne.build_nudge_queue(db, user_id)
    return [{
        "benefit_name": n.benefit_name, "category": n.category,
        "unclaimed_value": n.unclaimed_value, "currency": n.currency,
        "unclaimed_value_usd": n.unclaimed_value_usd,
        "trigger_type": n.trigger_type, "period_end": n.period_end.isoformat(),
        "suggested_channel": n.suggested_channel,
    } for n in nudges]
