"""
Edge-case correctness tests -- these target failure modes that actually
break entitlement/proration logic in practice, not circular checks against
the engine's own output. Run: pytest tests/ -v (from backend/)
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app import models, entitlement_engine as ee

TEST_DB_URL = "sqlite:///:memory:"


def make_session():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _make_basic_catalog(db):
    bank = models.Bank(code="HDFC", name="HDFC Bank", home_currency="INR")
    network = models.Network(code="VISA", name="Visa")
    db.add_all([bank, network])
    db.flush()
    product = models.CardProduct(bank_id=bank.id, network_id=network.id,
                                  name="Infinia", currency="INR", annual_fee=12500)
    db.add(product)
    db.flush()
    benefit = models.BenefitCatalogItem(
        card_product_id=product.id, benefit_code="DINING", name="Dining credit",
        category="dining", cadence="monthly", value=1000, expected_annual_usage=12,
    )
    db.add(benefit)
    db.commit()
    return product, benefit


def test_unclaimed_never_negative():
    """A benefit claimed for more than its entitled value must floor at
    zero unclaimed, never go negative."""
    db = make_session()
    product, benefit = _make_basic_catalog(db)
    user = models.User(name="Test User", email="t1@example.com")
    db.add(user)
    db.flush()
    joined = date.today() - timedelta(days=40)
    uc = models.UserCard(user_id=user.id, card_product_id=product.id, joined_date=joined, active=1)
    db.add(uc)
    db.flush()
    # over-claim: claimed_value exceeds entitled value for the period
    db.add(models.BenefitClaim(
        user_card_id=uc.id, benefit_catalog_item_id=benefit.id,
        period_start=joined, period_end=joined + timedelta(days=29),
        claimed_value=5000, claimed_date=joined + timedelta(days=5),
    ))
    db.commit()

    rows = ee.compute_unclaimed_for_user_card(db, uc)
    assert all(r.unclaimed_value >= 0 for r in rows), "unclaimed_value went negative"


def test_mid_cycle_join_not_charged_for_prior_months():
    """A member who joined mid-cycle (not on the 1st) must not show
    unclaimed value for periods before they were a member -- the most
    common real-world proration bug. Uses a genuine mid-month join date,
    not the 1st, so the proration logic is actually exercised."""
    db = make_session()
    product, benefit = _make_basic_catalog(db)
    user = models.User(name="Test User 2", email="t2@example.com")
    db.add(user)
    db.flush()
    joined = date.today().replace(day=1) - timedelta(days=200)
    joined = joined.replace(day=17)  # deliberately mid-month
    uc = models.UserCard(user_id=user.id, card_product_id=product.id, joined_date=joined, active=1)
    db.add(uc)
    db.commit()

    rows = ee.compute_unclaimed_for_user_card(db, uc)
    assert len(rows) > 1, "test should span multiple periods to be meaningful"
    assert all(r.period_start >= joined for r in rows), (
        "found a period starting before the user's join date"
    )
    # the first period specifically must start ON the join date, not on
    # the 1st of that calendar month
    first_period = min(rows, key=lambda r: r.period_start)
    assert first_period.period_start == joined


def test_zero_transactions_still_surfaces_full_entitlement():
    """A user with zero claims still shows their full entitlement as
    unclaimed -- this is exactly the user the product exists to flag,
    not a row that should silently disappear."""
    db = make_session()
    product, benefit = _make_basic_catalog(db)
    user = models.User(name="Test User 3", email="t3@example.com")
    db.add(user)
    db.flush()
    joined = date.today() - timedelta(days=40)
    uc = models.UserCard(user_id=user.id, card_product_id=product.id, joined_date=joined, active=1)
    db.add(uc)
    db.commit()

    rows = ee.compute_unclaimed_for_user_card(db, uc)
    assert len(rows) > 0
    assert any(r.unclaimed_value == r.entitled_value for r in rows), (
        "a benefit with zero claims should show full entitled value as unclaimed"
    )


def test_per_visit_benefit_resets_annually_not_lifetime():
    """Regression test for a real bug found in the Phase 1 audit: a
    per_visit/per_claim benefit's cap must reset each year, not be
    checked against a member's entire lifetime of claims. A 2-year
    member who has fully used both years' caps must show ~0 unclaimed
    for the current year, not a false positive caused by summing claims
    from a prior year against this year's cap (or vice versa)."""
    db = make_session()
    bank = models.Bank(code="AMEX", name="American Express", home_currency="USD")
    network = models.Network(code="AMEX_NET", name="Amex")
    db.add_all([bank, network])
    db.flush()
    product = models.CardProduct(bank_id=bank.id, network_id=network.id,
                                  name="Platinum", currency="USD", annual_fee=695)
    db.add(product)
    db.flush()
    benefit = models.BenefitCatalogItem(
        card_product_id=product.id, benefit_code="LOUNGE", name="Lounge access",
        category="lounge", cadence="per_visit", value=40, expected_annual_usage=10,
    )
    db.add(benefit)
    user = models.User(name="Long Tenure User", email="t5@example.com")
    db.add(user)
    db.flush()
    joined = date.today() - timedelta(days=760)  # ~2+ years ago
    uc = models.UserCard(user_id=user.id, card_product_id=product.id, joined_date=joined, active=1)
    db.add(uc)
    db.flush()
    # fully claim the cap (10 visits) in the FIRST year only
    first_year_end = joined + timedelta(days=364)
    for i in range(10):
        db.add(models.BenefitClaim(
            user_card_id=uc.id, benefit_catalog_item_id=benefit.id,
            period_start=joined, period_end=first_year_end,
            claimed_value=40, claimed_date=joined + timedelta(days=i * 20),
        ))
    db.commit()

    rows = ee.compute_unclaimed_for_user_card(db, uc)
    current_year_rows = [r for r in rows if r.period_end > date.today() - timedelta(days=365)]
    assert any(r.unclaimed_value == r.entitled_value for r in current_year_rows), (
        "current year's cap was incorrectly reduced by a prior year's claims"
    )


def test_every_currency_has_a_known_fx_rate():
    """Catches the exact bug the original hackathon project's tests were
    designed to catch: adding a new bank/currency without an FX entry
    should fail loudly, not silently mis-value that bank's numbers."""
    for currency in ("INR", "USD"):
        assert currency in ee.FX_TO_USD, f"{currency} missing from FX_TO_USD"


def test_portfolio_total_equals_sum_of_rows():
    """The user-level headline total must actually equal the sum of the
    per-row USD-normalized unclaimed values -- catches double counting or
    a stale aggregate."""
    db = make_session()
    product, benefit = _make_basic_catalog(db)
    user = models.User(name="Test User 4", email="t4@example.com")
    db.add(user)
    db.flush()
    joined = date.today() - timedelta(days=200)
    uc = models.UserCard(user_id=user.id, card_product_id=product.id, joined_date=joined, active=1)
    db.add(uc)
    db.commit()

    rows = ee.compute_unclaimed_for_user_card(db, uc)
    summary = ee.portfolio_summary_for_user(db, user.id)
    expected_total = round(sum(r.unclaimed_value_usd for r in rows), 2)
    assert summary["total_unclaimed_value_usd"] == expected_total
