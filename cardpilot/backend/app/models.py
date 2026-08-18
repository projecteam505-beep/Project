"""
Core schema.

Design principle (kept from the earlier design discussion): card_product
(bank's offering) is separate from user_card (a specific user holding it),
which is separate from benefit_catalog_item (what that product entitles
you to). This mirrors how issuers actually model their own systems, and
it's what lets one product be shared by many users without duplicating
its benefit catalog per user.

Currency handling: every bank has a home_currency, every benefit/claim/
transaction carries its native currency. Cross-user or cross-bank totals
must go through the FX-normalized USD columns computed in
entitlement_engine.py, never sum native-currency values directly --
see that module's header comment for why.
"""
from datetime import date, datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from .database import Base


class Bank(Base):
    __tablename__ = "banks"
    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False)   # e.g. "HDFC"
    name = Column(String(100), nullable=False)                # e.g. "HDFC Bank"
    home_currency = Column(String(3), nullable=False)         # e.g. "INR"

    card_products = relationship("CardProduct", back_populates="bank")


class Network(Base):
    __tablename__ = "networks"
    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False)    # VISA / MASTERCARD / RUPAY / AMEX
    name = Column(String(50), nullable=False)

    card_products = relationship("CardProduct", back_populates="network")


class CardProduct(Base):
    """The catalog: e.g. 'HDFC Infinia (Visa)'. One row per real card product,
    shared by every user who holds it."""
    __tablename__ = "card_products"
    id = Column(Integer, primary_key=True)
    bank_id = Column(Integer, ForeignKey("banks.id"), nullable=False)
    network_id = Column(Integer, ForeignKey("networks.id"), nullable=False)
    name = Column(String(100), nullable=False)                # e.g. "Infinia"
    currency = Column(String(3), nullable=False)               # native currency of fees/benefits
    annual_fee = Column(Float, default=0.0)

    bank = relationship("Bank", back_populates="card_products")
    network = relationship("Network", back_populates="card_products")
    benefits = relationship("BenefitCatalogItem", back_populates="card_product")


class BenefitCatalogItem(Base):
    """A single benefit a card product entitles its holder to, e.g.
    'HDFC Infinia -> 8 complimentary lounge visits / quarter'."""
    __tablename__ = "benefit_catalog_items"
    id = Column(Integer, primary_key=True)
    card_product_id = Column(Integer, ForeignKey("card_products.id"), nullable=False)
    benefit_code = Column(String(50), nullable=False)          # e.g. "LOUNGE_ACCESS"
    name = Column(String(150), nullable=False)
    category = Column(String(50), nullable=False)              # credit / lounge / protection / membership / milestone
    cadence = Column(String(20), nullable=False)                # monthly / quarterly / semiannual / annual / per_visit / per_claim
    value = Column(Float, nullable=False)                       # native-currency value per occurrence
    expected_annual_usage = Column(Integer, default=1)          # for per_visit/per_claim benefits
    notes = Column(Text, nullable=True)                         # e.g. source T&C reference, "illustrative, verify vs live T&Cs"

    card_product = relationship("CardProduct", back_populates="benefits")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    cards = relationship("UserCard", back_populates="user")


class UserCard(Base):
    """Links a real user to a card product they actually hold. joined_date
    drives tenure proration in the entitlement engine -- a user who joined
    in June must not be charged unclaimed value for January-May."""
    __tablename__ = "user_cards"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    card_product_id = Column(Integer, ForeignKey("card_products.id"), nullable=False)
    joined_date = Column(Date, nullable=False)
    active = Column(Integer, default=1)  # 1/0 boolean (portable across sqlite/postgres)

    user = relationship("User", back_populates="cards")
    card_product = relationship("CardProduct")
    claims = relationship("BenefitClaim", back_populates="user_card")
    transactions = relationship("Transaction", back_populates="user_card")


class BenefitClaim(Base):
    """The usage ledger: what the user has actually claimed/used, per
    benefit, per period. unclaimed = entitled - claimed, computed by the
    entitlement engine, never stored directly here."""
    __tablename__ = "benefit_claims"
    id = Column(Integer, primary_key=True)
    user_card_id = Column(Integer, ForeignKey("user_cards.id"), nullable=False)
    benefit_catalog_item_id = Column(Integer, ForeignKey("benefit_catalog_items.id"), nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    claimed_value = Column(Float, nullable=False)
    claimed_date = Column(Date, nullable=False)

    user_card = relationship("UserCard", back_populates="claims")
    benefit_catalog_item = relationship("BenefitCatalogItem")


class Transaction(Base):
    """Raw spend, used as a corroborating signal for nudges (e.g. spent at
    a lounge-partner merchant but never claimed lounge access this period).
    In production this table is populated by the Account Aggregator feed
    instead of the synthetic generator -- same schema either way."""
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    user_card_id = Column(Integer, ForeignKey("user_cards.id"), nullable=False)
    date = Column(Date, nullable=False)
    merchant = Column(String(150), nullable=False)
    category = Column(String(50), nullable=True)   # merchant category, used for spend corroboration
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)

    user_card = relationship("UserCard", back_populates="transactions")
