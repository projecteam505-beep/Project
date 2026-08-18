"""
Seed script: loads the real (illustrative) bank/network/card/benefit
catalog from seed_banks_india.json, then generates a handful of demo
users with cards, transactions, and partial claims -- enough to exercise
the entitlement engine and nudge engine end to end.

Run: python3 -m data.seed_data   (from the backend/ directory)
"""
import json
import os
import random
from datetime import date, timedelta

from app.database import Base, engine, SessionLocal
from app import models

random.seed(7)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_catalog(db):
    with open(os.path.join(DATA_DIR, "seed_banks_india.json")) as f:
        data = json.load(f)

    banks = {}
    for b in data["banks"]:
        bank = models.Bank(code=b["code"], name=b["name"], home_currency=b["home_currency"])
        db.add(bank)
        banks[b["code"]] = bank

    networks = {}
    for n in data["networks"]:
        network = models.Network(code=n["code"], name=n["name"])
        db.add(network)
        networks[n["code"]] = network

    db.flush()  # assign IDs before referencing them

    products = []
    for cp in data["card_products"]:
        product = models.CardProduct(
            bank_id=banks[cp["bank_code"]].id,
            network_id=networks[cp["network_code"]].id,
            name=cp["name"],
            currency=cp["currency"],
            annual_fee=cp["annual_fee"],
        )
        db.add(product)
        db.flush()
        for ben in cp["benefits"]:
            db.add(models.BenefitCatalogItem(
                card_product_id=product.id,
                benefit_code=ben["benefit_code"],
                name=ben["name"],
                category=ben["category"],
                cadence=ben["cadence"],
                value=ben["value"],
                expected_annual_usage=ben["expected_annual_usage"],
                notes=ben.get("notes"),
            ))
        products.append(product)

    db.commit()
    return products


DEMO_USERS = [
    ("Ananya Rao", "ananya.rao@example.com"),
    ("Vikram Mehta", "vikram.mehta@example.com"),
    ("Sara Thomas", "sara.thomas@example.com"),
]

MERCHANT_CATEGORIES = ["airport", "restaurant", "cinema", "fuel", "grocery", "online_retail"]


def generate_demo_users(db, products):
    today = date.today()
    for name, email in DEMO_USERS:
        user = models.User(name=name, email=email)
        db.add(user)
        db.flush()

        # each demo user holds 1-2 random cards, joined between 3 and 18 months ago
        for product in random.sample(products, k=random.randint(1, 2)):
            joined = today - timedelta(days=random.randint(90, 540))
            user_card = models.UserCard(
                user_id=user.id, card_product_id=product.id,
                joined_date=joined, active=1,
            )
            db.add(user_card)
            db.flush()

            # random transactions over the tenure
            for _ in range(random.randint(15, 40)):
                txn_date = joined + timedelta(days=random.randint(0, (today - joined).days))
                db.add(models.Transaction(
                    user_card_id=user_card.id,
                    date=txn_date,
                    merchant=random.choice(["Indigo Airlines", "PVR Cinemas", "Shell", "Zomato", "Amazon"]),
                    category=random.choice(MERCHANT_CATEGORIES),
                    amount=round(random.uniform(200, 5000), 2),
                    currency=product.currency,
                ))

            # partial, realistic claims -- not every benefit gets claimed,
            # which is exactly the gap this whole system exists to surface
            for benefit in product.benefits:
                if random.random() < 0.4:  # ~40% of periods get claimed
                    claim_date = joined + timedelta(days=random.randint(10, 60))
                    db.add(models.BenefitClaim(
                        user_card_id=user_card.id,
                        benefit_catalog_item_id=benefit.id,
                        period_start=joined,
                        period_end=min(today, joined + timedelta(days=30)),
                        claimed_value=benefit.value,
                        claimed_date=claim_date,
                    ))
    db.commit()


def main():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        products = load_catalog(db)
        generate_demo_users(db, products)
        print(f"Seeded {len(products)} card products across "
              f"{db.query(models.Bank).count()} banks, "
              f"{db.query(models.User).count()} demo users.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
