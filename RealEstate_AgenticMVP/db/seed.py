"""
Run once to populate Cloud SQL with sample properties and tour slots for the POC.
Usage: python -m db.seed
"""
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import text
from db.connection import engine
from db.models import Base, Property, TourSlot


SAMPLE_PROPERTIES = [
    {"property_id": "PROP_001", "name": "Sunset Lofts", "city": "Dallas", "state": "TX",
     "zip_code": "75201", "bedrooms": 2, "bathrooms": 2.0, "rent": 1850.0,
     "amenities": ["pool", "gym", "parking"], "available_from": "2026-06-01",
     "thumbnail_url": "https://cdn.example.com/PROP_001/thumbnail.jpg",
     "gallery_urls": ["https://cdn.example.com/PROP_001/gallery/1.jpg"]},

    {"property_id": "PROP_002", "name": "Riverside Commons", "city": "Austin", "state": "TX",
     "zip_code": "78701", "bedrooms": 1, "bathrooms": 1.0, "rent": 1400.0,
     "amenities": ["gym", "rooftop"], "available_from": "2026-05-15",
     "thumbnail_url": "https://cdn.example.com/PROP_002/thumbnail.jpg",
     "gallery_urls": ["https://cdn.example.com/PROP_002/gallery/1.jpg"]},

    {"property_id": "PROP_003", "name": "Midtown Heights", "city": "Dallas", "state": "TX",
     "zip_code": "75202", "bedrooms": 3, "bathrooms": 2.0, "rent": 2400.0,
     "amenities": ["pool", "gym", "parking", "concierge"], "available_from": "2026-06-15",
     "thumbnail_url": "https://cdn.example.com/PROP_003/thumbnail.jpg",
     "gallery_urls": []},

    {"property_id": "PROP_004", "name": "Greenway Flats", "city": "Houston", "state": "TX",
     "zip_code": "77002", "bedrooms": 2, "bathrooms": 1.0, "rent": 1600.0,
     "amenities": ["parking", "laundry"], "available_from": "2026-05-20",
     "thumbnail_url": "https://cdn.example.com/PROP_004/thumbnail.jpg",
     "gallery_urls": []},

    {"property_id": "PROP_005", "name": "The Reserve at Oak Lawn", "city": "Dallas", "state": "TX",
     "zip_code": "75219", "bedrooms": 1, "bathrooms": 1.0, "rent": 1250.0,
     "amenities": ["pool", "dog_park"], "available_from": "2026-06-01",
     "thumbnail_url": "https://cdn.example.com/PROP_005/thumbnail.jpg",
     "gallery_urls": []},
]


async def seed():
    async with engine.begin() as conn:
        # Create all tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)

        # Insert properties (skip if already present)
        for p in SAMPLE_PROPERTIES:
            await conn.execute(
                text("""
                    INSERT INTO properties (property_id, name, city, state, zip_code,
                        bedrooms, bathrooms, rent, amenities, available_from,
                        is_available, thumbnail_url, gallery_urls)
                    VALUES (:property_id, :name, :city, :state, :zip_code,
                        :bedrooms, :bathrooms, :rent, :amenities, :available_from,
                        true, :thumbnail_url, :gallery_urls)
                    ON CONFLICT (property_id) DO NOTHING
                """),
                {**p, "amenities": p["amenities"], "gallery_urls": p["gallery_urls"]},
            )

        # Seed tour slots — 3 slots per property for the next 7 days
        slot_times = ["10:00", "13:00", "16:00"]
        for prop in SAMPLE_PROPERTIES:
            for day_offset in range(1, 8):
                slot_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                for time in slot_times:
                    await conn.execute(
                        text("""
                            INSERT INTO tour_slots (property_id, slot_datetime, is_booked)
                            VALUES (:property_id, :slot_datetime, false)
                            ON CONFLICT DO NOTHING
                        """),
                        {"property_id": prop["property_id"],
                         "slot_datetime": f"{slot_date}T{time}:00"},
                    )

    print(f"Seeded {len(SAMPLE_PROPERTIES)} properties and tour slots.")


if __name__ == "__main__":
    asyncio.run(seed())
