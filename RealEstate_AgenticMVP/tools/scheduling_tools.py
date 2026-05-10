"""
Tour scheduling tools — backed by Cloud SQL.
book_tour includes an idempotency check: if the same email has already booked
the same slot, we return the existing booking rather than creating a duplicate.
"""
import uuid
from sqlalchemy import text
from db.connection import AsyncSessionLocal


async def get_available_slots(property_id: str, preferred_date: str | None = None) -> list[dict]:
    """
    Returns open tour slots for a property.
    If preferred_date (YYYY-MM-DD) is given, returns slots on that day only.
    """
    params: dict = {"property_id": property_id}
    date_filter = ""

    if preferred_date:
        date_filter = "AND slot_datetime::date = :preferred_date"
        params["preferred_date"] = preferred_date

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text(f"""
                SELECT slot_id::text, slot_datetime
                FROM tour_slots
                WHERE property_id = :property_id
                  AND is_booked = false
                  {date_filter}
                ORDER BY slot_datetime
                LIMIT 10
            """),
            params,
        )
        rows = result.mappings().all()

    return [dict(row) for row in rows]


async def book_tour(slot_id: str, prospect_name: str, email: str) -> dict:
    """
    Books a tour slot. Idempotent — if this email already booked this slot
    (e.g. due to a retry), returns the existing booking without creating a duplicate.
    """
    async with AsyncSessionLocal() as db:
        # Check if already booked by this prospect
        existing = await db.execute(
            text("SELECT booking_id::text FROM bookings WHERE slot_id = :slot_id AND email = :email"),
            {"slot_id": slot_id, "email": email},
        )
        row = existing.mappings().first()
        if row:
            return {"booking_id": row["booking_id"], "status": "already_confirmed"}

        # Check the slot is still open before inserting
        slot = await db.execute(
            text("SELECT is_booked FROM tour_slots WHERE slot_id = :slot_id"),
            {"slot_id": slot_id},
        )
        slot_row = slot.mappings().first()
        if not slot_row or slot_row["is_booked"]:
            return {"booking_id": None, "status": "slot_unavailable"}

        booking_id = str(uuid.uuid4())
        await db.execute(
            text("""
                INSERT INTO bookings (booking_id, slot_id, prospect_name, email)
                VALUES (:booking_id, :slot_id, :prospect_name, :email)
            """),
            {"booking_id": booking_id, "slot_id": slot_id,
             "prospect_name": prospect_name, "email": email},
        )
        await db.execute(
            text("UPDATE tour_slots SET is_booked = true WHERE slot_id = :slot_id"),
            {"slot_id": slot_id},
        )
        await db.commit()

    return {"booking_id": booking_id, "status": "confirmed"}


async def cancel_tour(booking_id: str) -> dict:
    """Cancels a booking and frees the slot. Safe to call multiple times."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT slot_id::text FROM bookings WHERE booking_id = :booking_id"),
            {"booking_id": booking_id},
        )
        row = result.mappings().first()
        if not row:
            return {"status": "booking_not_found"}

        slot_id = row["slot_id"]
        await db.execute(
            text("DELETE FROM bookings WHERE booking_id = :booking_id"),
            {"booking_id": booking_id},
        )
        await db.execute(
            text("UPDATE tour_slots SET is_booked = false WHERE slot_id = :slot_id"),
            {"slot_id": slot_id},
        )
        await db.commit()

    return {"status": "cancelled"}
