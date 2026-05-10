"""
Property search tools — backed by Cloud SQL (PostgreSQL).
These are read-only queries so no idempotency handling is needed.
"""
from sqlalchemy import text
from db.connection import AsyncSessionLocal


async def search_properties(
    bedrooms: int | None = None,
    rent_max: float | None = None,
    zip_codes: list[str] | None = None,
    amenities: list[str] | None = None,
    move_in_date: str | None = None,
) -> list[dict]:
    """
    Returns up to 10 available properties matching the prospect's criteria.
    Each result includes a thumbnail_url so the frontend can render a card grid.
    """
    filters = ["is_available = true"]
    params: dict = {}

    if bedrooms:
        filters.append("bedrooms = :bedrooms")
        params["bedrooms"] = bedrooms

    if rent_max:
        filters.append("rent <= :rent_max")
        params["rent_max"] = rent_max

    if zip_codes:
        filters.append("zip_code = ANY(:zip_codes)")
        params["zip_codes"] = zip_codes

    if amenities:
        # Property must have ALL requested amenities
        filters.append("amenities @> :amenities")
        params["amenities"] = amenities

    if move_in_date:
        filters.append("available_from <= :move_in_date")
        params["move_in_date"] = move_in_date

    where_clause = " AND ".join(filters)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text(f"""
                SELECT property_id, name, city, state, zip_code,
                       bedrooms, bathrooms, rent, amenities,
                       available_from, thumbnail_url
                FROM properties
                WHERE {where_clause}
                ORDER BY rent ASC
                LIMIT 10
            """),
            params,
        )
        rows = result.mappings().all()

    return [dict(row) for row in rows]


async def get_unit_details(property_id: str) -> dict | None:
    """
    Returns full property details including the gallery photo URLs.
    Called when the prospect selects a specific property from the list.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("""
                SELECT property_id, name, address, city, state, zip_code,
                       bedrooms, bathrooms, rent, amenities,
                       available_from, thumbnail_url, gallery_urls
                FROM properties
                WHERE property_id = :property_id
            """),
            {"property_id": property_id},
        )
        row = result.mappings().first()

    return dict(row) if row else None
