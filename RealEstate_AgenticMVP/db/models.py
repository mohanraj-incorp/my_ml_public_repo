from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ARRAY, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Property(Base):
    __tablename__ = "properties"

    property_id   = Column(String, primary_key=True)
    name          = Column(String, nullable=False)
    address       = Column(String)
    city          = Column(String)
    state         = Column(String)
    zip_code      = Column(String)
    bedrooms      = Column(Integer)
    bathrooms     = Column(Float)
    rent          = Column(Float)
    amenities     = Column(ARRAY(String))
    available_from = Column(String)   # ISO date string
    is_available  = Column(Boolean, default=True)
    thumbnail_url = Column(String)    # CDN URL to pre-generated 400x300 thumbnail
    gallery_urls  = Column(ARRAY(String))  # CDN URLs to full gallery


class PolicyDocument(Base):
    """Index of policy docs — actual content lives in Vertex AI Vector Search."""
    __tablename__ = "policy_documents"

    document_id    = Column(String, primary_key=True)
    property_id    = Column(String, nullable=True)   # null means global scope
    policy_type    = Column(String)   # pet_policy | early_termination | parking | building_rules
    scope          = Column(String)   # property | global
    source_file    = Column(String)
    effective_date = Column(String)
    version        = Column(String)


class TourSlot(Base):
    __tablename__ = "tour_slots"

    slot_id       = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    property_id   = Column(String, ForeignKey("properties.property_id"))
    slot_datetime = Column(String)    # ISO datetime string
    is_booked     = Column(Boolean, default=False)


class Booking(Base):
    __tablename__ = "bookings"

    booking_id      = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    slot_id         = Column(UUID(as_uuid=True), ForeignKey("tour_slots.slot_id"))
    prospect_name   = Column(String)
    email           = Column(String)
    created_at      = Column(DateTime, server_default=func.now())
    # Prevents duplicate inserts when tool is retried for the same session + slot
    idempotency_key = Column(String, unique=True)


class Application(Base):
    __tablename__ = "applications"

    application_id  = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    property_id     = Column(String, ForeignKey("properties.property_id"))
    prospect_email  = Column(String)
    prospect_name   = Column(String)
    monthly_income  = Column(Float)
    employment_status = Column(String)
    submitted_at    = Column(DateTime, server_default=func.now())
    status          = Column(String, default="pending")  # pending | approved | conditional | denied
    idempotency_key = Column(String, unique=True)


class Decision(Base):
    """Append-only — no UPDATE or DELETE ever runs on this table."""
    __tablename__ = "decisions"

    decision_id    = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id     = Column(String)
    property_id    = Column(String)
    prospect_email = Column(String)
    outcome        = Column(String)   # approve | conditional_approval | deny
    reasoning      = Column(Text)
    credit_score   = Column(Integer)
    income_ratio   = Column(Float)    # monthly_income / rent
    created_at     = Column(DateTime, server_default=func.now())
