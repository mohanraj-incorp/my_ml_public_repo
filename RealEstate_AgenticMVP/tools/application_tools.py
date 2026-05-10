"""
Application submission tools.
submit_application is idempotent: a second call for the same session
returns the existing application rather than creating a duplicate.
"""
import uuid
from sqlalchemy import text
from db.connection import AsyncSessionLocal
from google.cloud import storage
from config.settings import settings


async def submit_application(
    session_id: str,
    property_id: str,
    prospect_name: str,
    prospect_email: str,
    monthly_income: float,
    employment_status: str,
) -> dict:
    """
    Writes the application to Cloud SQL.
    Uses session_id as the idempotency key — retrying the same session
    returns the existing application_id, not a new row.
    """
    async with AsyncSessionLocal() as db:
        # Idempotency check: one application per session
        existing = await db.execute(
            text("SELECT application_id::text FROM applications WHERE idempotency_key = :key"),
            {"key": session_id},
        )
        row = existing.mappings().first()
        if row:
            return {"application_id": row["application_id"], "status": "already_submitted"}

        application_id = str(uuid.uuid4())
        await db.execute(
            text("""
                INSERT INTO applications
                    (application_id, property_id, prospect_email, prospect_name,
                     monthly_income, employment_status, status, idempotency_key)
                VALUES
                    (:application_id, :property_id, :prospect_email, :prospect_name,
                     :monthly_income, :employment_status, 'pending', :idempotency_key)
            """),
            {
                "application_id": application_id,
                "property_id": property_id,
                "prospect_email": prospect_email,
                "prospect_name": prospect_name,
                "monthly_income": monthly_income,
                "employment_status": employment_status,
                "idempotency_key": session_id,
            },
        )
        await db.commit()

    return {"application_id": application_id, "status": "submitted"}


async def upload_document(file_bytes: bytes, filename: str, property_id: str, prospect_email: str) -> dict:
    """
    Uploads a supporting document (pay stub, ID, etc.) to GCS.
    Path: gs://{bucket}/{property_id}/{prospect_email}/{filename}
    """
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(settings.gcs_properties_bucket)

    blob_path = f"applications/{property_id}/{prospect_email}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(file_bytes)

    return {"gcs_path": f"gs://{settings.gcs_properties_bucket}/{blob_path}", "status": "uploaded"}
