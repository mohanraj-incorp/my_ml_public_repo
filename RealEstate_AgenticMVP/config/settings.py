from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"

    gcp_project_id: str
    gcp_region: str = "us-central1"

    cloud_sql_instance: str
    db_name: str = "leasing"
    db_user: str = "leasing_app"
    db_password: str

    redis_host: str
    redis_port: int = 6379

    vector_search_index_endpoint: str
    vector_search_deployed_index_id: str

    gcs_properties_bucket: str
    gcs_policy_docs_bucket: str
    gcs_golden_dataset_bucket: str

    firestore_collection_prospects: str = "prospects"

    max_recursion_limit: int = 25
    max_session_turns: int = 50
    conversation_summary_threshold: int = 12


settings = Settings()
