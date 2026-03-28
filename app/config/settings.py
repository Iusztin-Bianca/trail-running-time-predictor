from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Centralized configuration for Trail Running Time Predictor

    IMPORTANT: All sensitive credentials (API keys, tokens, connection strings)
    must be set via environment variables or .env file. Never hardcode them here!
    """

    # Strava API Configuration (REQUIRED - set via environment variables)
    strava_client_id: str = ""
    strava_client_secret: str = ""
    strava_refresh_token: str = ""

    # Azure Storage Configuration (REQUIRED for cloud features)
    azure_storage_connection_string: str = ""
    azure_storage_container_name: str = "training-data"
    azure_parquet_blob_name: str = "strava_training_features.parquet"

    # Pipeline Settings
    strava_activities_per_page: int = 200
    min_elevation_gain_meters: int = 100
    activity_type_filter: str = "Run"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow loading from environment variables as well as .env file
        env_prefix = ""  # No prefix needed, use exact variable names

settings = Settings()
