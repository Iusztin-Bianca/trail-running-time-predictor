"""
Azure Blob Storage helper for managing parquet files and raw activity data.
"""
import json
import logging
import pandas as pd
from io import BytesIO
from typing import Optional, Dict, Any
from azure.storage.blob import BlobServiceClient, BlobClient
from app.config.settings import settings

logger = logging.getLogger(__name__)


class BlobStorageManager:
    """Manages Azure Blob Storage operations for training data."""

    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None, blob_name: Optional[str] = None):
        """
        Initialize Blob Storage Manager.

        Args:
            connection_string: Azure Storage connection string
            container_name: Blob container name
            blob_name: Parquet blob name (default: from settings)
        """
        self.connection_string = connection_string or settings.azure_storage_connection_string
        self.container_name = container_name or settings.azure_storage_container_name
        self.blob_name = blob_name or settings.azure_parquet_blob_name

        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

        # Create container if it doesn't exist
        try:
            self.container_client.create_container()
            logger.info(f"Created container: {self.container_name}")
        except Exception:
            logger.debug(f"Container {self.container_name} already exists")

    def upload_parquet(self, df: pd.DataFrame, overwrite: bool = True) -> None:
        """
        Upload DataFrame as parquet to blob storage.

        Args:
            df: DataFrame to upload
            overwrite: Whether to overwrite existing blob (default: True)
        """
        try:
            # Convert DataFrame to parquet bytes
            buffer = BytesIO()
            df.to_parquet(buffer, index=False, engine='pyarrow')
            buffer.seek(0)

            # Upload to blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.blob_name
            )
            blob_client.upload_blob(buffer, overwrite=overwrite)

            logger.info(f"Uploaded parquet to {self.container_name}/{self.blob_name} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Failed to upload parquet to blob storage: {e}")
            raise

    def download_parquet(self) -> Optional[pd.DataFrame]:
        """
        Download parquet file from blob storage as DataFrame.

        Returns:
            DataFrame if blob exists, None otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.blob_name
            )

            # Check if blob exists
            if not blob_client.exists():
                logger.warning(f"Blob {self.blob_name} does not exist")
                return None

            # Download blob data
            blob_data = blob_client.download_blob()
            buffer = BytesIO(blob_data.readall())

            # Read parquet
            df = pd.read_parquet(buffer, engine='pyarrow')
            df["start_date"] = pd.to_datetime(df["start_date"])
            df = df.sort_values("start_date").reset_index(drop=True)
            logger.info(f"Downloaded parquet from {self.container_name}/{self.blob_name}")

            return df

        except Exception as e:
            logger.error(f"Failed to download parquet from blob storage: {e}")
            raise

    def get_last_activity_timestamp(self) -> Optional[int]:
        """
        Get the timestamp of the most recent activity in the parquet file.

        Returns:
            Unix timestamp of most recent activity, or None if no data exists
        """
        try:
            df = self.download_parquet()

            if df is None or df.empty:
                return None

            # Parse start_date column and get maximum
            df['start_date_parsed'] = pd.to_datetime(df['start_date'])
            max_date = df['start_date_parsed'].max()

            # Convert to Unix timestamp
            timestamp = int(max_date.timestamp())
            logger.info(f"Last activity timestamp: {timestamp} ({max_date})")

            return timestamp

        except Exception as e:
            logger.error(f"Failed to get last activity timestamp: {e}")
            return None

    def append_and_upload(self, new_df: pd.DataFrame) -> None:
        """
        Append new data to existing parquet and upload.

        Args:
            new_df: DataFrame with new activities to append
        """
        try:
            # Download existing data
            existing_df = self.download_parquet()

            if existing_df is None or existing_df.empty:
                # No existing data, just upload new data
                logger.info("No existing data found, uploading new data")
                self.upload_parquet(new_df, overwrite=True)
                return

            # Normalize start_date to datetime in both DataFrames before concat
            existing_df["start_date"] = pd.to_datetime(existing_df["start_date"], utc=True)
            new_df = new_df.copy()
            new_df["start_date"] = pd.to_datetime(new_df["start_date"], utc=True)

            # Remove existing segments for activities that are in new_df (handles re-runs)
            new_activity_ids = set(new_df["activity_id"].unique())
            existing_df = existing_df[~existing_df["activity_id"].isin(new_activity_ids)]

            # Append new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Sort by start_date
            combined_df['start_date_parsed'] = pd.to_datetime(combined_df['start_date'])
            combined_df = combined_df.sort_values('start_date_parsed')
            combined_df = combined_df.drop(columns=['start_date_parsed'])

            # Upload combined data
            self.upload_parquet(combined_df, overwrite=True)

            logger.info(
                f"Appended {len(new_df)} new activities. "
                f"Total activities: {len(combined_df)} "
                f"(added {len(combined_df) - len(existing_df)} net new)"
            )

        except Exception as e:
            logger.error(f"Failed to append and upload data: {e}")
            raise

    # ========== RAW ACTIVITY DATA METHODS ==========

    def _get_raw_activity_blob_name(self, activity_id: int) -> str:
        """
        Get blob name for raw activity data.

        Args:
            activity_id: Strava activity ID

        Returns:
            Blob path: raw/activities/{activity_id}.json
        """
        return f"raw/activities/{activity_id}.json"

    def upload_raw_activity(
        self,
        activity_id: int,
        activity_metadata: Dict[str, Any],
        streams: Dict[str, Any],
        overwrite: bool = False
    ) -> bool:
        """
        Upload raw activity data (metadata + streams) as JSON to blob storage.

        Args:
            activity_id: Strava activity ID
            activity_metadata: Activity metadata from Strava API (name, start_date, etc.)
            streams: Raw streams data from Strava API (latlng, altitude, time)
            overwrite: Whether to overwrite existing blob (default: False)

        Returns:
            True if uploaded successfully, False if skipped (already exists)
        """
        blob_name = self._get_raw_activity_blob_name(activity_id)

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            # Check if already exists (skip if not overwriting)
            if not overwrite and blob_client.exists():
                logger.debug(f"Raw activity {activity_id} already exists, skipping upload")
                return False

            # Build raw activity document
            raw_activity = {
                "activity_id": activity_id,
                "activity_name": activity_metadata.get("name", "Unknown"),
                "start_date": activity_metadata.get("start_date"),
                "workout_type": activity_metadata.get("workout_type"),
                "total_elevation_gain": activity_metadata.get("total_elevation_gain"),
                "distance": activity_metadata.get("distance"),
                "moving_time": activity_metadata.get("moving_time"),
                "elapsed_time": activity_metadata.get("elapsed_time"),
                "type": activity_metadata.get("type"),
                "streams": streams
            }

            # Convert to JSON bytes
            json_bytes = json.dumps(raw_activity, indent=2).encode('utf-8')

            # Upload
            blob_client.upload_blob(json_bytes, overwrite=overwrite)
            logger.info(f"Uploaded raw activity {activity_id} to {self.container_name}/{blob_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to upload raw activity {activity_id}: {e}")
            raise

    def download_raw_activity(self, activity_id: int) -> Optional[Dict[str, Any]]:
        """
        Download raw activity data from blob storage.

        Args:
            activity_id: Strava activity ID

        Returns:
            Raw activity dict if exists, None otherwise
        """
        blob_name = self._get_raw_activity_blob_name(activity_id)

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            # Download directly - catches 404 without a separate HEAD request
            blob_data = blob_client.download_blob()
            json_bytes = blob_data.readall()
            raw_activity = json.loads(json_bytes.decode('utf-8'))

            logger.debug(f"Downloaded raw activity {activity_id}")
            return raw_activity

        except Exception as e:
            if "BlobNotFound" in str(e) or "404" in str(e):
                logger.debug(f"Raw activity {activity_id} not found in blob storage")
                return None
            logger.error(f"Failed to download raw activity {activity_id}: {e}")
            return None

    def raw_activity_exists(self, activity_id: int) -> bool:
        """
        Check if raw activity data exists in blob storage.

        Args:
            activity_id: Strava activity ID

        Returns:
            True if exists, False otherwise
        """
        blob_name = self._get_raw_activity_blob_name(activity_id)

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Failed to check raw activity existence: {e}")
            return False

    # ========== MODEL VERSIONING METHODS ==========
    # Models are stored as:
    #   models/model_v1.joblib, models/model_v2.joblib, ...
    #   models/model_v{n}_metadata.json  (per-version metrics + info)
    #   models/latest.json               (pointer to current version)

    _MODELS_PREFIX = "models"
    _LATEST_BLOB_NAME = "models/latest.json"

    def _model_blob_name(self, version: int) -> str:
        return f"{self._MODELS_PREFIX}/model_v{version}.joblib"

    def _model_metadata_blob_name(self, version: int) -> str:
        return f"{self._MODELS_PREFIX}/model_v{version}_metadata.json"

    def _get_latest_version_info(self) -> Optional[dict]:
        """Read models/latest.json. Returns dict or None if no model saved yet."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=self._LATEST_BLOB_NAME
            )
            if not blob_client.exists():
                return None
            blob_data = blob_client.download_blob()
            return json.loads(blob_data.readall().decode("utf-8"))
        except Exception as e:
            logger.error("Failed to read latest model info: %s", e)
            return None

    def _set_latest_version_info(self, info: dict) -> None:
        """Write models/latest.json."""
        json_bytes = json.dumps(info, indent=2).encode("utf-8")
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=self._LATEST_BLOB_NAME
        )
        blob_client.upload_blob(json_bytes, overwrite=True)

    def upload_model(self, model, model_name: str, metrics: dict = None) -> int:
        """
        Serialize and upload a new versioned model to blob storage.

        Each call creates a new version (model_v1, model_v2, ...) and updates
        models/latest.json to point to it.

        Args:
            model: Fitted model object (BaseModel subclass)
            model_name: Model type name (e.g. "ridge", "xgboost")
            metrics: Optional dict of evaluation metrics

        Returns:
            The new version number
        """
        import joblib
        from datetime import datetime as _dt

        try:
            # Determine next version number
            latest = self._get_latest_version_info()
            next_version = (latest["version"] + 1) if latest else 1

            # Upload versioned model blob
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)

            model_blob = self._model_blob_name(next_version)
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=model_blob
            )
            blob_client.upload_blob(buffer, overwrite=True)

            # Upload per-version metadata
            saved_at = _dt.now().isoformat()
            version_metadata = {
                "version": next_version,
                "model_name": model_name,
                "saved_at": saved_at,
                "metrics": metrics or {},
            }
            meta_blob = self._model_metadata_blob_name(next_version)
            blob_client_meta = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=meta_blob
            )
            blob_client_meta.upload_blob(
                json.dumps(version_metadata, indent=2).encode("utf-8"), overwrite=True
            )

            # Update latest pointer
            self._set_latest_version_info({
                "version": next_version,
                "blob": model_blob,
                "metadata_blob": meta_blob,
                "model_name": model_name,
                "saved_at": saved_at,
                "metrics": metrics or {},
            })

            logger.info(
                "Saved model '%s' as version v%d → %s",
                model_name, next_version, model_blob,
            )
            return next_version

        except Exception as e:
            logger.error("Failed to upload model to blob storage: %s", e)
            raise

    def download_model(self):
        """
        Download and deserialize the latest versioned model from blob storage.

        Returns:
            Fitted model object

        Raises:
            FileNotFoundError: if no model has been saved yet
        """
        import joblib

        try:
            latest = self._get_latest_version_info()
            if latest is None:
                raise FileNotFoundError("No model found in blob storage (models/latest.json missing).")

            model_blob = latest["blob"]
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=model_blob
            )
            blob_data = blob_client.download_blob()
            buffer = BytesIO(blob_data.readall())
            model = joblib.load(buffer)

            logger.info(
                "Downloaded model v%d ('%s') from Blob Storage",
                latest["version"], latest.get("model_name", "?"),
            )
            return model

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to download model from blob storage: %s", e)
            raise

    def model_exists(self) -> bool:
        """Check if at least one saved model exists (models/latest.json present)."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=self._LATEST_BLOB_NAME
            )
            return blob_client.exists()
        except Exception as e:
            logger.error("Failed to check model existence: %s", e)
            return False

    def download_model_metadata(self) -> Optional[dict]:
        """
        Return the latest.json info dict (version, model_name, saved_at, metrics).

        Returns:
            Metadata dict or None if no model saved yet
        """
        return self._get_latest_version_info()