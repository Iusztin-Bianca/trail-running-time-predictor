"""
Strava client using requests for synchronous API calls.
"""
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.strava.com/api/v3"


class StravaClient:
    """Strava API client."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

    def _ensure_valid_token(self) -> None:
        """Refresh access token if expired."""
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                return

        response = requests.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        data = response.json()
        self.access_token = data["access_token"]
        self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
        logger.info("Successfully refreshed Strava access token")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make an authenticated request to the Strava API."""
        self._ensure_valid_token()
        url = f"{BASE_URL}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

    def get_activities(self, page: int = 1, per_page: int = 200, after: Optional[int] = None) -> List[Dict]:
        """Fetch a page of activities from Strava."""
        params: Dict = {"page": page, "per_page": per_page}
        if after:
            params["after"] = after
        return self._make_request("GET", "/athlete/activities", params=params)

    def get_activity_streams(self, activity_id: int) -> Dict:
        """Fetch activity streams (latlng, altitude, time, distance)."""
        params = {
            "keys": "latlng,altitude,time,distance",
            "key_by_type": "true",
        }
        return self._make_request("GET", f"/activities/{activity_id}/streams", params=params)