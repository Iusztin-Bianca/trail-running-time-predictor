"""
Async Strava client using aiohttp for concurrent API calls.
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

BASE_URL = "https://www.strava.com/api/v3"


class StravaClient:
    """Async Strava API client with rate limiting."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        max_concurrent_requests: int = 30
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

        # Rate limiting: max 10 concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Session will be created when entering async context
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        await self._ensure_valid_token()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _ensure_valid_token(self):
        """Refresh access token if expired."""
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                return

        # Refresh token
        async with self.session.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token"
            }
        ) as response:
            data = await response.json()
            self.access_token = data["access_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
            logger.info("Successfully refreshed Strava access token")

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request with rate limiting."""
        await self._ensure_valid_token()

        url = f"{BASE_URL}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        async with self.semaphore:  # Rate limiting
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

    async def get_activities(self, page: int = 1, per_page: int = 200, after: Optional[int] = None) -> List[Dict]:
        """Fetch activities from Strava."""
        params = {"page": page, "per_page": per_page}
        if after:
            params["after"] = after

        return await self._make_request("GET", "/athlete/activities", params=params)

    async def get_activity_streams(self, activity_id: int) -> Dict:
        """Fetch activity streams (latlng, altitude, time, distance)."""
        endpoint = f"/activities/{activity_id}/streams"
        params = {
            "keys": "latlng,altitude,time,distance",
            "key_by_type": "true"
        }

        return await self._make_request("GET", endpoint, params=params)

    async def fetch_all_activities(
        self,
        min_elevation_m: float = 150.0,
        min_distance_m: float = 4000.0,
        after: Optional[int] = None
    ) -> List[Dict]:
        """Fetch all activities matching criteria."""
        all_activities = []
        page = 1
        per_page = 200

        while True:
            activities = await self.get_activities(page=page, per_page=per_page, after=after)

            if not activities:
                break

            # Filter
            filtered = [
                activity for activity in activities
                if activity.get("type") in ["Run", "TrailRun"]
                and activity.get("total_elevation_gain", 0) >= min_elevation_m
                and activity.get("distance", 0) >= min_distance_m
            ]

            all_activities.extend(filtered)

            if len(activities) < per_page:
                break

            page += 1

        logger.info(f"Fetched {len(all_activities)} activities with elevation >= {min_elevation_m}m")
        return all_activities
