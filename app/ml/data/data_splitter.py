import logging

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.constants import DEFAULT_TEST_SIZE

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Splits segment data at activity level, preserving temporal order.

    All segments belonging to the same activity stay together — they go
    entirely into train or entirely into val/test, never split across sets.
    """

    def __init__(self, test_ratio: float = DEFAULT_TEST_SIZE):
        self.test_ratio = test_ratio

    def _get_sorted_activity_ids(self, df: pd.DataFrame) -> list:
        """Return activity_ids sorted by their earliest start_date (oldest first)."""
        activity_dates = df.groupby("activity_id")["start_date"].first()
        activity_dates = pd.to_datetime(activity_dates).sort_values()
        return activity_dates.index.tolist()

    def split_train_test(self, df: pd.DataFrame):
        """Split at activity level: older activities → train, most recent → test.

        Returns train_df and test_df where no activity appears in both sets.
        """
        activity_ids = self._get_sorted_activity_ids(df)
        n = len(activity_ids)
        test_size = max(1, int(n * self.test_ratio))

        train_ids = set(activity_ids[: n - test_size])
        test_ids = set(activity_ids[n - test_size :])

        train_df = df[df["activity_id"].isin(train_ids)].reset_index(drop=True)
        test_df = df[df["activity_id"].isin(test_ids)].reset_index(drop=True)

        logger.info(
            "Split: %d train activities (%d segments), %d test activities (%d segments).",
            len(train_ids), len(train_df), len(test_ids), len(test_df),
        )
        return train_df, test_df