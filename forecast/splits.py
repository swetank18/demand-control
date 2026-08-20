"""The one place the evaluation split is defined.

Every trainer, baseline and study imports from here. A split that lives in an
argparse default gets quietly changed by whoever runs the script last, and then
two numbers in the same deck come from two different test sets. So it is frozen
here, it is printed on the slide, and ``tests/test_leakage.py`` asserts the
ordering holds.

The split is *temporal*, never random. Neighbouring 15-minute samples of a
building's load are almost the same number; a random split puts 20:00 in train
and 20:15 in test and every metric you report is then a memory test. That is the
single most common way a load-forecasting evaluation is wrong, and it is the
first thing a reviewer checks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Three contiguous blocks in time order. Test is touched once, at the end."""

    train_start: str = "2016-01-01 00:00"
    train_end: str = "2017-03-31 23:45"
    valid_start: str = "2017-04-01 00:00"
    valid_end: str = "2017-05-31 23:45"
    test_start: str = "2017-06-01 00:00"
    test_end: str = "2017-06-30 23:45"

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"train {self.train_start[:10]} to {self.train_end[:10]}  |  "
            f"valid {self.valid_start[:10]} to {self.valid_end[:10]}  |  "
            f"test {self.test_start[:10]} to {self.test_end[:10]}"
        )

    def mask(self, times: pd.Series, part: str) -> pd.Series:
        """Boolean mask selecting ``part`` in {'train','valid','test'} by target time."""
        t = pd.to_datetime(times)
        lo, hi = {
            "train": (self.train_start, self.train_end),
            "valid": (self.valid_start, self.valid_end),
            "test": (self.test_start, self.test_end),
        }[part]
        return (t >= pd.Timestamp(lo)) & (t <= pd.Timestamp(hi))


SPLIT = Split()

#: Rolling-origin folds for Section 6.2. Each entry is (train_end, valid_end),
#: an expanding training window advancing one month at a time. The spread across
#: folds matters as much as the mean: a forecaster that is excellent in winter
#: and poor in May is not one you want holding a demand ceiling.
ROLLING_FOLDS: tuple[tuple[str, str], ...] = (
    ("2016-09-30 23:45", "2016-10-31 23:45"),
    ("2016-10-31 23:45", "2016-11-30 23:45"),
    ("2016-11-30 23:45", "2016-12-31 23:45"),
    ("2016-12-31 23:45", "2017-01-31 23:45"),
    ("2017-01-31 23:45", "2017-02-28 23:45"),
    ("2017-02-28 23:45", "2017-03-31 23:45"),
    ("2017-03-31 23:45", "2017-04-30 23:45"),
    ("2017-04-30 23:45", "2017-05-31 23:45"),
)

#: Held out for the cold-start study: trained on the other buildings, evaluated
#: here having never seen a row of it. This is the "does it work on a building
#: you have not seen" question, answered rather than asserted.
COLD_START_BUILDING = "Fox_public_Denny"
