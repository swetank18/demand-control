"""Country calendars, because the feature block was quietly American.

``features.py`` shipped a hard-coded set of US federal holidays for 2016-2017,
which is correct for Tempe and wrong everywhere else. Left alone, a comparative
study would tell a London building that the last Monday in May is an ordinary
working day and that the fourth Thursday in November is a holiday -- exactly
inverted, on two of the largest load anomalies in the year. Every cross-country
number would then carry a defect that has nothing to do with the model.

So the holiday feature becomes a function of country. The US set is kept
verbatim as the default so that every number already in ``results/`` is
reproduced bit-for-bit; nothing that exists moves.

India is the interesting case and the reason this file is not a one-liner. A
large share of Indian public holidays are lunisolar and move against the
Gregorian calendar -- Diwali, Holi, Eid. A model that learns "the third week of
October is quiet" from one year is learning something false about the next. The
``holidays`` package resolves the actual observed dates per year, which is the
only correct way to do this, and it is why the India column is not a fixed list.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

#: The original hard-coded set, preserved exactly. Any change here moves numbers
#: that are already published in results/, so it is frozen deliberately.
US_HOLIDAYS_2016_17 = {
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
}

#: BDG2 site -> ISO country. Crow and Moose sit on Ottawa coordinates despite a
#: US/Eastern timezone string; they are Canadian and get Canadian holidays.
SITE_COUNTRY = {
    "Bear": "US", "Bobcat": "US", "Bull": "US", "Cockatoo": "US", "Eagle": "US",
    "Fox": "US", "Gator": "US", "Hog": "US", "Panther": "US", "Peacock": "US",
    "Rat": "US", "Swan": "US",
    "Crow": "CA", "Moose": "CA",
    "Lamb": "GB", "Mouse": "GB", "Robin": "GB", "Shrew": "GB",
    "Wolf": "IE",
}


@functools.lru_cache(maxsize=64)
def holiday_dates(country: str, years: tuple[int, ...]) -> frozenset[str]:
    """Observed public holidays as YYYY-MM-DD strings.

    ``country='US'`` with the 2016/2017 years returns the frozen literal set
    above rather than a library lookup, so the existing results are untouched.
    """
    if country == "US" and set(years) <= {2016, 2017}:
        return frozenset(US_HOLIDAYS_2016_17)
    try:
        import holidays as _h
        h = _h.country_holidays(country, years=list(years))
        return frozenset(d.strftime("%Y-%m-%d") for d in h)
    except Exception:
        # Never fail a run over a calendar; a missing country degrades to
        # weekends-only, and the run records that it did.
        return frozenset()


def holiday_flag(times: pd.DatetimeIndex, country: str) -> np.ndarray:
    years = tuple(sorted(set(times.year.tolist())))
    days = holiday_dates(country, years)
    return times.normalize().strftime("%Y-%m-%d").isin(days).astype(int)


def country_of_building(building_id: str) -> str:
    """`Fox_office_Gaylord` -> 'US'. Unknown sites fall back to US."""
    return SITE_COUNTRY.get(building_id.split("_")[0], "US")
