"""Forecast sources consumed by the controller.

Three implementations, one interface. The oracle and the mean-forecast baseline
are the *same controller* reading a different source, which is what makes the
results table an ablation rather than a beauty contest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim.thermal import PV, clear_sky_ghi, pv_output_kw

QCOLS = {"q05": "q05", "q25": "q25", "q50": "q50", "q75": "q75", "q95": "q95"}


class ForecastSource:
    #: Longest horizon this source can actually produce. The controller clamps its
    #: planning horizon to this rather than silently planning on padded values.
    max_horizon: int = 10**9

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        raise NotImplementedError

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        raise NotImplementedError


class OracleForecast(ForecastSource):
    """Perfect foresight. Every quantile is the truth, so the chance constraint
    collapses to a deterministic one -- which is exactly the point of the row."""

    name = "Perfect foresight oracle"

    def __init__(self, base_actual: np.ndarray, pv_actual: np.ndarray):
        self.b = np.asarray(base_actual, dtype=float)
        self.p = np.asarray(pv_actual, dtype=float)

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.b[k0 : k0 + H]

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.p[k0 : k0 + H]


class TensorForecast(ForecastSource):
    """Reads the precomputed (origin, horizon, quantile) tensor produced by
    ``forecast/train.py``. Base load is exogenous, so precomputing is exact and
    guarantees every controller sees byte-identical forecasts."""

    name = "quantile LightGBM"

    def __init__(
        self,
        tensor: "str | Path | pd.DataFrame",
        index: pd.DatetimeIndex,
        pv_quantiles: "PVQuantiles",
        horizon: int = 64,
    ):
        df = tensor if isinstance(tensor, pd.DataFrame) else pd.read_parquet(tensor)
        self.index = index
        self.H = horizon
        pos = pd.Series(np.arange(len(index)), index=index)
        df = df[df["origin"].isin(index) & df["target_time"].isin(index)]
        self.grid = {}
        for qname in QCOLS:
            arr = np.full((len(index), horizon), np.nan)
            o = pos.reindex(df["origin"]).to_numpy()
            h = df["horizon"].to_numpy().astype(int) - 1
            arr[o, h] = df[qname].to_numpy()
            self.grid[qname] = arr
        self.actual = np.full(len(index), np.nan)
        a = df.drop_duplicates("target_time")
        self.actual[pos.reindex(a["target_time"]).to_numpy()] = a["actual"].to_numpy()
        self.pvq = pv_quantiles
        self.max_horizon = horizon
        self._fallback = pd.Series(self.actual, index=index).ffill().bfill().to_numpy()

    def base(self, k0: int, H: int, q: str) -> np.ndarray:
        row = self.grid[q][k0, :H]
        # early origins have no history yet; fall back to persistence
        if np.isnan(row).any():
            fb = self._fallback[k0 : k0 + H]
            row = np.where(np.isnan(row), fb, row)
        return row

    def pv(self, k0: int, H: int, q: str) -> np.ndarray:
        return self.pvq.forecast(k0, H, q)


class PVQuantiles:
    """PV forecast quantiles derived from the site's own measured cloud record.

    The PV array itself is a design scenario (there is no solar meter at this
    site), but its *uncertainty* is not invented: we take the ratio of
    cloud-attenuated output to clear-sky output across the history, bucket it by
    solar elevation, and read off empirical quantiles. A controller that plans on
    q05 of solar is planning for a cloudy afternoon that the site actually had.
    """

    def __init__(self, index: pd.DatetimeIndex, cloud: pd.Series, t_out: pd.Series, pv_cfg: PV,
                 history_cloud: pd.Series | None = None, history_index: pd.DatetimeIndex | None = None):
        self.index = index
        self.pv_cfg = pv_cfg
        self.actual = pv_output_kw(index, cloud, t_out, pv_cfg).to_numpy()
        clear = pv_output_kw(index, pd.Series(0.0, index=index), t_out, pv_cfg).to_numpy()
        self.clear = clear

        hidx = history_index if history_index is not None else index
        hcloud = history_cloud if history_cloud is not None else cloud
        h_t = pd.Series(25.0, index=hidx)
        h_act = pv_output_kw(hidx, hcloud, h_t, pv_cfg).to_numpy()
        h_clear = pv_output_kw(hidx, pd.Series(0.0, index=hidx), h_t, pv_cfg).to_numpy()

        ok = h_clear > 1e-6
        ratio = np.zeros_like(h_clear)
        ratio[ok] = h_act[ok] / h_clear[ok]
        elev = np.digitize(h_clear / max(h_clear.max(), 1e-9), [0.05, 0.25, 0.5, 0.75])
        self.qmap: dict[str, np.ndarray] = {}
        for qname, qv in [("q05", 0.05), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95)]:
            per_bucket = np.ones(5)
            for bkt in range(5):
                m = ok & (elev == bkt)
                if m.sum() > 20:
                    per_bucket[bkt] = float(np.quantile(ratio[m], qv))
            self.qmap[qname] = per_bucket
        self.elev_now = np.digitize(clear / max(clear.max(), 1e-9), [0.05, 0.25, 0.5, 0.75])

    def forecast(self, k0: int, H: int, q: str) -> np.ndarray:
        if self.pv_cfg.kwp <= 0:
            return np.zeros(H)
        sl = slice(k0, k0 + H)
        return self.clear[sl] * self.qmap[q][self.elev_now[sl]]
