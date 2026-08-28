### China arm — provenance audit

The Chinese provincial series is the only tier-2 source in the study whose documentation describes a construction rather than a measurement. Wu and Kan take the daily maximum and minimum from NDRC reporting, trace a representative workday profile and a representative holiday profile off published load curves with WebPlotDigitizer, and rescale one of those two profiles to each day's envelope. This checks that claim against the data.

| Province | Days | Distinct normalised day-shapes | Days per shape | Numbers needed | Hourly values | Max reconstruction error |
| --- | --- | --- | --- | --- | --- | --- |
| Hainan (Haikou) | 365 | **2** | [115, 250] | 778 | 8760 | 0.000000% of mean |
| Guangdong (Guangzhou) | 365 | **2** | [115, 250] | 778 | 8760 | 0.000000% of mean |
| Shanghai | 365 | **2** | [115, 250] | 778 | 8760 | 0.000000% of mean |
| Beijing | 365 | **2** | [115, 250] | 778 | 8760 | 0.000000% of mean |
| Yunnan (Kunming) | 365 | **2** | [115, 250] | 778 | 8760 | 0.000000% of mean |
| Heilongjiang (Harbin) | 365 | **2** | [115, 250] | 778 | 8760 | 0.000000% of mean |

The documentation is accurate and the construction is exact. A whole year of hourly demand in each province is 778 numbers — 2 normalised 24-hour shapes and one (min, max) pair per day — reproducing all 8760 values to within 1.8e-09 of their magnitude -- the precision at which the source is published, not an approximation we introduced. A compression of 11.26:1.

**What this does to the skill column.** Skill here is a ratio against each series' own seasonal-naive baseline. When every workday shares one shape, seasonal naive recovers that shape exactly and its whole error is the daily envelope; so is ours. Both models are pinned to the same two administrative numbers per day, the ratio collapses toward zero, and a skill figure on this data is a statement about the data rather than about a forecaster. The China skill column is reported for completeness and is not evidence either way about the method.

**What survives.** Coverage is a property of interval width against realised error and does not run through the baseline, so the tier-2 under-coverage result can be read on this arm — and it replicates, on an error process unlike any other in the study. So does the null between forecast skill and climate.
