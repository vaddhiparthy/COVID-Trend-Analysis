## COVID Data Analysis — Technical Overview

### Objective

Build a unified, analysis-ready dataset that combines:
- COVID-19 cases and deaths (daily, by state)
- Vaccination progress (doses, series completion, boosters)
- Hospital and ICU capacity and utilization

Then:
- Aggregate to a **weekly state-level fact table** (`df_hvc`) for exploratory analysis and visualization.
- Preserve a **daily national time series** for ARIMA-based forecasting of new cases.

---

### Data Sources

1. **Vaccinations (CDC)**
   - Dataset: “COVID-19 Vaccinations in the United States, Jurisdiction”
   - Raw file: `Vaccine.csv` (≈29,784 rows, 82 columns → weekly ≈4,401 rows)
   - Key fields: distributed and administered doses by manufacturer (Pfizer, Moderna, Janssen), dose1/series completion percentages, booster/additional doses.
   - Location field: 2-letter state / jurisdiction codes (e.g., `NC`, `MI`, `BP2`, `LTC`).

2. **Cases and Deaths (CDC)**
   - Dataset: “United States COVID-19 Cases and Deaths by State over Time”
   - Raw file: `CND.csv` (≈47,220 rows, 15 columns)
   - Fields used: `submission_date`, `state`, `tot_cases`, `new_case`, `tot_death`, `new_death`.
   - Additional derived: `week` (ISO week), `uid` (state + year + week).

3. **Hospital Capacity (HHS)**
   - Dataset: “COVID-19 Reported Patient Impact and Hospital Capacity”
   - Raw file: `Hospitals.csv` (≈419,754 rows, 109+ columns)
   - Fields aggregated: 
     - `total_beds_7_day_sum`
     - `total_icu_beds_7_day_sum`
     - `inpatient_beds_used_7_day_sum`
     - `icu_beds_used_7_day_sum`
   - Granularity: hospital-week, later aggregated to state-week.

4. **State Lat/Long**
   - File: `statelatlong.csv`
   - Provides `State`, `City`, `Latitude`, `Longitude` for geo-enrichment.

---

### Core Engineering Pattern — Unified Weekly UID Key

Across all three datasets, a **canonical weekly key** is engineered:

- Convert date to `datetime`.
- Extract epidemiological **week number** with `dt.week` → padded to two digits:
  - `week = Series.dt.week.astype(str).str.pad(2, side='left', fillchar='0')`
- Derive **UID**:
  - `uid = state_code + year + week`  
    e.g., `CO202128` for Colorado, 2021, week 28.

This UID allows consistent joining of:
- Vaccination data (`df_vaccine`)
- Case/death data (`df_cases`)
- Hospital capacity data (`df_icu`)

into a **single state–week grain** dataset.

---

### Vaccination Pipeline (`df_vaccine`)

1. **Load + inspect**
   - `df_vaccine = pd.read_csv("Vaccine.csv")`
   - 29,784 daily rows → 82 columns.

2. **Type normalization**
   - `Date` → `datetime64[ns]`.
   - Generate `week` (two-digit string) from `Date`.
   - Create `uid = Location + year + week`.

3. **Deduplicate to weekly resolution**
   - Sort by `Date` descending so the latest daily record in a week survives:
     ```python
     df_vaccine = (
       df_vaccine
       .sort_values('Date', ascending=False)
       .drop_duplicates('uid')
       .sort_index()
     )
     ```
   - Result: 4,401 unique `uid` rows (state-week).

4. **Null handling**
   - `df_vaccine = df_vaccine.fillna(0)` and verification of zero null-rows.

5. **Schema selection + renaming**
   - Lowercase columns and keep only analysis-critical vaccination fields:
     - `uid`, `state`, `date`, `mmwr_week`, `week`
     - Manufacturer-wise distributed and series-complete counts
     - Aggregate `distributed`, `dist_per_100k`
     - `administered`, `admin_per_100k`
     - Dose1 recipients & % (`administered_dose1_recip`, `%`)
     - `series_complete_yes`, `series_complete_pop_pct`
     - Booster/additional doses (+ manufacturer breakdown).
   - `Location` → `state` for consistency.

6. **Export**
   - `df_vaccine.to_csv("COVID_UID_Vaccine.csv")`

---

### Cases/Deaths Pipeline (`df_cases`, `df_cases_daily`, `df_ts`)

1. **Load + base cleaning**
   - `df_cases = pd.read_csv("CND.csv")`
   - Convert `submission_date` to `datetime`.
   - Derive **weekly key**: `week` + `uid` (identical pattern to vaccines).

2. **Column pruning**
   - Keep only fully-populated, non-null-core fields:
     - `submission_date`, `state`, `tot_cases`, `new_case`, `tot_death`, `new_death`, `week`.

3. **Data quality: negative values**
   - Remove records where `new_case < 0` or `new_death < 0`.
   - Ensures monotonic consistency of cumulative counts.

4. **Daily analytical copy (`df_cases_daily`)**
   - Rename for clarity and time-series modeling:
     - `submission_date` → `date`
     - `tot_cases` → `cases_total`
     - `new_case` → `cases_new`
     - `tot_death` → `death_total`
     - `new_death` → `death_new`
   - Add `uid` and `week`; reset index.
   - This preserves **per-state daily granularity** for future modeling.

5. **National aggregate daily time series (`df_ts`)**
   - Group by `date` and sum across all states:
     ```python
     df_ts = df_cases_daily.groupby('date').sum()
     ```
   - Columns: `cases_total`, `cases_new`, `death_total`, `death_new`.
   - Later enriched with ARIMA forecasts:
     - `forecast_ARIMA` appended as 5th/6th column.

6. **Weekly state-level cases table (`df_cases`)**
   - Collapse to unique `uid` per state-week using latest `submission_date`:
     - sort desc by date, `drop_duplicates('uid')`.
   - Keep:
     - `uid`, `cases_total`, `cases_new`, `death_total`, `death_new`.

---

### Hospital Capacity Pipeline (`df_hospitals`, `df_icu`)

1. **Load**
   - `df_hospitals = pd.read_csv("Hospitals.csv")`
   - Granularity: hospital-week.

2. **Week and UID**
   - `collection_week` → `datetime` → `week` → `uid = state + year + week`.

3. **Null and negative filtering**
   - Focus on key capacity metrics:
     - `total_beds_7_day_sum`
     - `total_icu_beds_7_day_sum`
     - `inpatient_beds_used_7_day_sum`
     - `icu_beds_used_7_day_sum`
   - Drop rows with nulls in these columns.
   - Drop rows with negative values in any of these metrics.

4. **State-week aggregation (`df_icu`)**
   - Pivot / group by `uid` with `np.sum`:
     ```python
     df_icu = pd.pivot_table(
       df_hospitals,
       index=['uid'],
       aggfunc={
         'total_beds_7_day_sum': np.sum,
         'total_icu_beds_7_day_sum': np.sum,
         'inpatient_beds_used_7_day_sum': np.sum,
         'icu_beds_used_7_day_sum': np.sum
       }
     )
     ```
   - Rename:
     - `total_beds_7_day_sum` → `total_beds`
     - `total_icu_beds_7_day_sum` → `total_icu`
     - `inpatient_beds_used_7_day_sum` → `used_inpatient`
     - `icu_beds_used_7_day_sum` → `used_icu`
   - Result: ≈4,625 rows (state-week ICU summary).

---

### Master Analytic Mart (`df_hvc`)

1. **Join vaccines + cases → `df_vc`**
   - Merge on `uid`:
     ```python
     df_vc = pd.merge(df_vaccine, df_cases, on='uid')
     ```
   - Result: ~3,819 rows (state-week with cases/deaths + vaccines).

2. **Add ICU capacity → `df_hvc`**
   - Merge `df_vc` with `df_icu` on `uid`:
     ```python
     df_hvc = pd.merge(df_vc, df_icu, on='uid')
     ```
   - Add state metadata by merging with `statelatlong.csv`:
     - Brings `city` (used as state name index), `latitude`, `longitude`.
   - Set index to readable state label.

3. **Final schema (selected columns)**
   - Dimensions:
     - `state` (index; from city/state names)
     - `stat_abr` (2-letter code)
     - `date`, `mmwr_week`, `week`, `uid`, `w_id` (year+week)
     - `latitude`, `longitude`
   - Case/death facts:
     - `cases_total`, `cases_new`, `death_total`, `death_new`
   - Hospital capacity:
     - `total_beds`, `total_icu`, `used_inpatient`, `used_icu`
     - Derived: `pct_icu_occ = used_icu / total_icu * 100`
   - Vaccination facts:
     - Manufacturer-wise distributed and series-complete counts
     - `administered`, `admin_per_100k`
     - `administered_dose1_recip`, `administered_dose1_pop_pct`
     - `series_complete_yes`, `series_complete_pop_pct`
     - Booster/additional doses and `additional_doses_vax_pct`

4. **Exports**
   - Full mart: `covid_master_data.csv`
   - ICU-focused subset: `icu_data.csv` (lat/long + ICU metrics + vaccination percentages)

---

### Time-Series Modeling (ARIMA on `df_ts`)

Using the **national daily aggregate**:

1. **Stationarity diagnostics**
   - Use `adfuller` from `statsmodels.tsa.stattools` to test stationarity of `cases_new`.
   - Use `ndiffs` from `pmdarima.arima.utils` to estimate required differencing `d`.

2. **Model identification**
   - Inspect **ACF** and **PACF** plots:
     - `plot_acf`, `plot_pacf` from `statsmodels.graphics.tsaplots`
   - Choose `(p, d, q)` (manual or via `pmdarima` utilities / auto-ARIMA style).

3. **Model fitting**
   - `ARIMA(df_ts['cases_new'], order=(p, d, q))` via `statsmodels.tsa.arima.model.ARIMA`.
   - Fit model and generate in-sample or out-of-sample forecasts.

4. **Evaluation**
   - Compare `forecast_ARIMA` vs actual `cases_new` using:
     - `mean_squared_error` + `sqrt` to compute RMSE.
   - Append `forecast_ARIMA` to `df_ts` for visualization.

---

### Visualization & Analytics

Using `matplotlib`/`seaborn` (and exporting to CSV for **Tableau**):

1. **State-level trend plots**
   - `cases_total` over time for `NY`, `MI`.
   - `death_total` over time for `NY`, `MI`.
   - Weekly aggregated `cases_new` via `df_hvc_sum` grouped by `w_id`.

2. **ICU occupancy**
   - Line plots for:
     - `total_icu` vs `used_icu` (e.g., `NY`).
     - Derived `% ICU occupancy` vs time.

3. **Vaccine manufacturer comparisons**
   - For `CA` and `MI`, multi-line plots:
     - `series_complete_janssen`, `series_complete_pfizer`, `series_complete_moderna`.
   - Booster uptake:
     - `additional_doses_janssen`, `additional_doses_moderna`, `additional_doses_pfizer`.

4. **Cross-state comparisons**
   - Barplots by state (using `df_hvc2`):
     - `series_complete_pop_pct` (fully vaccinated %).
     - `additional_doses_vax_pct` (booster coverage %).

5. **National weekly curve**
   - Group `df_hvc` by `w_id` to get `cases_new` per epidemiological week for the US and plot.

---

### Final Artifacts

- **Cleaned weekly tables:**
  - `df_cases` (cases/deaths by state-week)
  - `df_vaccine` (vaccination metrics by state-week)
  - `df_icu` (hospital/ICU utilization by state-week)
  - `df_hvc` (fully joined “COVID master” analytic mart)
- **Daily modeling tables:**
  - `df_cases_daily` (state-level daily)
  - `df_ts` (national daily with ARIMA forecast)
- **Exports for BI:**
  - `covid_master_data.csv`
  - `icu_data.csv`

Overall, the project is a **full data pipeline**:
raw open data → cleaning & quality filters → key design (`uid`) → state-week fact mart → ARIMA modeling → visual analytics (Python + Tableau).
