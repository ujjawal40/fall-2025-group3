import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Real Estate Price Prediction using Machine Learning""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            The goal of this project is to develop a Predictive model and a recommendation engine for real estate prices.
            The Predictive model will analyze historical real estate data to forecast future property prices based on various features such as location, size, number of rooms, and amenities.
            The recommendation engine will suggest optimal buying or selling times and strategies based on market trends and individual
            preferences. This project aims to provide valuable insights for buyers, sellers, and investors in the real estate market.
            """,
            
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            Companys Internal Dataset
            1. Property Listings: A comprehensive dataset containing historical and current property listings, including features such as location, size, number of rooms, amenities, and price.
            2. Market Trends: Data on real estate market trends, including average prices, demand-supply dynamics, and economic indicators.
            3. User Preferences: Data on user preferences and behaviors, including search history, saved properties, and transaction history.
            4. External Data Sources: Integration of external data sources such as census data, economic reports, and geographic information to enhance the predictive model.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            We are starting by turning raw event logs (listings, price changes, sales) into clean, time-aligned
            training examples that mirror the real decision: given a new listing today, what will it sell for if/when
            it sells soon? Concretely, we pair each listing to its next recorded sale within a bounded window and build
            strictly “as-of” features—property attributes, seasonality, and lagged ZIP-level signals—so the model
            never sees the future.

            This framing gives us three immediate benefits at project kickoff:
            1) **Validity:** eliminates common sources of label/feature leakage.
            2) **Adaptability:** enables walk-forward, month-ahead evaluation that reflects deployment and surfaces
            periods of stress early.
            3) **Scalability:** Snowflake-native column pruning + chunked ingestion lets us train on millions of
            episodes without expensive preprocessing outside the warehouse.

            The initial baseline will use gradient-boosted trees to learn nonlinear relations between listing traits and
            local market context, providing a strong, interpretable starting point. From there we can iterate toward
            calibrated uncertainty, drift monitoring, and retraining cadences. The end goal is a robust, auditable
            pricing service that reduces manual CMA effort, shortens listing cycles, and maintains accuracy across
            changing market regimes.
        """
        # -----------------------------------------------------------------------------------------------------------------------
            "Methodology":
            """
                
                
             [Data ingestion & normalization (Snowflake-first)]
            - Source table: `APIFY_SOLD_IN_7_DAYS_ENCODED`.
            - Column pruning in-warehouse to cut I/O: keep all “as-of” listing attributes and drop
            leakage/unused fields (e.g., raw `PRICE`, view counts, HOA text, materials, etc.).
            - Chunked streaming to pandas (`CHUNK_ROWS_SNOWFLAKE=250k`) to scale to millions of rows.
            - Parse/flatten JSON `PRICEHISTORY` so each historical event becomes a row; retain all original
            columns + five transaction fields:
            (`HISTORICAL_TRANSACTION_DATE`, `HISTORICAL_TRANSACTION_PRICE`, `HISTORICAL_EVENT_TYPE`,
            `HISTORICAL_PRICE_CHANGE_RATE`, `HISTORICAL_DATA_SOURCE`).
            - Hygiene filters: require ZIP + event type + valid date; keep prices in a plausible band
            (10k–20M); light type coercions & memory down-casting.

            [Split the stream into LIST vs SOLD and add timestamps]
            - Build two frames from the flattened events:
            - **LIST** = rows where `HISTORICAL_EVENT_TYPE == "Listed for sale"`.
            - **SOLD** = rows where `HISTORICAL_EVENT_TYPE == "Sold"`.
            - Derive time features on each:
            - Listing: `LIST_TRANSACTION_DATE`, `LIST_PRICE`, `LIST_TRANSACTION_YEAR/MONTH`,
                month angle encodings (`LIST_MONTH_SIN/COS`).
            - Sold:   `SOLD_TRANSACTION_DATE`, `SOLD_PRICE`, `SOLD_TRANSACTION_YEAR/MONTH`.

            [Pairing strategy → “next sale within horizon”]
            - For each listing, `merge_asof` **forward** on the same `ZPID` to the **next** sale within
            `MAX_DAYS_TO_SALE` (configurable; used 360 for the baseline split and 180 for walk-forward).
            - Compute `DAYS_TO_SALE`; drop pairs without a sale in the horizon. This mirrors the operational
            question (“what will this listing sell for soon?”) and avoids matching to stale future sales.

            [Market context features with strict “as-of” semantics]
            - Aggregate sold events by ZIP×month to form:
            `ZIP_MEDIAN_PRICE`, `ZIP_MEAN_PRICE`, `ZIP_SALES_COUNT`.
            - Shift each metric **by one month** and forward-fill within ZIP before joining, so the listing
            at month m only sees information available *through m–1*.
            - Backward `merge_asof` from listing date to the latest available ZIP stats.
            - Derive `LIST_PRICE_RATIO = LIST_PRICE / ZIP_MEDIAN_PRICE` to normalize for local price level.

            [Feature matrix construction]
            - Exclude identifiers & future fields: `ZPID`, sold timestamps/prices, raw `PRICE`,
            free-text, and leakage-prone attrs (see drop list).
            - Coerce numeric-like strings (e.g., `SQFT`, beds/baths, school distances) to numbers;
            treat remaining strings as categorical (passed natively to LightGBM).
            - Fill numerics with 0; keep 16 categorical features (observed in baseline run).

            [Modeling: gradient-boosted trees baseline]
            - Estimator: LightGBM regression with conservative defaults:
            `learning_rate=0.05`, `num_leaves=127`, `max_depth=8`,
            `min_data_in_leaf=150`, subsampling & feature fraction at 0.8.
            - Early stopping (patience=50) over ≤1000 rounds.
            - Metrics: MAE (primary), R² (reporting).

            [Evaluation protocols]
            - **Single time split**: hold out the most recent 24 months by listing date to mimic deployment
            and check headline generalization (baseline MAE ≈ $30k, R² ≈ 0.93).
            - **Walk-forward backtest (expanding window)**:
            - Require ≥24 months of training.
            - Evaluate one month ahead per fold (step=1), over the full history.
            - Overall across ≈1.6M validation rows: MAE ≈ $24.7k, R² ≈ 0.925.
            - Persist per-row predictions to `/tmp/walkforward_preds.csv` and stage to `@~` for audits.
            - Visualization: monthly MAE and R² trajectories, plus MAE-vs-R² scatter sized by fold rows,
            to surface regime stress (e.g., spikes around late-2021/2022).

            [Operationalization & guardrails]
            - Scalability: all heavy lifting stays close to Snowflake; pandas only holds the active
            working set (chunk iterator + memory reduction).
            - Reproducibility: fixed random seed; deterministic pairing; explicit feature drops.
            - Leakage checks: (1) forward pairing with tolerance; (2) ZIP stats shifted and joined
            backward; (3) time-based validation only.
            - Artifacts & audit: keep pairing horizons, feature lists, LightGBM params, and monthly
            metrics under version control; store fold-level predictions for post-hoc error slicing.

            [Repository layout (proposed from the notebook cells)]
            - `data/ingest.py`            → create_complete_property_dataset(...)
            - `data/pairing.py`           → pair_listing_to_next_sold(...)
            - `features/zip_lags.py`      → make_zip_monthly_shifted(...)
            - `features/matrix.py`        → build_feature_matrix(...)
            - `models/lgbm.py`            → train/evaluate LightGBM
            - `eval/walkforward.py`       → walk_forward_backtest(...)
            - `viz/report.py`             → plots for MAE/R² over time, extremes, scatter
            - `configs/*.yaml`            → horizons, drops, LightGBM params, split policy

            [What we will tackle next]
            - Price-only vs. joint objective with `DAYS_TO_SALE` (multi-task or two-stage).
            - Calibration: conformal intervals or quantile boosting for P50/P90 bands.
            - Drift monitoring: ZIP-level rolling error, alerting on spike patterns like 2021–2022.
            - Segment diagnostics: error by price tier, home type, geography, days-to-sale buckets.
            - Retraining cadence: monthly refresh with expanding window; backtest-gated promotion.
            - Feature extensions (still “as-of”): mortgage rate at list date, county-level DOM,
            supply/demand indices already available in the table, and broker-provided attributes.

            [`train.py` & `backtest.py` mapping]
            - `train.py`: run the single 24-month holdout pipeline end-to-end; emit model, features,
            metrics, and SHAP/importance plots.
            - `backtest.py`: run the expanding walk-forward; write per-fold metrics + predictions to stage;
            generate all figures used in reviews.

        """

        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            [Understanding  1 week
            [`model.py` & `train.py`] 2 weeks
            [`utils/env.py` & `test.py`] 7 weeks
            [`benchmark.py`] 2 weeks (start writting research paper here)
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            Goal is to work solo. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            1) Leakage-safe supervision from event logs  
            - We formalize a general pairing strategy that converts raw listing/price-change/sale events into supervised 
                (listing → next-sale) examples with a configurable horizon, avoiding peeking into post-listing information.  
            - We contribute a principled “as-of” feature rule set (backward joins + one-month-shifted aggregates) that can be 
                reused in other temporal-leakage-prone domains.

            2) Scalable Snowflake-native data engineering pattern  
            - A push-down ETL design that parses/flat-tens JSON histories in-warehouse, prunes columns before egress, and 
                streams in 250k-row chunks—demonstrated on ~9M+ transactional rows.  
            - Memory-aware pandas utilities (type coercion, downcasting) that make large-scale experimentation accessible on 
                commodity compute.

            3) Market-context features with strict “as-of” semantics  
            - A reusable recipe for locality aggregates (ZIP×month median/mean price and sales count), shifted and forward-filled 
                to guarantee only pre-listing information is used.  
            - A simple but powerful normalization (`LIST_PRICE_RATIO = LIST_PRICE / ZIP_MEDIAN_PRICE`) that improves cross-region 
                comparability.

            4) Transparent time-series evaluation protocols  
            - A deployment-faithful **24-month holdout** and an **expanding walk-forward backtest** (monthly steps) with per-fold 
                MAE/R², per-row predictions, and audit-ready artifacts.  
            - Visualization suite (MAE/R² timelines, MAE↔R² scatter sized by fold rows) to reveal market-regime stress 
                (e.g., 2021–2022 volatility) and drift.

            5) Strong, simple baseline with clear extension hooks  
            - A LightGBM baseline (no deep feature stacks) delivering competitive accuracy (≈$25k MAE, R²≈0.93 in walk-forward) 
                that sets a clean yardstick for future work.  
            - Drop-in hooks for quantile/conformal prediction, multi-task learning with `DAYS_TO_SALE`, and richer macro features 
                (rates, supply/demand).

            6) Generalizable methodology beyond real estate  
            - The event-pairing + “as-of” feature pattern applies to other verticals (autos, rentals, e-commerce resale) where 
                listing and sale events are observed and leakage risks are high.

            7) Open research artifacts (code & specs)  
            - We plan to release the framework as open-source code: Snowflake SQL templates, pairing/feature builders, 
                walk-forward harness, and visualization scripts—plus a synthetic (non-proprietary) benchmark mirroring class 
                balances and temporal structure for reproducibility.

            8) Empirical and methodological insights  
            - Evidence that careful temporal hygiene + simple locality context captures a large share of signal, offering a 
                strong baseline before complex models.  
            - Practical guidance on horizon selection, regime-aware evaluation, and operational guardrails for production pricing.

            Collectively, these contributions provide a scalable, audit-friendly blueprint for building and evaluating 
            listing-time price models. Results and tooling are suitable for technical reports, industry data-science venues, 
            and as a starting point for academic investigations into leakage-aware learning and uncertainty quantification 
            in transactional forecasting.
        """

        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
                    - **Temporal leakage hazards**
          - Market aggregates (ZIP median/mean price, sales count) must be computed strictly **as-of** the listing date; any same-month look-ahead leaks.
          - Non-event covariates (e.g., “HOTNESS_SCORE”, mortgage rates) may be published with latency—using month-t values at time-t listing can still leak future knowledge unless shifted.
          - Multiple listings/price changes per property can reveal future outcomes if not carefully filtered.

        - **Pairing & censoring biases (listing → next sold)**
          - The merge-as-of pairing assumes the **next** sold event within `MAX_DAYS_TO_SALE` belongs to the listing; relists, flips, or title transfers can mis-pair.
          - Right-censoring: listings that never sell within the window are dropped, biasing toward faster-selling homes and understating uncertainty.
          - Window choice inconsistency (e.g., 360 days in the single holdout vs. 180 in the walk-forward) can make results hard to compare.

        - **Aggregation sparsity & staleness**
          - Many ZIP×month cells are empty; the 1-month shift + forward-fill avoids leakage but can propagate stale values in low-liquidity areas.
          - Missingness handling in EDA (monthly `asfreq`) produced **0 valid ZIP series** with ≥36 periods due to long global ranges; this can hide useful shorter local histories.

        - **Feature hygiene & proxies**
          - Some demographic features (race/ethnicity shares, income) or their proxies raise **fair housing** and compliance concerns; even if dropped at serve-time, they can contaminate training.
          - High-cardinality categoricals (e.g., CITY, ZIPCODE) can memorize idiosyncrasies; need regularization, target encoding with **strict time folds**, or careful use as categories.

        - **Metric sensitivity & regime shifts**
          - MAE is dominated by expensive markets; per-price-band MAE, quantile loss, and calibration checks are needed.
          - 2011–2014 and 2021–2022 show stress: spikes in MAE with still-high R² indicate **distribution shift**; a single global model may underperform in regime changes.

        - **Modeling pitfalls**
          - LightGBM with many weakly-informative features can overfit without robust time-based validation and monotonic/interaction constraints.
          - Using only month sin/cos may under-capture seasonality vs. weekly dynamics; weekly city signals didn’t translate to monthly ZIP analysis during EDA.
          - Label noise (recording errors, concessions, non-arm’s-length sales) caps achievable accuracy.

        - **Evaluation & comparability**
          - The single 24-month holdout and the monthly walk-forward use different pairing horizons; report both or harmonize to avoid mixed conclusions.
          - Early folds have tiny sample sizes—variance is high; aggregate headlines should weight by rows.

        - **Compute, cost, and ops**
          - 9M+ events → 1.6–1.8M training pairs: memory pressure, long wall-clock, and egress costs from Snowflake.
          - Per-fold LightGBM training + prediction can be expensive; need job orchestration, deterministic seeds, and artifact caching.

        - **Reproducibility**
          - Chunked ingestion + random sampling must pin seeds and ordering; schema changes in the warehouse can silently break feature contracts.
          - Version and stage your “as-of” feature SQL, pairing rules, and model configs to ensure auditability.

        - **Scope creep**
          - Mixing price-level prediction with time-to-sale (DAYS_TO_SALE) as a secondary objective without clear multi-task design can blur targets and degrade both.
        """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Ujjawal Dwivedi",
        "Proposed by email": "ujjawal.dwivedi@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "None",
        "funding_opportunity": "Internship Extension",
        "github_repo": "fall-2025-group3",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")