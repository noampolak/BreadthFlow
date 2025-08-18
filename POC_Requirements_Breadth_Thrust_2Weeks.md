# POC Requirements – Breadth/Thrust Signals (2 Weeks)

**Audience:** Noam (engineering), Yossi (quant/research).**Goal:** Deliver a runnable, near–real‑time **Breadth/Thrust signal** POC with backtests and a CLI demo.**Scope:** Minimal but complete pipeline: replayed price data → streaming features → aggregated breadth → weighted score → backtest → CLI outputs → demo artifacts.**Out-of-scope:** Real‑time market data subscriptions, options chain ingestion, REST service, production hardening (unless time permits).

---

## Global Conventions

- **Language/Runtime:** Python 3.11, PySpark (Spark 3.5.x), Kafka 3.x, MinIO (S3 API), Delta Lake 3.x.
- **Repo layout (suggested):**
  ```
  /ingestion        # replay tools, csv loaders
  /streaming        # spark structured streaming jobs (features)
  /features         # shared feature code (ema, ma, zscores, windows)
  /model            # scoring, thresholds, calibration
  /backtests        # engine + strategies
  /cli              # bf CLI entrypoints
  /data             # sample csvs (OHLCV); small, for POC only
  /docs             # specs, notebooks for demo plots
  /infra            # docker-compose, env templates
  ```
- **Env & Secrets:** `.env.example` (no secrets in git). Required keys: `KAFKA_BROKERS`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET=breadthflow`, `SPARK_MASTER=local[*]`.
- **Time/Clock:** All timestamps UTC internally; display local TZ as needed.
- **Message Keys:** Kafka key = `symbol` (string); value JSON (schemas below).
- **Partitioning:** Delta tables partitioned by `date` (YYYY‑MM‑DD) and optionally `symbol`.
- **Quality Gates:** No missing timestamps beyond tolerance; monotonic time in stream; schema validation at source boundaries.

---

## Data Schemas

### 1) CSV OHLCV (input for replay)

Columns (required):

- `ts` (ISO 8601, UTC)
- `symbol` (str)
- `open`, `high`, `low`, `close` (float)
- `volume` (float/int)

### 2) Kafka topic: `quotes_replay`

Value JSON:

```json
{
  "ts": "2024-01-02T14:35:00Z",
  "symbol": "AAPL",
  "o": 181.23,
  "h": 181.80,
  "l": 180.95,
  "c": 181.50,
  "v": 345678
}
```

Key: `symbol`Headers: `source=csv_replay`, `granularity=1m|5m|1d`.

### 3) Delta table: `features/ad_basics`

Partition: `date`. Columns:

- `ts`, `symbol`, `ret_1` (close/prev_close − 1), `is_adv` (bool),
- `vol_up` (volume if `is_adv` else 0), `vol_dn` (volume if not),
- Aggregates per window (market‑level): `adv_count`, `dec_count`, `adv_ratio`, `up_vol`, `dn_vol`, `up_dn_vol_ratio`, `window` (e.g., `1m`, `5m`, `1d`).

### 4) Delta table: `features/ma_flags`

- Per `symbol`: `ma20`, `ma50`, `above_ma20` (bool), `above_ma50` (bool).
- Market aggregates: `%_above_ma20`, `%_above_ma50`.

### 5) Delta table: `features/mcclellan`

- Inputs: daily `AD = adv_count − dec_count`.
- Outputs: `ema19`, `ema39`, `mcclellan_osc = ema19 − ema39`, `summation` (cumulative osc).
- Flags: `mcc_extreme_pos`, `mcc_extreme_neg` by documented thresholds.

### 6) Delta table: `signals/breadth_score`

- `ts` (market close or window end), `score_raw` (z‑score sum), `score_0_100` (scaled), `flags` (bitmask: `thrust`, `overbought`, `oversold`), `inputs_version`, `calibration_version`.

---

## Algorithms (formal)

- **Return:** `ret_1 = close_t / close_{t-1} − 1`.
- **A/D Issues:** `is_adv = (ret_1 > 0)`; counts per universe window.
- **A/D Volume:** `up_vol = Σ volume[is_adv]`; `dn_vol = Σ volume[!is_adv]`; `up_dn_vol_ratio = up_vol / max(dn_vol, ε)`.
- **Moving Averages:** `ma_k` via rolling mean on `close` (EMA allowed if data sparse). Flags: `above_maK = close > maK`.
- **McClellan:** Daily `AD_t = adv_t − dec_t`; `EMA_k(AD) = EMA_{k}(AD)`. Oscillator = `EMA_19 − EMA_39`; Summation = cumulative oscillator.
- **Zweig Breadth Thrust (ZBT):** Let `B_t = adv_t / (adv_t + dec_t)` daily. A ZBT occurs if a 10‑day **simple moving average** of `B` rises from `< 0.40` to `> 0.615` **within 10 trading days**. Produce `zbt_flag` on the day the condition is satisfied.
- **Score (baseline):** For selected indicators `I = {adv_ratio, up_dn_vol_ratio, %above_ma20, %above_ma50, mcclellan_osc zscore, zbt_flag}`, compute z‑scores (winsorized to [−3, 3]); `score_raw = Σ w_i z_i`, with `w_i = 1/|I|`. Map to 0–100 via min‑max on a training window. Thresholds: `buy if score ≥ 60`, `reduce/short if ≤ 40` (tunable).

---

# Task‑Level Requirements (Detailed)

Each task below includes: Objective, Type/Owner, Dependencies, Inputs/Outputs, Non‑functional, Steps, Acceptance Criteria, Test Cases, and Done Checklist.

## T‑01 — Define POC Scope & Demo Success

- **Type/Owner:** Research — Yossi
- **Objective:** Freeze POC boundaries and success metrics for the 2‑week window.
- **Dependencies:** None.
- **Inputs:** This spec, time constraints.
- **Outputs:** `/docs/poc_scope.md` containing: in‑scope list, out‑of‑scope list, KPIs (Sharpe, hit‑rate, turnover, max DD), demo storyline.
- **Non‑functional:** Clarity over completeness; ≤ 2 pages.
- **Steps:** Draft → review with Noam → finalize.
- **Acceptance:** Doc exists, committed, referenced in README; both agree on KPIs and demo steps.
- **Tests:** N/A (review).
- **DoD:** Merged to main, linked from README.

## T‑02 — Select Universe (~100 liquid tickers)

- **Type/Owner:** Research — Yossi
- **Objective:** Compile a static list of ~100 liquid US symbols (or TA‑35 alternative) for POC.
- **Inputs:** Public lists (e.g., S&P 100); liquidity heuristics.
- **Outputs:** `/data/universe_poc.csv` with columns: `symbol, name, sector`.
- **Steps:** Pick list → add sectors → save CSV.
- **Acceptance:** ≥ 90 symbols valid; no duplicates; sectors populated.

## T‑03 — Docker Compose: Spark + Kafka + MinIO

- **Type/Owner:** Dev — Noam
- **Dependencies:** None
- **Inputs:** `infra/docker-compose.yml`, `.env.example`.
- **Outputs:** Running stack via `make up`; health endpoints OK.
- **Steps:** Prepare compose services; mount volumes; seed bucket `breadthflow`.
- **Acceptance:** `docker compose up` brings cluster; `spark UI` reachable; topic creation possible.
- **Tests:** Smoke: publish & consume from `quotes_replay`.

## T‑04 — Obtain OHLCV History (sample)

- **Type/Owner:** Research — Noam
- **Objective:** Download/import daily or 5‑min OHLCV for the universe (limited range, e.g., last 2 years daily or last 3 months 5m).
- **Outputs:** `/data/ohlcv/*.csv` (per symbol or concatenated); schema as defined.
- **Acceptance:** Missing ≤ 2% rows; timestamps UTC; columns validated.

## T‑05 — Data Quality Criteria (DQ spec)

- **Type/Owner:** Research — Yossi
- **Outputs:** `/docs/dq_rules.md` with rules: missing tolerance, gap detection, outlier z>5 handling, min volume filters, trading day calendar.
- **Acceptance:** DQ rules referenced by feature jobs.

## T‑06 — Replay CLI: CSV → Kafka (`quotes_replay`)

- **Type/Owner:** Dev — Noam
- **Dependencies:** T‑03, T‑04
- **CLI:** `python -m ingestion.replay --csv /data/ohlcv --topic quotes_replay --speed 60x --granularity 1m|5m|1d --loop`
- **Inputs:** CSV OHLCV
- **Outputs:** Kafka messages (schema above)
- **Functional:** Rate‑controlled publish; backpressure tolerant; graceful stop.
- **Acceptance:** Can stream a full month file in ≤ real‑time/60x; schema validated; logs per 10k msgs.
- **Tests:** Unit (schema marshal); integration (produce→consume count match ±1%).

## T‑07 — Delta Lake I/O (basic)

- **Type/Owner:** Dev — Noam
- **Objective:** Helpers to read/write Delta/Parquet; partition by date; idempotent writes.
- **Outputs:** `/features/common/io.py` with `write_delta(df, path, partition=['date'])`, `read_delta(path)`.
- **Acceptance:** Write+read round‑trip; schema preserved; partition pruning effective.

## T‑08 — Streaming Features: A/D Issues & A/D Volume

- **Type/Owner:** Dev — Noam
- **Dependencies:** T‑06, T‑07
- **Job:** `spark-submit streaming/ad_job.py --source quotes_replay --window 1m --sink s3a://breadthflow/features/ad_basics`
- **Logic:** Compute per‑symbol `ret_1`, `is_adv`; aggregate `adv_count`, `dec_count`, `adv_ratio`, `up_vol`, `dn_vol`, `up_dn_vol_ratio` per window.
- **Acceptance:** End‑to‑end stream produces rows in Delta with watermarking; late data tolerance = 2 windows; exactly‑once semantics via checkpoints.
- **Tests:** Integration with replay; correctness on a known micro‑dataset.

## T‑09 — Streaming Features: % above MA(20/50)

- **Type/Owner:** Dev — Noam
- **Logic:** Maintain rolling MA(20/50) per symbol; flag `above_ma20`, `above_ma50`; aggregate market percentages.
- **Acceptance:** Rolling state correct across micro‑batches; aggregates update each window.

## T‑10 — Indicator Math Specs (formulas)

- **Type/Owner:** Research — Yossi
- **Outputs:** `/docs/indicator_math.md` defining: `B_t`, ZBT procedure, McClellan EMA params, thresholds for extremes; references.
- **Acceptance:** Used by T‑12, T‑14.

## T‑11 — Entry/Exit Rule Spec (backtest policy)

- **Type/Owner:** Research — Yossi
- **Outputs:** `/docs/policy.md` with: buy/sell thresholds, holding windows (k=5/10 days), position sizing (fixed notional), liquidity filters, transaction costs (e.g., 2 bps/side + 1 tick slippage).
- **Acceptance:** Backtester can consume it as YAML or constants.

## T‑12 — Baseline Weighted Score (z‑scores)

- **Type/Owner:** Dev — Noam
- **Dependencies:** T‑08, T‑09, T‑10
- **Outputs:** `/model/score.py`; table `signals/breadth_score`.
- **Logic:** Winsorize z in [−3,3]; equal weights; scale 0–100; flags (`thrust` if ZBT true, `overbought/oversold` via percentiles).
- **Acceptance:** Deterministic on fixed data; unit tests for z and scale mapping.

## T‑13 — Backtester (k=5/10 days)

- **Type/Owner:** Dev — Noam
- **Dependencies:** T‑11, T‑12
- **API:**
  ```python
  run_backtest(prices_df, score_df, policy) -> metrics, trades_df
  ```
- **Features:** Transaction costs; portfolio equity; metrics (Sharpe, hit‑rate, max DD, turnover).  
- **Acceptance:** Reproducible results on seed; unit tests for P&L math; CSV exports in `/backtests/out/`.

## T‑14 — McClellan Oscillator

- **Type/Owner:** Dev — Noam
- **Dependencies:** T‑08
- **Outputs:** Add `mcclellan_osc`, `summation` to Delta; z‑score variant for model.
- **Acceptance:** Matches hand‑computed sample; thresholds configurable.

## T‑15 — ZBT Flag

- **Type/Owner:** Dev — Noam; **Spec:** T‑10
- **Outputs:** Boolean `zbt_flag` at daily granularity, stored per date.
- **Acceptance:** Backtest identifies known historical ZBT dates (unit tests include 2–3 fixtures).

## T‑16 — Thresholds & Calibration (no ML)

- **Type/Owner:** Testing — Yossi
- **Dependencies:** T‑12, T‑13
- **Outputs:** `/model/calibration.yaml` with `buy_score`, `sell_score`, `cooldown_days`, `max_positions`.
- **Acceptance:** OOS slice shows uplift vs. naive baseline; changes tracked in commit.

## T‑17 — OOS Report (KPIs)

- **Type/Owner:** Testing — Yossi
- **Inputs:** Backtest outputs.
- **Outputs:** `/docs/oos_report.md` + `/docs/plots/*.png` (equity curve, rolling Sharpe, drawdown, exposure).
- **Acceptance:** KPIs computed; narrative summary ≤ 1 page.

## T‑18 — CLI: `score` / `signals`

- **Type/Owner:** Dev — Noam
- **Commands:**
  - `bf score --date 2024-06-01` → prints/exports score & flags.
  - `bf signals --from 2024-05-01 --to 2024-06-30` → lists entries/exits per policy.
- **Outputs:** stdout table + CSV in `/out/`.
- **Acceptance:** Works on sample Delta; validated vs. backtest events.

## T‑19 — E2E Demo via Replay

- **Type/Owner:** Testing — Yossi
- **Dependencies:** T‑06..T‑13
- **Procedure:** Start stack → run replay (1 month) → streaming features produce tables → compute score → run backtest → run CLI → capture screenshots.
- **Acceptance:** Step‑by‑step doc `/docs/demo_run.md` with exact commands and expected outputs.

## T‑20 — README POC

- **Type/Owner:** Docs — Yossi (author), Noam (review)
- **Contents:** 10‑minute quickstart; architecture diagram; what is breadth/thrust; how to run demo; limitations; results snapshot.
- **Acceptance:** A new user can run demo end‑to‑end.

---

# Stretch Tasks (Optional)

## S‑01 — Option Playbook (without chains)

- **Owner:** Yossi
- **Output:** `/docs/options_playbook.md` mapping score/vol regimes → strategies (Condor, spreads) with DTE/Δ heuristics.

## S‑02 — 5‑Minute Scoring

- **Owner:** Noam
- **Output:** Extend replay & feature jobs to 5‑minute windows; ensure state continuity and checkpointing.

---

## Non‑Functional Requirements

- **Reproducibility:** Fixed seeds; `requirements.txt` / `poetry.lock` checked in.
- **Performance:** On laptop, replay + streaming over 1 month (100 symbols, 5‑min bars) finishes ≤ 10 minutes at 60× speed.
- **Observability (POC):** Structured logs; basic counters (messages processed, late rows, rows written) printed every N batches.

## Testing Strategy

- **Unit:** z‑score, MA, EMA, McClellan, score mapping, P&L math.
- **Integration:** replay→kafka→spark→delta path; deterministic micro‑dataset with golden outputs.
- **OOS Split:** Chronological split (e.g., train/calibrate: year‑1; OOS: last 3 months).

## Definition of Done (per task)

- Code merged to `main` with passing tests.
- Docs updated and cross‑linked.
- Sample command(s) provided.
- Output tables/files verified exist and conform to schema.

---

## Appendix A — Pseudocode Snippets

### A/D Aggregation (Structured Streaming)

```python
# read
df = spark.readStream.format("kafka").option("subscribe", "quotes_replay").load()
msg = from_json(col("value").cast("string"), schema).alias("m")
ts = to_timestamp(col("m.ts"))

# per-symbol returns
win = Window.partitionBy("symbol").orderBy(ts)
ret = col("m.c")/lag(col("m.c")).over(win) - 1
is_adv = (ret > 0).cast("int")

# window aggregation
w = window(ts, "1 minute", "1 minute")
aggr = df.groupBy(w).agg(
  sum(is_adv).alias("adv_count"),
  (count("*") - sum(is_adv)).alias("dec_count"),
  sum(when(is_adv==1, col("m.v")).otherwise(0)).alias("up_vol"),
  sum(when(is_adv==0, col("m.v")).otherwise(0)).alias("dn_vol")
)
aggr = aggr.withColumn("adv_ratio", col("adv_count")/(col("adv_count")+col("dec_count")))            .withColumn("up_dn_vol_ratio", col("up_vol")/greatest(col("dn_vol"), lit(1)))

# write delta with checkpoint
aggr.writeStream   .format("delta").option("checkpointLocation", ckpt)   .start(out_path)
```

### Baseline Score

```python
features = ["adv_ratio", "%_above_ma20", "%_above_ma50", "up_dn_vol_ratio", "mcc_z", "zbt_flag"]
z = winsorize_z(df[features])
score_raw = sum(z[f] for f in features)/len(features)
score_0_100 = scale_0_100(score_raw, hist_window)
```

---

## Appendix B — CLI I/O Formats

- `bf score --date YYYY-MM-DD` → prints:

```
DATE       SCORE  FLAGS       %ABOVE20  %ABOVE50  ADV_RATIO  U/D_VOL
2024-06-01 67.2   THRUST      71.3%     55.0%     0.61       1.45
```

- `bf signals --from A --to B` → CSV:

```
date,symbol,action,reason,score,holding_days
2024-06-03,SPY,BUY,score>=60,62.1,5
```

---

**Version:** 1.0 — 2025‑08‑10 (Asia/Jerusalem)
