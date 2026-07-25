# SN126 Poker44 — Authoritative Validator Specification

Traced from the **current local repository** (not memory). Every claim is tagged
with a file/line reference or explicitly marked as documentation-only, inferred,
or backend-controlled. Line numbers reflect the repo state at authoring time.

---

## 1. Where production data originates

- Live benchmark tables on Poker44 platform infrastructure generate real hands;
  hands are persisted in platform SQL; the backend builds labeled evaluation
  batches. **[DOC]** `docs/validator.md` (Canonical Chunk Lifecycle), `docs/miner.md`.
- The validator **consumes** those batches from the central eval API; it does not
  generate them. `POKER44_RUNTIME_MODE` must be `provider_runtime`
  (`neurons/validator.py:74-81`).
- Fetch path: `ProviderRuntimeDatasetProvider.fetch_hand_batch()` →
  `GET /internal/eval/current` (`poker44/validator/runtime_provider.py:407`);
  optional admin publish `POST /internal/eval/publish-current`
  (`runtime_provider.py:386`). Base URL `https://api.poker44.net`
  (`runtime_provider.py:26`, `ProviderRuntimeConfig.from_env`).

## 2. Platform-controlled vs validator-controlled

| Concern | Controller | Evidence |
|---|---|---|
| Hand generation, chunk composition, labels | **Platform/backend** | `runtime_provider.py:407-474`, docs |
| Publication cadence (24h windows, 120h epochs) | **Platform/backend** | `runtime_provider._current_competition_epoch:43-110` (mirror only), docs |
| Competition winner settlement | **Platform/backend** | `GET /internal/competition/current/weights` (`runtime_provider.py:309-327`) |
| Polling, miner query, response validation | Validator | `poker44/validator/forward.py` |
| Local reward computation, EMA | Validator | `poker44/score/scoring.py`, `base/validator.py:553-598` |
| Weight submission to chain | Validator (values may come from backend) | `base/validator.py:294-349` |
| Miner rotation, integrity registries | Validator | `forward.py:510-581`, `poker44/validator/integrity.py` |

## 3. Batches and hidden labels

- Fetched entries → `LabeledHandBatch(hands=..., is_human=...)`
  (`poker44/core/models.py:323-329`). `is_human = not bool(entry.get("is_bot"))`
  (`runtime_provider.py:453`). **[CODE]**
- Per-chunk numeric label: `batch_label = 0 if batch.is_human else 1` →
  **1 = bot, 0 = human** (`forward.py:127`). Positive class = bot end to end
  (`scoring.py:99` uses `labels == 1`). **[CODE]**
- Labels are **never sent to miners**: they live only on the validator
  (`LabeledHandBatch.is_human`), and identity/label fields are stripped from the
  miner payload (`payload_view.py` `_LEAKAGE_KEYS:8-14`). **[CODE]**
- Label semantics correctness verified across the chain: no inversion
  (`tests/` + `runtime_provider.py:453` → `forward.py:127` → `scoring.py:99`).
  The `LabeledHandBatch.is_human` docstring comment is misleading but
  non-functional. **[CODE]**

## 4. Exactly when `prepare_hand_for_miner` is applied

- Applied **on the validator, per hand, before** the `DetectionSynapse` is built:
  `chunk_dicts.append(prepare_hand_for_miner(hand_payload))` (`forward.py:121`),
  then `synapse = DetectionSynapse(chunks=chunks)` (`forward.py:146`). **[CODE]**
- Therefore the miner receives **already-sanitized** chunks and must **not**
  re-apply it. `prepare_hand_for_miner` = `build_miner_payload_hand`
  (`payload_view.py:455-457`); it windows actions to 5–8 (except 1-action→12-copy
  padding), buckets+jitters amounts, aliases seats, zeroes `outcome`, empties
  `board_cards`, nulls `hole_cards`, drops blinds, and strips leakage keys. **[CODE]**

## 5. Miner eligibility and 16-miner rotation

`forward.py:_get_candidate_miners:510-581`:
- Excludes `uid == UID_ZERO` (`:548`); optional `POKER44_TARGET_MINER_UIDS` filter
  (`:513, 550`); excludes validator-permitted hotkeys with
  `stake >= POKER44_MIN_VALIDATOR_STAKE` (default 17000) (`:515, 557`); requires a
  valid `ip`/`port>0` (`:559-562`). **[CODE]**
- Rotation: `POKER44_MINERS_PER_CYCLE` default **16** (`:514, 516`). If eligible
  count > 16, deterministic rotation
  `offset = ((forward_count-1)*16) % N` and take the 16-slice (`:566-578`). So a
  given miner is queried roughly every `ceil(N/16)` cycles. **[CODE]**

## 6. Poll interval, timeout, retries

- **Poll interval**: `POKER44_POLL_INTERVAL_SECONDS` or `config.poll_interval_seconds`
  default **300s** (`neurons/validator.py:70-72, 104-106`; `config.py:90-95`). The
  forward cycle sleeps `poll_interval` at the end of each cycle
  (`forward.py:343`, and on empty-data paths `:74, 95, 281`). **[CODE]**
- **Timeout per query**: `180.0s`, overridable via `config.neuron.timeout` or
  `POKER44_MINER_QUERY_TIMEOUT_SECONDS`, floored to `max(30.0, timeout)`
  (`forward.py:149-159`). **[CODE]**
- **Retries**: `_dendrite_with_retries(..., attempts=3)` with 0.5s backoff; on
  exhaustion returns `[None]*len(axons)` (`forward.py:164-170, 724-748`). **[CODE]**

## 7. Exact response validation rules

`forward.py:176-251` per (uid, response):
- `resp is None` → `coverage_rate=0.0` (`:177-183`). **[CODE]**
- `risk_scores is None` → `coverage_rate=0.0` (`:192-199`). **[CODE]**
- Coerce `scores_f = [float(s) for s in scores]` (`:202`). **[CODE]**
- **Length gate**: `len(scores_f) != len(chunks)` → warn + **discard** the entire
  response, `coverage_rate=0.0`, append `0.0` to coverage buffer (`:204-214`).
  → a wrong-length response scores nothing that cycle. **[CODE]**
- On valid length: `coverage_rate = len/expected` (`:217-221`), record latency
  from `resp.dendrite.process_time` (`:636-645`), extend `prediction_buffer[uid]`
  and `label_buffer[uid]` with `scores_f` / `effective_labels` (`:239-240`). **[CODE]**
- Scores are **not** range-clamped by the validator; AP/recall use them as-is;
  the 0.5 threshold logic treats `score >= 0.5` as a positive prediction. **[CODE]**

## 8. Exact local reward formula

`poker44/score/scoring.py`:
```
AP_WEIGHT=0.35  BOT_RECALL_WEIGHT=0.30  HUMAN_SAFETY_WEIGHT=0.20
CALIBRATION_WEIGHT=0.10  LATENCY_WEIGHT=0.05           (scoring.py:8-12)

ap_score          = average_precision_score(labels, scores)        (:100-104)
bot_recall        = recall at human-FPR <= 0.05 (_recall_at_fpr)   (:15-42, 105)
TSQ               = threshold_sanity_quality @ 0.5 (_threshold_metrics)  (:45-91, 106)
human_safety_pen  = TSQ                                            (:107)
calibration_qual  = TSQ                                            (:108, same value)
latency_quality   = 1.0 (constant)                                 (:109)

if human_safety_penalty <= 0:  reward = 0.0                        (:111-113, HARD CLIFF)
else: reward = clip(0.35*ap + 0.30*bot_recall
                    + 0.20*TSQ + 0.10*TSQ + 0.05*1.0, 0, 1)        (:115-122)
             = 0.35*AP + 0.30*recall@5%FPR + 0.30*TSQ + 0.05
```
- TSQ at threshold 0.5 (`_threshold_metrics:45-91`): `1.0` if only one class
  present; `0.0` if both classes present but zero true-positives at ≥0.5;
  `1.0` if `hard_fpr <= 0.10`; else `max(0, 1-(hard_fpr-0.10)/0.90)`. **[CODE]**
- **AP and recall@5%FPR are rank-invariant** (both from `argsort(scores)`), so any
  strictly monotonic score transform (e.g. our calibration) changes **only** TSQ.
  **[CODE, INFERRED consequence]**
- **Windowing**: reward computed over a per-uid window
  (`_compute_windowed_rewards:584-633`), `window = current_eval_sample_count`
  (= this cycle's chunk count). Requires `len(pred_buf) >= window` else reward 0
  (`:602-620`). **[CODE]**

## 9. Local reward vs backend competition settlement

- Local per-cycle winner: `_select_weight_targets` (`forward.py:700-722`);
  winner-take-all; `BURN_EMISSIONS=True`, `BURN_FRACTION=0.00`
  (`poker44/validator/constants.py`). Feeds `update_scores` EMA
  (alpha `moving_average_alpha` default 0.05, `base/validator.py:594-597`). **[CODE]**
- **On-chain weights** are resolved in `set_weights`
  (`base/validator.py:294-349`): `_extract_competition_weight_vector(provider,...)`
  (`:41-110`) pulls the backend settlement vector via
  `provider.get_competition_settlement_weights()` →
  `GET /internal/competition/current/weights` (`runtime_provider.py:309-327`).
  If the backend returns a usable vector it is used **directly**
  (`weights_source ∈ {competition_settlement, competition_fallback, competition_runtime}`);
  **only if `raw_weights is None`** does the validator fall back to local
  `self.scores` (`base/validator.py:338-349`). `BACKEND_BURN_FRACTION=0.00`
  (`:37`). Verified by `tests/test_competition_weight_resolution.py`. **[CODE]**
- **Net:** the local `reward()` is primarily a *reporting/telemetry and fallback*
  signal; the on-chain winner is settled server-side. The exact backend
  aggregation over a competition epoch is **not in this repo**. **[CODE + UNKNOWN]**

## 10. Known competition periods

| Period | Value | Evidence |
|---|---|---|
| Query cycle | ~`poll_interval` = **300s (5 min)** default | `validator.py:70`, `forward.py:343` **[CODE]** |
| Per-query timeout | 180s (min 30s) | `forward.py:149-159` **[CODE]** |
| Miner rotation | 16 miners/cycle → each ~every `ceil(N/16)` cycles | `forward.py:514-578` **[CODE]** |
| 24h evaluation window | daily 24h blocks | `runtime_provider._current_competition_epoch:87-103`, `docs/validator.md` **[CODE mirror + DOC]** |
| 120h competition epoch | 120h (5 days), anchored | `runtime_provider.py:60-73`, `docs/validator.md` **[CODE mirror + DOC]** |
| Weight-submission cadence | every `epoch_length` blocks (default **50** ≈ ~10 min at ~12s/block) when `should_set_weights` true | `config.py:37-42`, `base/neuron.py:199-211`, `base/validator.py:216-229` **[CODE]** |

> The epoch-timing function `_current_competition_epoch` is a **local mirror** the
> validator uses for observability; the **authoritative** window/epoch boundaries
> and settlement are backend-driven (`/internal/competition/*`). **[CODE + UNKNOWN]**

## 11. Manifest, compliance, and disqualification paths

- Miner returns `synapse.model_manifest`; validator records it
  (`forward.py:_record_model_manifest:361-413`). **[CODE]**
- Compliance: `evaluate_manifest_compliance` (`model_manifest.py:198-241`) →
  status `transparent`/`opaque`. Required fields `MIN_REQUIRED_MANIFEST_FIELDS:12-20`;
  policy violations `repo_commit_invalid`, `repo_url_must_point_to_model_repo`
  (`:229-232`). **[CODE]**
- Suspicion flags: `evaluate_manifest_suspicion` (`integrity.py:164-179`);
  registries persisted (`integrity.py`, `forward.py:453-507`). **[CODE]**
- Served-chunk fingerprinting (anti-replay): `record_served_chunks`
  (`integrity.py:107-161`, `forward.py:430-450`). **[CODE]**
- Audit lane: encrypted per-cycle evidence, optional external verifier
  (`neurons/validator.py:343-383`, `poker44/validator/audit.py`, `docs/validator.md`). **[CODE + DOC]**
- **Disqualification**: compliance/suspicion are **not** terms in `reward()`. Per
  `docs/miner.md` (Model Manifest), a **high-scoring** miner may be reviewed and,
  if the published repo/commit does not match served logic, be *penalized,
  disqualified, or reduced to 0* — a **manual/backend** action, not an automated
  scoring term. **[DOC + UNKNOWN]**

---

## Evidence classification

### CONFIRMED BY CODE
- Data fetch path & provider_runtime requirement (`runtime_provider.py`, `validator.py:74-81`).
- Label semantics 1=bot/0=human; labels validator-only (`runtime_provider.py:453`, `forward.py:127`, `payload_view.py:8-14`).
- `prepare_hand_for_miner` applied validator-side before synapse (`forward.py:121,146`).
- Eligibility + 16-miner rotation (`forward.py:510-581`).
- Poll interval 300s, timeout 180s/min30, 3 retries (`validator.py:70`, `forward.py:149-170`).
- Response validation incl. length-gate discard (`forward.py:176-251`).
- Exact reward formula + hard 0-cliff + windowing (`scoring.py`, `forward.py:584-633`).
- On-chain weights = backend settlement, local scores are fallback (`base/validator.py:294-349`, `runtime_provider.py:309-327`).
- EMA alpha, winner-take-all, `BURN_FRACTION=0.00` (`base/validator.py:594`, `forward.py:700-722`, `constants.py`).
- Manifest/compliance/suspicion/served-chunk/audit mechanics (`model_manifest.py`, `integrity.py`, `audit.py`).

### CONFIRMED BY DOCUMENTATION (not fully in repo code)
- Live tables → SQL → backend batch build (`docs/validator.md`, `docs/miner.md`).
- 24h evaluation windows nested in 120h epochs as the operational cadence (`docs/validator.md`).
- High-score manifest review / disqualification policy (`docs/miner.md`).
- Public benchmark exposes labels for training (`docs/training-benchmark.md`).

### INFERRED (logical consequence of code, not stated verbatim)
- AP & recall@5%FPR are rank-invariant → calibration moves only TSQ.
- Effective per-miner query rate ≈ `ceil(N/16) × poll_interval`, multiplied by number of active validators.
- Local `reward()` functions mainly as a reporting/fallback signal given backend settlement precedence.

### UNKNOWN / BACKEND-CONTROLLED (must not be guessed)
- Exact chunk composition served per window (chunk count, bot fraction, hands/chunk).
- Exact competition **aggregation formula** across the 5×24h rounds of a 120h epoch.
- Authoritative window/epoch boundaries and winner settlement (`/internal/competition/*`).
- Whether/when a manifest triggers manual disqualification, and its magnitude.
- Whether the live eval distribution equals the public benchmark distribution.
