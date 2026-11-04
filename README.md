# compliance-flag-eval

An LLM reviews financial coaching transcripts (Australia) and flags compliance breaches. It flags
too much. This repo is the loop that fixes that without trading false flags for missed breaches:

1. a set of 60 synthetic excerpts, pinned by SHA-256 in every run, with AI-drafted labels and
   independent two-developer labelling pending ([`data/`](data/)),
2. two rubric prompts: a loose baseline and a tightened version ([`prompts/`](prompts/)),
3. a runner that scores a rubric on the set and saves every prediction with its tokens and cost,
4. metrics that report false flags and missed breaches together,
5. a compare step that lists every changed case and a gate that blocks a change that misses a
   breach the old prompt caught.

## Results

Live runs on `claude-sonnet-5-5` (effort `low`, adaptive thinking), one run per rubric, on eval set
sha256 `153f0afd98d6`. Copied from [`reports/report_v1_vs_v2.md`](reports/report_v1_vs_v2.md), which has
every changed case.

| Metric | v1 (loose) | v2 (tightened) |
|---|---|---|
| Precision [95% CI] | 58.8% [41.7% to 75.7%] | 100.0% [100.0% to 100.0%] |
| Recall [95% CI] | 100.0% [100.0% to 100.0%] | 100.0% [100.0% to 100.0%] |
| False positive rate | 35.0% | 0.0% |
| TP / FP / FN / TN | 20 / 14 / 0 / 26 | 20 / 0 / 0 / 40 |
| Flags failing the evidence check | 0 of 34 | 0 of 20 |
| Borderline cases correct | 2 of 5 | 5 of 5 |
| New misses (v1 → v2) | | 0 |
| Gate (max 0 new misses, recall ≥ 95%) | | **PASS** |
| Later calls that read the prompt cache | 59 of 59 | 59 of 59 |
| Total cost (USD) | $0.2249 | $0.2360 |
| Cost per 100 sessions (USD) | $0.3749 | $0.3933 |

v2 fixed all 14 of v1's false flags (5 `general_education`, 4 `debt_strategy_general`,
3 `budgeting_action`, 1 `member_mentions_product`, 1 `repeating_member`) and introduced no new
misses and no new false flags. McNemar exact p = 0.0001 (reported, not gated).

**How far to trust this.** The gold labels are still the AI-drafted ones (human labelling is
pending), the cases and v2 came out of the same AI workflow, and two blind AI labellers already
agreed with every drafted label, so the set is clear-cut. A perfect v2 score shows the tightened
rubric matches these labels; it is not evidence it would be perfect on real transcripts. If the
human labellers change any label, both runs are repeated on the new set.

Label agreement before settling (Cohen's kappa): **pending**, see [How the labels were made](#how-the-labels-were-made).

## What the reviewer should look for

| Question | Where it is answered |
|---|---|
| Are false flags and missed breaches measured together? | Every report puts precision, recall and false positive rate side by side, with recall per breach type and false flags per trap ([`metrics.py`](evalharness/metrics.py)). |
| Can a prompt change be blocked? | `compare` exits 1 when the new rubric misses a breach the old one caught, or recall drops below the floor ([`compare.py`](evalharness/compare.py), `test_gate_fails_when_a_new_miss_appears`). |
| Is it the same set every run? | Each run records the SHA-256 of the eval set and the rubric; `compare` refuses runs made on different sets. |
| Are the labels explained? | [`data/LABELLING.md`](data/LABELLING.md): the test, who labels, how disagreements are settled, and a status table (human labelling is pending). |
| Is cost under control? | The rubric is cached; the first case runs alone to write the cache; tokens and USD are saved per call and summed per run. |

## How it works

```
data/eval_set.jsonl ─┐
prompts/rubric_vN.md ┴─► run (LangGraph)                                     ─► runs/vN/predictions.jsonl
                         START → warm_cache (1 case, writes the prompt cache)     runs/vN/manifest.json
                               → fan_out ─Send─► assess × N  (≤ --workers at once)
                               → END
                         assess = Claude call → evidence check → prediction + cost

runs/v1 + runs/v2 ─► compare ─► reports/report_v1_vs_v2.md + gate exit code
```

- **Classifier** ([`classifier.py`](evalharness/classifier.py)). One call per excerpt. The rubric is
  the `system` prompt, marked `cache_control: ephemeral`. The answer comes back through one
  strict tool, `record_assessment`: `decision`, `category`, `evidence_quote`, `coach_turn_index`,
  `reason`. The client is injected, so the tests use a fake and need no key.
- **Evidence check** ([`evidence.py`](evalharness/evidence.py)). A flag must quote words the coach
  actually said in the turn it names, after normalising case, whitespace and quote marks. A flag
  that fails stays a flag, marked `needs_review`. It is never turned into `no_breach`, because
  that could hide a real breach. Reports show how many flags failed and how many of those were
  false flags.
- **Runner** ([`runner.py`](evalharness/runner.py), [`graph.py`](evalharness/graph.py)). Each
  prediction is appended to `predictions.jsonl` as soon as its call returns, with the model,
  rubric SHA-256, all four token counts, cost and latency. Cases already saved are skipped, so a
  rerun of a finished run makes no calls. A case that fails (API error after the SDK's retries,
  refusal, no tool call) is reported and not saved, so the next rerun retries just that case.
- **Metrics** ([`metrics.py`](evalharness/metrics.py)). Confusion matrix, precision, recall, F1,
  false positive rate, recall by breach type, false flags by trap, borderline cases, and 95%
  bootstrap intervals on precision and recall (1,000 resamples, fixed seed). A run that does not
  cover every case is refused.
- **Compare and gate** ([`compare.py`](evalharness/compare.py)). Pairs the two runs case by case
  and lists false flags fixed, new false flags, new misses and misses fixed. The McNemar exact
  test is reported but never decides the gate. The gate fails when new misses exceed
  `--max-new-misses` (default 0) or recall falls below `--recall-floor` (default 0.95).

## Run it

### With Docker

```bash
cp .env.example .env          # then put your key after ANTHROPIC_API_KEY=
mkdir -p runs reports         # otherwise Docker creates them owned by root
export UID GID=$(id -g)       # so files in runs/ and reports/ are owned by you

docker compose run --rm harness validate data/eval_set.jsonl
docker compose run --rm harness agreement data/labels_a.csv data/labels_b.csv
docker compose run --rm harness run --rubric prompts/rubric_v1.md --name v1
docker compose run --rm harness run --rubric prompts/rubric_v2.md --name v2
docker compose run --rm harness compare runs/v1 runs/v2 --max-new-misses 0 --recall-floor 0.95

docker compose --profile dev run --rm check          # ruff + pytest, no key needed
```

Only `ANTHROPIC_API_KEY` is passed into the container, at run time. It is never in the image.
Compose rebuilds the image on each run (seconds when cached), so it never runs stale code.
`data/`, `prompts/` and `config/` are mounted read-only, so edits on the host need no rebuild.

### With uv

```bash
uv sync
cp .env.example .env          # then put your key after ANTHROPIC_API_KEY=

uv run python -m evalharness validate data/eval_set.jsonl
uv run python -m evalharness agreement data/labels_a.csv data/labels_b.csv
uv run --env-file .env python -m evalharness run --rubric prompts/rubric_v1.md --name v1
uv run --env-file .env python -m evalharness run --rubric prompts/rubric_v2.md --name v2
uv run python -m evalharness compare runs/v1 runs/v2 --max-new-misses 0 --recall-floor 0.95

uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Only `run` needs the key; `--env-file .env` loads it. `run` takes `--model` (default
`claude-sonnet-5-5`), `--effort` (`low`, `medium` or `high`; default `low`) and `--workers`
(default 4). A run directory is tied to one rubric, model, effort
and eval set; change any of them and the runner asks for a new `--name`.

## How the labels were made

The full process is in [`data/LABELLING.md`](data/LABELLING.md). In short: the test is *does the
coach, in their own words, recommend or give an opinion on a specific product, credit step,
outcome or tax or legal step for this member?* Member speech never counts.

What has been done so far:

- The 60 excerpts are synthetic (invented first names, no real transcripts) and entirely
  AI-generated: Claude agents wrote them from the definitions and a per-group spec that fixed
  each label in advance, a separate Claude agent audited each group and could rewrite dialogue,
  and a final Claude critic rewrote 15 cases to remove near-duplicates and keyword leaks. No
  human has written or edited an excerpt. Two further Claude agents labelled all 60 blind and
  matched every drafted label; that is a sanity check, not the human labelling.
- **Pending:** two developers label all 60 cases independently from the blind sheet
  [`data/labelling_sheet.md`](data/labelling_sheet.md) (transcripts only) into `labels_a.csv` and
  `labels_b.csv`; kappa before settling is reported here; every disagreement, and every case
  where both differ from the drafted label, is settled and logged in
  [`data/disagreements.md`](data/disagreements.md).

The set: 20 breaches (5 each of `personal_advice`, `credit_assistance`, `guarantee`,
`tax_legal`) and 40 non-breaches built as traps a loose prompt flags (`general_education`,
`member_mentions_product`, `referral_to_adviser`, `budgeting_action`, `debt_strategy_general`,
`repeating_member`). Five cases are marked borderline with the reason. Case ids were assigned
after a fixed-seed shuffle, so their order says nothing about the label.

## Cost

- The rubric sits in `system` with `cache_control: ephemeral`. `claude-sonnet-5-5` caches
  prompts from 512 tokens; `rubric_v2` is well past that, and in the live run `rubric_v1` cached too, because the tool
  definition counts towards the cached prefix.
- The first case runs alone to write the cache; the rest then run in parallel and read it.
  The run summary and the compare report show how many later calls read the cache.
- Prices live in [`config/pricing.toml`](config/pricing.toml): $2 input and $10 output per million
  tokens, cache writes 1.25× input, cache reads 0.05× input. Checked against the
  [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) on 2026-11-02;
  re-check before each live run.
- Each prediction stores its cost; reports show total cost per run and cost per 100 sessions.

## Design decisions

- **Spec corrections for `claude-sonnet-5-5`.** The model rejects `temperature` (and the
  anthropic SDK 2.x no longer accepts it), rejects a forced `tool_choice`, and cannot turn
  thinking off. So the classifier sends no sampling parameters, uses `tool_choice: auto` with a
  `strict` tool plus an explicit instruction to call it, and runs adaptive thinking at a fixed,
  recorded effort. A reply without the tool call is a failure, not a guess.
- **LangGraph** orchestrates the run (warm the cache on one case, fan the rest out with `Send`,
  cap concurrency). It is a deliberate choice and lives in one file, [`graph.py`](evalharness/graph.py);
  Claude itself is called through the official `anthropic` SDK.
- **Both rubrics share a byte-identical Output section.** Only the review criteria differ, so the
  comparison measures the criteria, not the formatting instructions.
- **How blind `rubric_v2` is.** A separate Claude agent wrote it without access to `data/`, from
  the brief's definitions plus tie-breakers set by the same AI workflow that specified the cases,
  so it shares their trap list and the rulings behind borderline cases c004, c015 and c025. Its
  example lines were then checked against every eval sentence, the ones that resembled eval cases
  were rewritten onto scenarios no case uses, and [`tests/test_rubrics.py`](tests/test_rubrics.py)
  fails if an example drifts back towards an eval sentence.

## Known limits

- **Small, synthetic set.** 60 cases give wide confidence intervals (5 breaches per type), and
  synthetic excerpts are cleaner than real sessions.
- **One run per rubric.** Sampling cannot be fixed on this model, so reruns can differ. A
  production version would repeat runs and report the spread.
- **One AI workflow specified both the cases and the tightened rubric.** v2 never saw the
  excerpts, but it shares their trap list and three borderline rulings. A held-out set written
  by someone else is the real test.
- **The set may be easier than real sessions.** Two blind Claude labellers matched all 60
  drafted labels, so the cases are clear-cut for a strong model; real transcripts will be messier.
- **Refused or cut-off calls** are billed by the API but not counted in the run cost, because
  they return no usable prediction. They are reported as failures.
- **Legal framing is plain-English and simplified.** The rubrics are an engineering sample, not
  legal advice.

## Repo layout

```
config/pricing.toml        prices per model
data/                      eval set, labelling guide, labeller CSVs, disagreements, blind sheet
prompts/                   rubric_v1.md (loose), rubric_v2.md (tightened)
evalharness/               schema, classifier, evidence, graph, runner, metrics, compare, cost, report, cli
tests/                     pytest, fake Claude client, no key needed
runs/                      one folder per live run: manifest + predictions.jsonl
reports/                   compare reports, e.g. report_v1_vs_v2.md
Dockerfile, compose.yaml   runtime image and a dev image that runs the checks
.github/workflows/ci.yml   ruff + pytest on Python 3.11 and 3.15, plus a Docker build
```
