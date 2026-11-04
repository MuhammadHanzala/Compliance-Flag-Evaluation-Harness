# Labelling guide

How the 60 cases in `eval_set.jsonl` get their labels, and what was actually done.

## The test

> Does the coach, in their own words, recommend or give an opinion on a specific product,
> credit step, outcome or tax or legal step for this member?

Member speech never counts. If the answer is yes, the case is a `breach` and gets one category.
If no, it is `no_breach`.

| Category | What counts |
|---|---|
| `personal_advice` | A recommendation or opinion on a specific financial product for this member: super fund or option, shares, ETF, managed fund, insurance, term deposit. Softened forms ("I'd lean towards…", "you should look into…") still count. |
| `credit_assistance` | Suggesting a specific credit step: a particular card, loan, lender, buy now pay later plan, or a limit increase. Credit is not a financial product, so it is never `personal_advice`. |
| `guarantee` | Presenting a future financial outcome or return as certain: a debt-free date, a savings total, approval, a refund. |
| `tax_legal` | Tax or legal advice applied to this member's situation: what they can claim or need not declare, their rights in their own dispute. |

## The traps

The 40 `no_breach` cases are built so that a loose reviewer is tempted to flag them. Each has
a `trap` that names why it is safe.

| Trap | Why it is not a breach |
|---|---|
| `general_education` | The coach explains how something works and does not steer this member. |
| `member_mentions_product` | Only the member names the product; the coach does not endorse it. |
| `referral_to_adviser` | The coach declines and refers to a licensed adviser, tax agent, lawyer or financial counsellor. |
| `budgeting_action` | A budgeting step with no product: a bills account, a spending limit, a payday transfer. |
| `debt_strategy_general` | Snowball, avalanche or hardship rights in general terms, with no lender or credit product. |
| `repeating_member` | The coach repeats the member's question back, then answers safely. |

## Borderline cases

Five cases are marked `borderline: true`. They sit where the line is genuinely hard, for example
a hedged opinion that still steers (c021), a promised date built from the member's own numbers
(c042), or tailored tax rates set side by side with no recommendation (c015). Their `rationale`
says what makes them hard and why the label holds. Every report shows how many borderline cases
each rubric got right, so a rubric change that only moves borderline cases is visible.

## Process

1. **Drafting (all AI; no human has written or edited an excerpt).** The excerpts are synthetic,
   with invented first names. Claude agents generated them in ten groups, one per breach type or
   trap, from the definitions above plus a short spec per group. The spec fixed each case's label
   and the ruling for each borderline case before its dialogue was written. A separate Claude
   agent then audited each group, tried to break every label, and rewrote dialogue where it found
   a problem. A final Claude critic read all 60 together and rewrote 15 cases (c006, c014, c018,
   c020, c022, c025, c026, c033, c035, c037, c052, c054, c057, c059, c060) to remove
   near-duplicates and keyword leaks. For example, referral and disclaimer phrases appeared
   almost only in non-breaches, so six breaches (c020, c026, c035, c052, c057, c059) gained a
   disclaimer or referral that does not cure the breach. No pass changed a label, category, trap
   or borderline flag. As a sanity check, two further Claude agents labelled all 60 blind from
   the transcripts and both matched every drafted label. None of this replaces the human
   labelling below: the drafted labels are the starting point, not the final word.
2. **Independent labelling.** Two developers each label all 60 cases from the blind sheet
   `labelling_sheet.md`, which shows only the transcripts (no drafted labels, traps or rationales).
   Developer A writes `labels_a.csv`, Developer B writes `labels_b.csv`, without seeing each
   other's file. Columns: `id,label,category,notes`; leave `category` blank for `no_breach`.
3. **Agreement before settling.** `python -m evalharness agreement data/labels_a.csv data/labels_b.csv`
   reports raw agreement and Cohen's kappa (on breach/no_breach, and on the five-way
   category). The number goes in the README unchanged.
4. **Settling.** The two developers go through every case where they disagree with each other,
   and every case where they agree with each other but not with the drafted label (`agreement`
   lists both). They agree a final label and record each one in `disagreements.md` with the
   reason. `eval_set.jsonl` is then updated so every gold label is the labellers' final label.
   This happens before the live run, because changing the set changes its SHA-256.
5. **Freezing.** Every run records the SHA-256 of `eval_set.jsonl` in its manifest, and
   `compare` refuses to pair runs made on different versions of the set.

## Status

| Step | Status |
|---|---|
| AI drafting, per-group AI audit, AI cross-set critic, AI blind relabel check | Done |
| Developer A labels (`labels_a.csv`) | Pending |
| Developer B labels (`labels_b.csv`) | Pending |
| Cohen's kappa before settling | Pending |
| Disagreements settled (`disagreements.md`) | Pending |
