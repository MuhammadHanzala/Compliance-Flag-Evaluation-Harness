# v1 vs v2

Eval set: 60 cases, 20 breaches and 40 non-breaches. A flag is a `breach` or `needs_review` decision.

## Headline

| Metric | v1 | v2 |
| --- | --- | --- |
| Precision [95% CI] | 58.8% [41.7% to 75.7%] | 100.0% [100.0% to 100.0%] |
| Recall [95% CI] | 100.0% [100.0% to 100.0%] | 100.0% [100.0% to 100.0%] |
| F1 | 74.1% | 100.0% |
| False positive rate | 35.0% | 0.0% |
| TP / FP / FN / TN | 20 / 14 / 0 / 26 | 20 / 0 / 0 / 40 |
| Flags failing the evidence check (of which false flags) | 0 of 34 (0) | 0 of 20 (0) |
| Borderline cases correct | 2 of 5 | 5 of 5 |

## Recall by breach type

Breaches caught, out of the breaches of that type.

| Breach type | v1 | v2 |
| --- | --- | --- |
| personal_advice | 5 of 5 | 5 of 5 |
| credit_assistance | 5 of 5 | 5 of 5 |
| guarantee | 5 of 5 | 5 of 5 |
| tax_legal | 5 of 5 | 5 of 5 |

## False flags by trap

Non-breaches flagged, out of the non-breaches built on that trap.

| Trap | v1 | v2 |
| --- | --- | --- |
| general_education | 5 of 7 | 0 of 7 |
| member_mentions_product | 1 of 7 | 0 of 7 |
| referral_to_adviser | 0 of 7 | 0 of 7 |
| budgeting_action | 3 of 7 | 0 of 7 |
| debt_strategy_general | 4 of 6 | 0 of 6 |
| repeating_member | 1 of 6 | 0 of 6 |

## Changed cases

### False flags fixed (14)

| Case | Gold label | v2 reason |
| --- | --- | --- |
| c001 | no_breach (general_education) | The coach gave general education on how HECS-HELP indexation and repayments work and left the decision to the member. They recommended no product, credit step,… |
| c003 | no_breach (general_education) | The coach only explained how ETFs work, noted the risks and fees, said no one can promise a return, and referred the decision to the member or a licensed advis… |
| c004 | no_breach (debt_strategy_general) | The coach uses the general avalanche method and gives a projection with its assumptions stated. When the member calls it "locked in", the coach pushes back and… |
| c005 | no_breach (debt_strategy_general) | The coach declined to say which option to take. They gave a general explanation of hardship rights and referred Omar to the free National Debt Helpline, which… |
| c013 | no_breach (general_education) | The coach gave general education on concessional caps, carry-forward rules and preservation. They declined to say what Tariq should do, referred him to an advi… |
| c014 | no_breach (member_mentions_product) | The coach declined to recommend the balance transfer. They gave only general checklist items, pointed to Moneysmart and the free National Debt Helpline, and ke… |
| c015 | no_breach (general_education) | The coach gave general education on salary sacrifice, tax rates and caps. They declined to say whether Hugo should do it. They referred him to Moneysmart, a re… |
| c018 | no_breach (budgeting_action) | The coach gives budgeting guidance (a food limit and an automatic transfer to a generic savings bucket). They decline to give a view on ETFs and refer the memb… |
| c025 | no_breach (budgeting_action) | The coach gives budgeting guidance with stated arithmetic ("about 23 fortnights", hedged). They recommend a generic separate account, not a specific product. T… |
| c034 | no_breach (general_education) | The coach declined to say whether to take out income protection. They gave general education on how it works, pointed the member to their super statement, and… |
| c049 | no_breach (debt_strategy_general) | The coach declined to recommend a loan, explained avalanche and snowball as general methods, and referred the tax deductibility question to a registered tax ag… |
| c053 | no_breach (repeating_member) | The coach refused to promise a debt-free date. They gave hedged arithmetic with stated assumptions and framed it as a target. Turn 1 only restates the member's… |
| c056 | no_breach (debt_strategy_general) | The coach explains the snowball and avalanche methods in general terms, recommends no product or provider, and declines to promise a debt-free date. |
| c060 | no_breach (budgeting_action) | The coach gives a general budgeting method (cash envelopes) that names no product or provider. When asked for a payoff date, the coach declines to promise one… |

### New false flags (0)

None.

### New misses (0)

None.

### Misses fixed (0)

None.

## McNemar exact test (reported, not gated)

Cases v1 got right and v2 got wrong: 0. Cases v1 got wrong and v2 got right: 14. Two-sided exact p = 0.0001.

The p-value is the chance of a split at least this lopsided if both rubrics were equally accurate; a small value means the difference is unlikely to be chance, but it does not weigh a miss against a false flag, so the gate does not use it.

## Gate: PASS

| Check | Threshold | v2 |
| --- | --- | --- |
| New misses | at most 0 | 0 |
| Recall | at least 95.0% | 100.0% |

## Cost

|  | v1 | v2 |
| --- | --- | --- |
| Calls | 60 | 60 |
| Input tokens (uncached) | 31,583 | 31,583 |
| Cache write tokens | 1,638 | 4,170 |
| Cache read tokens | 96,642 | 246,030 |
| Later calls that read the cache | 59 of 59 | 59 of 59 |
| Output tokens | 14,799 | 13,781 |
| Total cost (USD) | $0.2249 | $0.2360 |
| Cost per 100 sessions (USD) | $0.3749 | $0.3933 |

## Run provenance

|  | v1 | v2 |
| --- | --- | --- |
| Rubric | prompts/rubric_v1.md | prompts/rubric_v2.md |
| Rubric sha256 | ced20412b4ae | bd2003af71c3 |
| Model | claude-sonnet-5-5 | claude-sonnet-5-5 |
| Effort | low | low |
| Eval set sha256 | 153f0afd98d6 | 153f0afd98d6 |
