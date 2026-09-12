# 10×10 Significant-Questions K-Map Design

Date: 2026-09-12  
Status: goal-locked (Approach 1) after unanswered section approvals under active `/goal`  
Related draft: `/Users/sweeden/kagg_defense/kmap_10x10_design_draft.md`

## Goal

Produce a clear **10×10** matrix of **significant questions**, differentially informative vs the **initial terminal configuration** baseline, via architecture training evidence and a **best-of-n** comparison. Deliverables: BoN comparison artifact + winning 10×10 k-map.

## Cell semantics

Each cell: `{qid, text, channel, family, sigma, mean, n, rank_in_family}`  
- Rows (10 families, Gray-adjacent order): time, capital, labor, crops, market, opponent, risk, poison_trust, schedule_flops, fellowship  
- Cols (10): σ-rank deciles within family (0 = highest σ)  
- Every `text` must end with `?`  
- Baseline (idle terminals): empty / σ=0  
- Reject episode `107982856`; trust `107982855` (+ COMPLETE self-play metrics)

## Pipeline

1. Terminal baseline → zero 10×10  
2. Expand question bank to ≥100 probes (10 families × ≥10)  
3. Score on trusted trajectories  
4. Fold top-10/family by σ into matrix  
5. Diff report vs baseline  
6. Best-of-n (`inherit`, `composer-2.5-fast`) → winner package

## Acceptance

- Legible 10×10 JSON + render  
- Non-trivial Δ vs baseline  
- BoN comparison with named winner  
- Training/architecture evidence cited  
