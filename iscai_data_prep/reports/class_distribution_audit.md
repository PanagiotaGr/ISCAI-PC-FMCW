# Paired dataset class-distribution audit

**Status:** `PASS_FROZEN`

## Selection

- Training: 72,085
- Validation: 44,097
- Total: 116,182
- Exact combined bytes: 680,625,437,915

## Class-preservation gate

### training

- All observed classes preserved: `True`
- Supported-class ratio gate: `True`
- Overall class gate: `True`
- Missing classes: `[]`

#### context_actors

- `TYPE_CYCLIST`: full=136,755, selected=20,592, share Δ=+0.0118 pp
- `TYPE_PEDESTRIAN`: full=1,670,762, selected=248,733, share Δ=+0.0409 pp
- `TYPE_VEHICLE`: full=16,819,253, selected=2,491,129, share Δ=-0.0527 pp

#### anchor_valid_actors

- `TYPE_CYCLIST`: full=120,343, selected=18,202, share Δ=+0.0146 pp
- `TYPE_PEDESTRIAN`: full=1,368,712, selected=203,919, share Δ=+0.0417 pp
- `TYPE_VEHICLE`: full=15,284,233, selected=2,264,166, share Δ=-0.0563 pp

#### forecasting_target_candidates

- `TYPE_CYCLIST`: full=118,803, selected=17,966, share Δ=+0.0145 pp
- `TYPE_PEDESTRIAN`: full=1,344,282, selected=200,250, share Δ=+0.0404 pp
- `TYPE_VEHICLE`: full=15,144,382, selected=2,243,424, share Δ=-0.0549 pp

#### scenario_presence

- `TYPE_CYCLIST`: full=89,076, selected=13,335, share Δ=+0.1060 pp
- `TYPE_PEDESTRIAN`: full=301,171, selected=44,554, share Δ=-0.0513 pp
- `TYPE_VEHICLE`: full=486,929, selected=72,071, share Δ=-0.0547 pp

### validation

- All observed classes preserved: `True`
- Supported-class ratio gate: `True`
- Overall class gate: `True`
- Missing classes: `[]`

#### context_actors

- `TYPE_CYCLIST`: full=11,735, selected=11,735, share Δ=+0.0000 pp
- `TYPE_PEDESTRIAN`: full=149,683, selected=149,683, share Δ=+0.0000 pp
- `TYPE_VEHICLE`: full=1,517,521, selected=1,517,521, share Δ=+0.0000 pp

#### anchor_valid_actors

- `TYPE_CYCLIST`: full=10,425, selected=10,425, share Δ=+0.0000 pp
- `TYPE_PEDESTRIAN`: full=124,099, selected=124,099, share Δ=+0.0000 pp
- `TYPE_VEHICLE`: full=1,390,024, selected=1,390,024, share Δ=+0.0000 pp

#### forecasting_target_candidates

- `TYPE_CYCLIST`: full=10,275, selected=10,275, share Δ=+0.0000 pp
- `TYPE_PEDESTRIAN`: full=121,837, selected=121,837, share Δ=+0.0000 pp
- `TYPE_VEHICLE`: full=1,377,228, selected=1,377,228, share Δ=+0.0000 pp

#### scenario_presence

- `TYPE_CYCLIST`: full=7,864, selected=7,864, share Δ=+0.0000 pp
- `TYPE_PEDESTRIAN`: full=27,177, selected=27,177, share Δ=+0.0000 pp
- `TYPE_VEHICLE`: full=44,090, selected=44,090, share Δ=+0.0000 pp

## Interpretation

- Complete Scenario records are retained; actors are not filtered from selected records.
- Validation is the complete exact-matching validation intersection.
- Training is deterministic SHA-256 scenario-ID selection.
- The audit is based on non-SDC actors.
- Forecasting target eligibility uses anchor validity and at least two valid causal states.
