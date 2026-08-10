# Paired WOMD/WOMD-LiDAR inventory

**Mode:** read-only

## Disk

- Total: 967.48 GiB
- Used: 715.94 GiB
- Free: 251.54 GiB

## Local branches

- `training`: 424.11 GiB
- `validation`: 38.35 GiB
- `testing`: 29.72 GiB
- `training_20s`: 86.17 GiB
- `validation_interactive`: 37.72 GiB
- `testing_interactive`: 29.21 GiB

## Split inventory

### training

- Local motion scenarios: 486,995
- Local shard bytes: 424.11 GiB
- Remote LiDAR objects: 487,054
- Remote LiDAR total: 2.18 TiB
- Exact paired scenarios: 486,988
- Paired fraction of local: 99.9986%
- Exact paired combined bytes: 2.59 TiB

### validation

- Local motion scenarios: 44,097
- Local shard bytes: 38.35 GiB
- Remote LiDAR objects: 44,102
- Remote LiDAR total: 202.29 GiB
- Exact paired scenarios: 44,097
- Paired fraction of local: 100.0000%
- Exact paired combined bytes: 240.62 GiB

## Capacity estimates

Each plan reserves 2% of calculated pair capacity for packaging/index overhead.

### reserve_200_gib
- `strict_no_original_deletion`:
  - all paired validation does not fit
- `transactional_replace_training_validation_only`:
  - count-max upper bound: 102,428 pairs, 503.72 GiB
  - deterministic hash-order plan: 92,296 pairs, 503.72 GiB
- `transactional_replace_core_and_remove_unused_branches`:
  - count-max upper bound: 139,158 pairs, 682.88 GiB
  - deterministic hash-order plan: 125,153 pairs, 682.88 GiB

### reserve_250_gib
- `strict_no_original_deletion`:
  - all paired validation does not fit
- `transactional_replace_training_validation_only`:
  - count-max upper bound: 92,135 pairs, 454.72 GiB
  - deterministic hash-order plan: 83,325 pairs, 454.72 GiB
- `transactional_replace_core_and_remove_unused_branches`:
  - count-max upper bound: 129,234 pairs, 633.88 GiB
  - deterministic hash-order plan: 116,182 pairs, 633.88 GiB

### reserve_300_gib
- `strict_no_original_deletion`:
  - all paired validation does not fit
- `transactional_replace_training_validation_only`:
  - count-max upper bound: 81,699 pairs, 405.72 GiB
  - deterministic hash-order plan: 74,345 pairs, 405.72 GiB
- `transactional_replace_core_and_remove_unused_branches`:
  - count-max upper bound: 119,223 pairs, 584.88 GiB
  - deterministic hash-order plan: 107,189 pairs, 584.88 GiB

## Transactional staging

- Source shards containing at least one pair: 1,150
- Maximum all-paired output from one source shard: 3.06 GiB
- P95 all-paired output from one source shard: 2.85 GiB

