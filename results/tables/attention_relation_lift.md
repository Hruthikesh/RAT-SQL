### Attention lift by relation type (1 = uniform attention; head-averaged; our measurements)

| Relation | Meaning | rat L1 | rat last | vanilla L1 | vanilla last |
|---|---|---|---|---|---|
| qq_dist_0 | question -> itself | 1.44 | 1.87 | 2.59 | 1.04 |
| qq_dist_1 | question -> next token | 1.29 | 3.42 | 2.44 | 1.38 |
| qc_em | question -> exactly-matched column | 1.98 | 3.30 | 2.87 | 2.51 |
| qc_pm | question -> partially-matched column | 3.57 | 1.79 | 1.56 | 1.48 |
| qc_vem | question -> value-matched column | 0.53 | 0.90 | 0.76 | 2.12 |
| qc_default | question -> unlinked column | 0.94 | 0.29 | 0.56 | 0.83 |
| qt_em | question -> exactly-matched table | 2.32 | 4.03 | 1.11 | 1.27 |
| qt_default | question -> unlinked table | 0.89 | 1.63 | 0.40 | 1.25 |
| ct_primary_key | column -> its table (PK) | 4.05 | 2.85 | 0.44 | 1.85 |
| ct_belongs_to | column -> its table | 1.82 | 2.79 | 0.79 | 1.77 |
| ct_default | column -> other table | 2.06 | 1.05 | 0.69 | 1.46 |
| cc_fk_forward | column -> FK target column | 1.35 | 5.75 | 1.00 | 0.73 |
| cc_same_table | column -> same-table column | 1.86 | 0.81 | 0.71 | 0.90 |
| cc_default | column -> unrelated column | 0.50 | 0.65 | 0.72 | 0.74 |
| tt_fk_forward | table -> FK-linked table | 1.38 | 2.31 | 0.62 | 1.65 |
| tt_default | table -> unrelated table | 4.45 | 1.81 | 0.80 | 1.30 |
