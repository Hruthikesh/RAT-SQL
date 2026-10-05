### Primary error category (Spider dev failures, our measurements)

| Category | full | no_grammar | no_linking | simple | vanilla |
|---|---|---|---|---|---|
| grammar_error | 0 | 754 | 2 | 0 | 0 |
| nested_query_error | 56 | 9 | 86 | 94 | 78 |
| schema_linking_error | 107 | 22 | 229 | 680 | 366 |
| wrong_table | 52 | 12 | 68 | 108 | 129 |
| wrong_join | 43 | 4 | 28 | 30 | 69 |
| wrong_column | 85 | 8 | 91 | 32 | 63 |
| aggregation_error | 22 | 1 | 9 | 1 | 6 |
| where_error | 21 | 7 | 22 | 2 | 12 |
| group_by_error | 11 | 0 | 3 | 2 | 3 |
| order_by_error | 15 | 1 | 10 | 0 | 9 |
| wrong_value | 19 | 11 | 17 | 0 | 7 |
| semantic_reasoning_error | 1 | 0 | 0 | 0 | 0 |
| **total failures** | 432 | 829 | 565 | 949 | 742 |
