### Schema-graph join inference at serialisation time (Spider dev, our measurements)

| Model | Variant | EM (%) | EX (%) | SQL valid (%) | Rewritten queries |
|---|---|---|---|---|---|
| full | as decoded | 61.4 | 63.7 | 92.4 | 0 |
| full | + FK join inference | 62.8 | 66.6 | 98.6 | 156 |
| vanilla | as decoded | 33.1 | 31.4 | 54.7 | 0 |
| vanilla | + FK join inference | 37.5 | 48.9 | 94.0 | 530 |
| no_linking | as decoded | 48.7 | 53.0 | 88.0 | 0 |
| no_linking | + FK join inference | 50.9 | 57.8 | 97.1 | 200 |
| simple | as decoded | 9.6 | 10.0 | 31.7 | 0 |
| simple | + FK join inference | 10.4 | 20.6 | 90.9 | 709 |
| final | as decoded | 64.9 | 66.9 | 90.0 | 0 |
| final | + FK join inference | 65.7 | 69.9 | 96.4 | 169 |
