Our decoder predicts join conditions explicitly; RAT-SQL instead derives FROM and ON from the
foreign-key graph when unparsing. Re-serialising the *same decoded trees* with foreign-key join
inference (`ratsql/sql/join_inference.py`: FROM completed with the tables of all referenced columns,
ON clauses from shortest FK paths) raises the controlled FULL model's EX from 63.7 % to 66.6 % and its
executable-SQL rate from 92.4 % to 98.6 %, and the final model's EX from 66.9 % to 69.9 %. The vanilla
model gains the most (+17.5 EX), confirming that its main deficit is the relational schema structure.
All main-table numbers are *as decoded*; this variant is reported separately because it is a
post-processing step, not part of the learned model.
