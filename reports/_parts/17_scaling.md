## 17. Data-scaling results

{{table:scaling_results.md}}

![data scaling](../results/plots/data_scaling.png)

**Data scaling.** With the architecture fixed and epochs scaled so small subsets still converge,
EM grows roughly log-linearly with training-set size: 4.2 % (83 examples), 24.7 % (414), 34.5 % (828),
47.1 % (2,070), 54.1 % (4,140), 61.4 % (8,280). The first doublings are worth 10–20 points, the last
two about 7 points each — diminishing returns per example, but no plateau at the full Spider training
set, which suggests that more (or augmented) data would still help this model size.

Protocol: each training subset is an independent uniform random sample (`random.sample`, seed 42) of the training split — subsets are not guaranteed to be nested; smaller subsets get proportionally more epochs (epochs = 20 × min(10, 1/√fraction)) and validation cadence so that every run converges, and every run is selected on the same 379 validation examples and evaluated on the full dev set.
