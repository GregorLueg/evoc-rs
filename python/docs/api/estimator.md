# Estimators

`EVoC` runs the whole pipeline on the CPU. `EVoCGpu` moves the kNN stage onto
the GPU and is only present in a build carrying the `gpu` feature.

::: evoc_rs.estimator
