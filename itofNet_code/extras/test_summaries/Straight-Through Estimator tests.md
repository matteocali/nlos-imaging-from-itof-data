# Straighth-Through Estimator tests

## Test results on the fixed camera dataset

| attempt | description                                                      |   accuracy   |
| :-----: | :--------------------------------------------------------------: | :----------: |
|    1    |  no IoU loss                                                     |    0.8102    |
|    2    |  standar IoU loss (no STE) (IoU weightet 3)                      |  **0.8622**  |
|    3    |  STE with only linear functions                                  |  **0.8556**  |
|    4    |  STE with only linear functions (IoU weightet 3)                 |    0.8425    |
|    5    |  STE witu only sigmoid functions                                 |    0.8476    |
|    6    |  STE with linear for the cleaning and sigmoid for the threshold  |    0.8492    |
|    7    |  STE parametric version                                          |    0.8490    |

## Test on the full dataset

| attempt | description                                                      |   accuracy   |
| :-----: | :--------------------------------------------------------------: | :----------: |
|    1    |  STE with only linear functions                                  |  **0.7814**  |

## Legend

- **no IoU loss**: Old network architecture based only on the MAE loss;
- **standard IoU loss**: Network architecture based on the MAE loss and the nondifferentiable IoU loss;
- **STE with only linear function**: Network architecture based on the one using the IoU where to deal with the nondifferentiable IoU loss, the STE is used with only linear functions for the backpropagation;
- **STE with only sigmoid functions**: Network architecture based on the one using the IoU where to deal with the nondifferentiable IoU loss, the STE is used with only sigmoid functions for the backpropagation;
- **STE with linear for the cleaning and sigmoid for the threshold**: Network architecture based on the one using the IoU where to deal with the nondifferentiable IoU loss, the STE is used with linear functions for the backpropagation for the cleaning step and sigmoid functions for the backpropagation for the threshold step;
- **STE parametric version**: Network architecture based on the one using the IoU where to deal with the nondifferentiable IoU loss, the STE is used win its parametric version:
    $$
        \begin{cases}
            clean\_itof_{diff} = itof_{predict} + (itof_{hard} - itof_{predict})\text{.detach()} \\
            depth_{diff} = depth + (depth_{hard} - depth)\text{.detach()}
        \end{cases}
    $$
    where:
    $$
        itof_{predict} = \text{prediction of the network}\,, \qquad
        depth = \text{itof2depth}(itof_{predict})\,;
    $$
    $$
        itof_{hard} =
            \begin{cases}
                0 & \text{if } \; itof_{predict} < 0.05 \\
                itof_{pred} & \text{otherwise}
            \end{cases}, \qquad
        depth_{hard} =
            \begin{cases}
                0 & \text{if } \; depth = 0 \\
                1 & \text{otherwise}
            \end{cases}.
    $$
