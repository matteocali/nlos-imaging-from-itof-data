# Results analysis 21/02/2023

## Legend

* **`custom_losses`**: the loss function used are the `BalancedMAELoss` and the `BalancedBCELoss`
* **`detach`**: the predicted `mask` is detached from the backpropagation graph before beeing multiplied by the predicted `depth`
* **`hard_mask`**: the predicted `mask` is thresholded to 0 or 1 before beeing multiplied by the predicted `depth` (the mask loss is still computed on the original mask)
* **`gt_only`**: the loss over the predicted `depth` is computed only on the pixels where the ground truth `mask` is 1
* **`n_out_channel`**: the number of output channels of the last convolutional layer of the U-Net network

## Results

| attempt | custom_losses | detach | hard_mask | gt_only | n_out_channel | Depth loss | Mask loss  |
| :-----: | :-----------: | :----: | :-------: | :-----: | :-----------: | :--------: | :--------: |
| 1       | **x**         | **x**  |           |         | 8             | 0.24       | 0.20       |
| 2       | **x**         | **x**  |           |         | 16            | 0.19       | ***0.16*** |
| 3       | **x**         | **x**  | **x**     |         | 8             | 0.21       | 0.20       |
| 4       | **x**         | **x**  | **x**     |         | 16            | 0.21       | 0.29       |
| 5       | **x**         | **x**  | **x**     | **x**   | 8             | ***0.12*** | 0.21       |
| 6       | **x**         | **x**  | **x**     | **x**   | 16            | 0.34       | 0.25       |
|         |               |        |           |         |               |            |            |

## Comments

| Attempts | Additional notes |
| :------: | :--------------- |
| 1-2      | The predicted `mask` of `attempt 1` is a bit smaller wrt the one of `attempt 2`, and so cloaser to the target one. The predicted `mask` is also the best one among the other tests|
| 3-4      | Visually they reach similar performance both for `mask` and `depth` prediction. Note that `attempt 3` has a lower `mask` error (delta = 0.2) while the depth loss has almost the same behaviours|
| 5-6      | The prediction of this two attemts are really similar but `attempt 6` need more time toreach slightly worst performance. The predicted `depth`, visually, is overall the best one. On the other hand this approach give back the worst predicted `mask` |
|          |                  |  
