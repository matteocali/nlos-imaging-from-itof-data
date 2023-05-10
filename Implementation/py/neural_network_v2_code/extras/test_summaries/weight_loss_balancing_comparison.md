<!-- markdownlint-disable MD012 MD033 -->

# WEIGHT LOSS PERFORMANCE COMPARISON

## Comparison description

In this comparison, it will be analyzed the effect of the different balancing of the two main classes (*object*, *background*) on the overall performance of the network.

To perform the balancing during the dataset loading procedure the script automatically computes the ratio between the number of background pixels and the number of objects ones, throughout the whole dataset:
$$r = \frac{\# of background's pixels}{\# of object pixels} = \frac{p_{BG}}{p_{OBJ}} \approx 30.$$
Then, during the MAE (Mean Absolute Error) loss computation, before computing the mean over the whole image matrix, the loss value of each object pixel (considering the *GT* as a reference to locate the object position) is multiplied by the aforementioned ratio $r$.
By doing that we're essentially balancing an unbalanced dataset giving the same weight to both classes. This turns out to not be the best solution since it is better to reduce the disparity between the background and the object pixels. For that reason we've decided to multiply the ratio $r$ by a factor $\alpha$ in the range $[\frac{1}{10}, \frac{1}{9}, \frac{1}{8}, \frac{1}{7}, \frac{1}{6}, \frac{1}{5}, \frac{1}{4}]$.

Doing that the final MAE loss (for each pixel) is computed as follows:
$$\alpha \cdot r \cdot loss_{OBJ};$$
$$1 \cdot loss_{BG}.$$

The following are shown the results obtained by testing the network with different values of $\alpha$.

## Results

<div class=result_table>

| attempt \# | $\alpha$ | avarage MAE (depth) |  min MAE (depth) | max MAE (depth) |avarge IoU (depth) |min IoU (depth) |  max IoU (depth) |
| :--------: | :------: | :-----------------: | :--------------: | :-------------: | :---------------: | :------------: | :--------------: |
| 1          | 1/10     | **0.1168**          | 0.043            | 0.2467          | **0.2884**        | 0.0            | 0.6873           |
| 2          | 1/9      | **0.0632**          | 0.0205           | 0.1867          | **0.5302**        | 0.0            | 0.8203           |
| 3          | 1/8      | **0.0698**          | 0.0129           | 0.1552          | **0.4838**        | 0.0            | 0.8842           |
| 4          | 1/7      | **0.0585**          | 0.0236           | 0.1444          | **0.5433**        | 0.0            | 0.5485           |
| 5          | 1/6      | **0.0777**          | 0.0241           | 0.1781          | **0.4851**        | 0.0            | 0.7742           |
| 6          | 1/5      | **0.0807**          | 0.04             | 0.2104          | **0.4964**        | 0.0            | 0.7711           |
| 7          | 1/4      | **0.0707**          | 0.0273           | 0.2108          | **0.5163**        | 0.0            | 0.7932           |

</div>


<!-- HTML styles -->
<style>
    .result_table {
        text-align: center;
    }
    .result_table th {
        word-wrap: break-word;
        text-align: center;
    }
    .result_table tr:nth-child(4) { background: green; }
</style>
