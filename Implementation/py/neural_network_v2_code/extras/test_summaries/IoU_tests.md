<!-- markdownlint-disable MD012 MD033 -->

# IOU TESTS

## Tests explanation

Considering the tests performed on the weights of the loss in `weight_loss_balancing_comparison.md` we've selected as the best option the one using $\alpha = 1/7$. In addition to the weighted MAE loss, we've also added an IoU loss, which is defined as:
$$loss_{IoU} = 1 - IoU(pred, GT).$$
Also for that loss we can give more or less weight to it so we've performed some tests to see which is the best option for the $\lambda$ value to use:
$$loss = loss_{MAE} + \lambda loss_{IoU}.$$

## Tests results

<div class=result_table>

| attempt \# | $\lambda$ | avarage MAE (depth) |  min MAE (depth) | max MAE (depth) |avarge IoU (depth) |min IoU (depth) |  max IoU (depth) |
| :--------: | :-------: | :-----------------: | :--------------: | :-------------: | :---------------: | :------------: | :--------------: |
| 1          | 1.0       | **0.0551**          | 0.013            | 0.1457          | **0.762**         | 0.4749         | 0.9325           |
| 2          | 1.5       | **0.053**           | 0.0102           | 0.1623          | **0.7648**        | 0.4779         | 0.9473           |
| 3          | 2.0       | **0.0531**          | 0.0097           | 0.1796          | **0.7622**        | 0.4702         | 0.9507           |
| 4          | 2.5       | **0.0562**          | 0.0113           | 0.1806          | **0.7542**        | 0.4689         | 0.9377           |
| 5          | 3.0       | **0.0498**          | 0.0089           | 0.1823          | **0.7742**        | 0.4755         | 0.9578           |
| 6          | 5.0       | **0.0552**          | 0.0153           | 0.1845          | **0.7662**        | 0.4739         | 0.9225           |

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
    .result_table tr:nth-child(5) { background: green; }
</style>
