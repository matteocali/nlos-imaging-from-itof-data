import numpy as np
import h5py
from tqdm import trange
from matplotlib import pyplot as plt
import sys
sys.path.append("../training/src/")
import PredictiveModel_hidden as PredictiveModel
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Function for transient data prediction on the full image


def plot_results(out_path, name, gt_depth_data, gt_alpha_data, pred_depth, pred_alpha):
    # Compute the MAE on the final images
    pred_depth_masked = pred_depth * gt_alpha_data
    gt_depth_masked = gt_depth_data * gt_alpha_data
    # MAE on the alpha mask
    pred_alpha_masked_ones = pred_alpha * gt_alpha_data
    num_ones = np.sum(gt_alpha_data)
    alpha_mae_obj = np.sum(np.abs(pred_alpha_masked_ones - gt_alpha_data)) / num_ones
    pred_alpha_masked_zeros = pred_alpha * (1 - gt_alpha_data)
    num_zeros = np.sum(1 - gt_alpha_data)
    alpha_mae_bkg = np.sum(np.abs(pred_alpha_masked_zeros - np.zeros(gt_alpha_data.shape, dtype=np.float32))) / num_zeros
    alpha_mae = np.sum(alpha_mae_obj + alpha_mae_bkg) / 2
    # MAE on the depth map
    pred_depth_masked_ones = pred_depth * gt_alpha_data
    depth_mae_obj = np.sum(np.abs(pred_depth_masked_ones - gt_depth_data)) / num_ones
    pred_depth_masked_zeros = pred_depth * (1 - gt_alpha_data)
    depth_mae_bkg = np.sum(np.abs(pred_depth_masked_zeros - np.zeros(gt_alpha_data.shape, dtype=np.float32))) / num_zeros
    depth_mae = np.sum(depth_mae_obj + depth_mae_bkg) / 2

    # Plot the results
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(name)

    img0 = ax[0, 0].matshow(gt_depth_masked, cmap='jet')
    img0.set_clim(np.min(gt_depth_data), np.max(gt_depth_data))
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax)
    ax[0, 0].set_title("Ground truth depth map")
    ax[0, 0].set_xlabel("Column pixel")
    ax[0, 0].set_ylabel("Row pixel")

    img1 = ax[0, 1].matshow(pred_depth_masked, cmap='jet')
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax)
    img1.set_clim(np.min(gt_depth_data), np.max(gt_depth_data))
    ax[0, 1].set_title("Predicted depth map")
    ax[0, 1].set_xlabel("Column pixel")
    ax[0, 1].set_ylabel("Row pixel")

    img2 = ax[0, 2].matshow(pred_depth_masked - gt_depth_masked, cmap='jet')
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img2, cax=cax)
    max_cbar_val = np.max(np.abs(pred_depth_masked - gt_depth_masked))
    max_cbar_val = np.ceil(max_cbar_val * 100) / 100
    img2.set_clim(-max_cbar_val, max_cbar_val)
    box_style = dict(boxstyle="round", fc="w", ec="black", alpha=0.9)
    ax[0, 2].text(20, 20, f"MAE: {str(round(depth_mae, 3))}", ha='left', va='top', fontsize=11, color='black', bbox=box_style)
    ax[0, 2].set_title("Difference of the depth maps")
    ax[0, 2].set_xlabel("Column pixel")
    ax[0, 2].set_ylabel("Row pixel")

    img3 = ax[1, 0].matshow(gt_alpha_data, cmap='jet')
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img3, cax=cax)
    ax[1, 0].set_title("Ground truth alpha map")
    ax[1, 0].set_xlabel("Column pixel")
    ax[1, 0].set_ylabel("Row pixel")

    img4 = ax[1, 1].matshow(pred_alpha, cmap='jet')
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img4, cax=cax)
    img4.set_clim(np.min(gt_alpha_data), np.max(gt_alpha_data))
    ax[1, 1].set_title("Predicted alpha map")
    ax[1, 1].set_xlabel("Column pixel")
    ax[1, 1].set_ylabel("Row pixel")

    img5 = ax[1, 2].matshow(pred_alpha - gt_alpha_data, cmap='jet')
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img5, cax=cax)
    max_cbar_val = np.max(np.abs(pred_alpha - gt_alpha_data))
    max_cbar_val = np.ceil(max_cbar_val * 100) / 100
    img5.set_clim(-max_cbar_val, max_cbar_val)
    box_style = dict(boxstyle="round", fc="w", ec="black", alpha=0.9)
    ax[1, 2].text(20, 20, f"MAE: {str(round(alpha_mae, 3))}", ha='left', va='top', fontsize=11, color='black', bbox=box_style)
    ax[1, 2].set_title("Difference of the alpha maps")
    ax[1, 2].set_xlabel("Column pixel")
    ax[1, 2].set_ylabel("Row pixel")

    plt.tight_layout()
    plt.savefig(f"{out_path}/{name}_PLOTS.svg")
    plt.close()


def test_img(attempt_name, weight_names, P, freqs, out_path, lr, plot=False, loss_fn="mae", n_single_layers=None, fl_scale=False, fil_dir=8,
             dim_t=2000, dts_path=None, dropout=None):

    ff = freqs.shape[0]
    s_pad = int((P - 1) / 2)

    with h5py.File(dts_path, "r") as f:
        v_in = f["raw_itof"][:]
        gt_alpha = f["gt_alpha"][:]
        gt_depth = f["gt_depth"][:]
        names = f["names"][:]

    dim_dataset = v_in.shape[0]

    # Load the model
    net = PredictiveModel.PredictiveModel(name=f'test_net_{attempt_name}', dim_b=dim_dataset, freqs=freqs, P=P,
                                          saves_path='./saves', dim_t=dim_t, fil_size=fil_dir, lr=lr,
                                          loss_name=loss_fn, single_layers=n_single_layers, dropout_rate=dropout)

    # Load the weights
    net.DirectCNN.load_weights(weight_names[0])

    for i in trange(dim_dataset, desc="Testing", unit="img", leave=True, unit_scale=True, mininterval=0.1, smoothing=0.3, total=dim_dataset):
        name = names[i].decode("ascii", "ignore")

        itof_data = v_in[i, ...]
        gt_alpha_data = gt_alpha[i, ...]
        gt_alpha_data = np.swapaxes(gt_alpha_data, 0, 1)
        gt_depth_data = gt_depth[i, ...]
        gt_depth_data = np.swapaxes(gt_depth_data, 0, 1)

        # Scaling
        if fl_scale:
            norm_factor = np.sqrt(itof_data[..., 0]**2 + itof_data[..., ff]**2)
            norm_factor = norm_factor[..., np.newaxis]

            # Scale all factors
            itof_data /= norm_factor
            itof_data = itof_data[np.newaxis, ...]
            itof_data = np.pad(itof_data, pad_width=[[0, 0], [s_pad, s_pad], [s_pad, s_pad], [0, 0]], mode="reflect")

        # Make prediction
        pred_depth, pred_alpha = net.DirectCNN(itof_data, training=False)
        pred_depth = np.squeeze(pred_depth.numpy())
        pred_depth = np.swapaxes(pred_depth, 0, 1)
        pred_alpha = np.squeeze(pred_alpha.numpy())
        pred_alpha = np.swapaxes(pred_alpha, 0, 1)
        pred_alpha[np.where(pred_alpha <= 0.5)] = 0
        pred_alpha[np.where(pred_alpha > 0.5)] = 1

        if plot:
            plot_results(out_path, name, gt_depth_data, gt_alpha_data, pred_depth, pred_alpha)

        with h5py.File(f"{out_path}/{name}_TEST.h5", "w") as f:
            f.create_dataset("depth_map", data=pred_depth)
            f.create_dataset("alpha_map", data=pred_alpha)
            f.create_dataset("depth_map_gt", data=gt_alpha_data)
            f.create_dataset("alpha_map_gt", data=gt_depth_data)
