def plot_depth_diff(depths, dist, freqs=['20e6', '50e6', '60e6'], path=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, a in enumerate(ax):
        diff = depths[i + 1] - depths[0]
        img = a.matshow(diff.T, cmap='jet')
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        img.set_clim(-0.05, 0.05)
        f.colorbar(img, cax=cax)
        a.set_title(f'Depth difference between {freqs[i + 1]} and {freqs[0]}')
    f.suptitle(f"Depth difference comparisonfor an object at distace {dist}m")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        exit()
    plt.show()


def plot_ampl_ratio(ampls, dist, freqs=['20e6', '50e6', '60e6'], path=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, a in enumerate(ax):
        ratio = np.divide(ampls[i + 1], ampls[0], where=ampls[0] != 0)
        ratio = np.where(ampls[0] == 0, 0, ratio)
        ratio = np.where(ratio > 1.2, 1.2, ratio)  # Ignore outliers
        img = a.matshow(ratio.T, cmap='jet')
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #img.set_clim(0, 1)
        f.colorbar(img, cax=cax)
        a.set_title(f'Amplitude ratio between {freqs[i + 1]} and {freqs[0]}')
    f.suptitle(f"Amplitude ratio comparison for an object at distace {dist}m")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        exit()
    plt.show()