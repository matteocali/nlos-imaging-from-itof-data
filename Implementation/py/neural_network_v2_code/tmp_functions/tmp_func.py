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


def plot_plane(depth):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    focal = 0.5 * 320 / np.tan(0.5 * 60 * np.pi / 180)

    x = np.repeat(np.expand_dims(np.array((range(depth.shape[0]))), 0), depth.shape[1], axis=0)
    y = np.repeat(np.expand_dims(np.array((range(depth.shape[1]))), 1), depth.shape[0], axis=1)
    z = depth
    z[:, 239] = depth[:, 238]
    z = np.flip(np.flip(z.T, axis=0), axis=1)
    z = depth_radial2cartesian(z, focal)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.elev= 5
    ax.plot_surface(X=x, Y=y, Z=z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


def plane_fitting(depth):
    """
    Fits a plane to the given points.
    """

    import numpy as np

    y = np.repeat(np.expand_dims(np.array((range(depth.shape[1]))), 1), depth.shape[0], axis=1)
    x = np.repeat(np.expand_dims(np.array((range(depth.shape[0]))), 0), depth.shape[1], axis=0)

    z = depth
    z[:, 239] = depth[:, 238]
    z = np.flip(np.flip(z.T, axis=0), axis=1)
    z_cart = depth_radial2cartesian(z, focal)

    # Fit a plane to the points
    A = np.array([x, y, np.ones(len(x))]).T
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    # Calculate the normal vector
    n = np.array([a, b, -1])
    n = n / np.linalg.norm(n)

    # Calculate the distance from the origin
    d = np.dot(n, np.array([0, 0, 0]))

    return a, b, c, d


def plot_fitted_plane(X, Y, Z, a, b, c, d):
    """
    Plots the given points and the fitted plane.
    :param X: x coordinates of the points
    :param Y: y coordinates of the points
    :param Z: z coordinates of the points
    :param a: a parameter of the plane equation
    :param b: b parameter of the plane equation
    :param c: c parameter of the plane equation
    :param d: d parameter of the plane equation
    :return: None
    """

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Plot the points
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)

    # Plot the plane
    xx, yy = np.meshgrid(range(320), range(240))
    zz = (a * xx + b * yy + d) / c
    ax.plot_surface(xx, yy, zz, alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
