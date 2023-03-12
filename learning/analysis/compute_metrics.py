from dtaidistance import dtw_ndim,dtw_barycenter

def get_TrajSimilarity(traj_set1, traj_set2, window=None):
    """ Uses dynamic time warping after a barycenter average to determine warped distance as a metric of trajectory similarity
        packages:  from dtaidistance import dtw_ndim,dtw_barycenter
        from: https://dtaidistance.readthedocs.io/en/latest/modules/dtw_barycenter.html
    :param traj_set1: length i list of [n x m] trajectories where n=time dim and m = state features
    :param traj_set2: length i list of [n x m] trajectories where n=time dim and m = state features
    :param window: Only allow for maximal shifts from the two diagonals smaller than this number.
    :return d:  distance/similarity post warp
    """
    Series1, Series2 = traj_set1, traj_set2

    # Calculate barycenter averages
    dba_Series1 = dtw_barycenter.dba(Series1, c=None, use_c=True, window=window)
    dba_Series2 = dtw_barycenter.dba(Series2, c=None, use_c=True, window=window)

    # Perform DTW on dba_Series
    series1, series2 = dba_Series1, dba_Series2
    d = dtw_ndim.distance(series1, series2)
    # path = dtw_ndim.warping_path(series1, series2)

    # print(f'##### dtaidistance #########')
    # print(f'DTW dist: {d}')
    # print(f'DTW path: {path}')

    """ ---- PLOTTING UTILS ---------
    # Plot Barycenter Averages
    dba_Seriesi = [dba_Series1,dba_Series2]
    fig,axs = plt.subplots(1,2)
    for iax, ax in enumerate(axs):
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Series {iax+1}')
    for iax, series in enumerate([Series1,Series2]):
        for s in series:
            x = s[:,1]; y = -1*s[:,0]
            axs[iax].plot(x,y,lw=1,ls=':')
        x = dba_Seriesi[iax][:, 1]
        y = -1 * dba_Seriesi[iax][:, 0]
        axs[iax].plot(x,y,c='k',lw='2')
    plt.show()


    # Show just dba_Series
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.plot(dba_Series1[:, 1],-1*dba_Series1[:, 0],label='Series 1')
    ax.plot(dba_Series2[:, 1],-1*dba_Series2[:, 0],label='Series 2')
    ax.legend()
    plt.show()



    # Plot DTW matrix path
    n = series1.shape[0]
    m = series2.shape[0]
    dtw_mat = np.zeros([n, m])
    for idxs in path: dtw_mat[idxs] = 1
    plt.imshow(dtw_mat)
    plt.show()
    dtwvis.plot_warping(series1, series2, path, filename="warp.png")

    """

    return d

