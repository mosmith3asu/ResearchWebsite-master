"""
A comparative analysis of trajectory similarity measures
https://www.tandfonline.com/doi/full/10.1080/15481603.2021.1908927
"""
import warnings

from scipy.spatial.distance import euclidean

import numpy as np
import matplotlib.pyplot as plt
import similaritymeasures
import scipy
from fastdtw import fastdtw
from tslearn.metrics import dtw, dtw_path,dtw_limited_warping_length

T = {# [row col]
    'up': np.array([1, 0]),
    'left': np.array([0, -1]),
    'down': np.array([0, 1]),
    'right': np.array([1, 0]),
    'wait': np.array([0, 0])
}


def main():

    traj0 = np.array([[0,0,0,0,0,0]])
    traj1 = traj0.copy()
    # At0 = [
    #     ['up','wait','wait'],
    #     ['wait','wait','wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['wait','wait','wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #
    # ]
    # At1 = [
    #     ['up', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['up', 'wait', 'wait']
    # ]
    # At0 = [
    #     ['up','wait','wait'],
    #     ['up','wait','wait'],
    #     ['wait','wait','wait'],
    #     ['right', 'wait', 'wait'],
    #     ['right', 'wait', 'wait'],
    #     # ['wait','wait','wait'],
    #     # ['wait', 'wait', 'wait'],
    #     # ['up', 'wait', 'wait'],
    #
    # ]
    # At1 = [
    #     ['right', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['right', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['wait','wait','wait'],
    # ]
    # for actions in At0: traj0 = np.vstack([traj0,transition(traj0[-1,:], actions)])
    # for actions in At1: traj1 = np.vstack([traj1,transition(traj1[-1,:], actions)])
    # eval(traj0, traj1)

    # #########################################
    # traj0 = np.array([[0,0,0,0,0,0]])
    # traj1 = traj0.copy()
    # At0 = [
    #     ['up','wait','wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    # ]
    # At1 = [
    #     ['up', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['up', 'wait', 'wait']
    # ]
    # for actions in At0: traj0 = np.vstack([traj0,transition(traj0[-1,:], actions)])
    # for actions in At1: traj1 = np.vstack([traj1,transition(traj1[-1,:], actions)])
    # eval(traj0, traj1)
    # #######################################
    traj0 = np.array([[0, 0, 0, 0, 0, 0]])
    traj1 = traj0.copy()
    # At0 = [
    #     ['up', 'wait','wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['wait', 'wait', 'wait'],
    #     ['up', 'wait','wait'],
    #     ['down', 'wait', 'wait'],
    # ]
    # At1 = [
    #     ['down', 'wait', 'wait'],
    #     ['down', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #     ['up', 'wait', 'wait'],
    #
    # ]

    At0 = [
        ['left', 'wait', 'wait'],
        ['left', 'wait', 'wait'],
        ['up', 'wait', 'wait'],
        ['up', 'wait', 'wait'],
        ['right', 'wait', 'wait'],
        ['right', 'wait', 'wait'],
    ]
    At1 = [
        ['right', 'wait', 'wait'],
        ['right', 'wait', 'wait'],
        ['wait', 'wait', 'wait'],
        ['up', 'wait', 'wait'],
        ['up', 'wait', 'wait'],
        ['left', 'wait', 'wait'],
        ['wait', 'wait', 'wait'],
        ['left', 'wait', 'wait'],
    ]
    for actions in At0: traj0 = np.vstack([traj0, transition(traj0[-1, :], actions)])
    for actions in At1: traj1 = np.vstack([traj1, transition(traj1[-1, :], actions)])
    eval(traj0, traj1)



def eval(traj0,traj1):
    traj0 = traj0[:,0:4].copy()
    traj1 = traj1[:,0:4].copy()



    # print(traj0); print(traj1)
    print(f'Stats: n0={traj0.shape[0]} \t n1={traj1.shape[0]}')
    print(f'\t| DynamicTimeWarping: {fastdtw(traj0,traj1,dist=2)[0]}')
    print(f'\t| DynamicTimeWarping: {custom_dtw(traj0, traj1, p=2)}')
    print(f'\t| DynamicTimeWarping: {get_dtw(traj0, traj1)[0]}')
    d,dtw_mat = custom_dtw(traj0, traj1,get_matrix=True)
    plt.figure()
    plt.imshow(dtw_mat)
    d, dtw_mat = get_dtw(traj0, traj1)
    plt.figure()
    plt.imshow(dtw_mat)
    plt.show()
    # print(f'\t| DynamicTimeWarping: {get_dtw(traj0, traj1)}')
    # print(fastdtw(traj0,traj1,dist=euclidean)[1])
    # similartyEval = TrajectorySimilarity()
    # print(f'\t| DiscreteFrechet: {similartyEval.DiscreteFrechet(traj0,traj1)}')
    # print(f'\t| DynamicTimeWarping2: {similartyEval.DynamicTimeWarping(traj0,traj1)}')
    # print(f'\t| DirectedHausdorff: {similartyEval.DirectedHausdorff(traj0,traj1)}')
    # print(f'\t| Area: {similartyEval.Area(traj0,traj1)}')
    # print(f'\t| PartialCurveMapping: {similartyEval.PartialCurveMapping(traj0,traj1)}')

def transition(state, actions):
    a1, a2, a3 = actions
    joint_action = np.hstack([T[a1],T[a2],T[a3]])
    new_state = state + joint_action
    return new_state


def custom_dtw(traj0,traj1,window=100,p=2,get_matrix=False):
    """
    sakoe_chiba_radius
    :param window: max diff between indicies
    :param p: norm order
    :return:
    """
    n,m = traj0.shape[0],traj1.shape[0]
    w = np.max([window, abs(n - m)])
    # w = window
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            dtw_matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            cost = np.linalg.norm(traj0[i - 1] - traj1[j - 1],ord=p)
            # cost = abs(traj0[i - 1] - traj1[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min

    dist = dtw_matrix[-1, -1]
    if get_matrix:
        print(dtw_matrix)
        dtw_path = np.zeros(dtw_matrix.shape)
        i,j = 0,0
        n, m =dtw_matrix.shape
        dtw_path[i, j] = 1
        while i!=n-1 or j!=m-1:

            cbelow = (i+1, j)
            cdiag = (i+1, j+1)
            cright = (i , j+1)
            warp_trans = [cbelow,cdiag,cright]

            costs = []
            costs.append(dtw_matrix[cbelow] if (cbelow[0]<n and cbelow[1]<m) else 2*np.max(dtw_matrix))
            costs.append(dtw_matrix[cdiag] if (cdiag[0]<n and cdiag[1]<m) else 2*np.max(dtw_matrix))
            costs.append(dtw_matrix[cright] if (cright[0]<n and cright[1]<m) else 2*np.max(dtw_matrix))
            # print(i, j, np.argmin(costs),costs)


            i,j = warp_trans[np.argmin(costs)]
            dtw_path[i, j] = 1

            # if i == n - 1:  dtw_path[i, j:] = 1;break
            # if j == m - 1:  dtw_path[i:, j] = 1;break

        # print(dtw_path)
        return dist, dtw_path
    else: return dist




def get_dtw(traj0,traj1,max_length=7):
    # cost = dtw(traj0,traj1)
    # cost = dtw(traj0,traj1, global_constraint="sakoe_chiba", sakoe_chiba_radius=5)
    # cost = dtw(traj0,traj1, global_constraint = "itakura", itakura_max_slope = 2.)
    # cost = dtw_limited_warping_length(traj0,traj1, max_length)
    # optimal_path, cost = dtw_path(traj0,traj1, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)

    # https: // tslearn.readthedocs.io / en / stable / user_guide / dtw.html
    ############### SELECTED DTW CONSTRAINT ########################
    optimal_path, cost = dtw_path(traj0,traj1, global_constraint = "itakura", itakura_max_slope = 4.)
    ############### SELECTED DTW CONSTRAINT ########################

    n,m = traj0.shape[0],traj1.shape[0]
    dtw_mat = np.zeros([n+1,m+1])
    dtw_mat[0,0] = 1
    for i,j in optimal_path:
        dtw_mat[i+1,j+1] = 1
    print(optimal_path)
    return cost,dtw_mat


# class TrajectorySimilarity(object):
#     """https://pypi.org/project/similaritymeasures/
#     Objective: Measure spacial and temporal similarity but prioritize spacial
#     """
#     def DiscreteFrechet(self,traj1,traj2):
#         """
#         ISSUE: ONLY MEASURES SPACIAL SIMILARITY
#         Discrete Frechet distance: The shortest distance in-between two curves,
#         where you are allowed to very the speed at which you travel along each curve
#         independently (walking dog problem) """
#         return similaritymeasures.frechet_dist(traj1,traj2)
#     def DirectedHausdorff(self, traj1, traj2):
#         """https://userguide.mdanalysis.org/stable/examples/analysis/trajectory_similarity/psa.html"""
#         d, index1,index2 = scipy.spatial.distance.directed_hausdorff(traj1, traj2)
#         return d
#     def DynamicTimeWarping(self,traj1, traj2):
#         """ Dynamic Time Warpingy (DTW): A non-metric distance between two time-series curves
#         that has been proven useful for a variety of applications
#         PARTIALLY CONSIDERS TIME IMPLICITLY
#         CAN INCREASE CONSIDERATION OF TIME BY LIMITING WARPING
#         """
#         dtw, _ = similaritymeasures.dtw(traj1, traj2)
#         return dtw
#     def Area(self, traj1, traj2):
#         """ Area method: An algorithm for calculating the Area between two curves in 2D space [2]
#         (created specifically for material parameter identification) """
#         return similaritymeasures.area_between_two_curves(traj1, traj2)
#     # def PartialCurveMapping(self, traj1, traj2):
#     #     """ Partial Curve Mapping (PCM) method: Matches the
#     #     area of a subset between the two curves [1]
#     #     (created specifically for material parameter identification)"""
#     #     return similaritymeasures.pcm(traj1, traj2)
#
#



if __name__ == "__main__":
    main()
