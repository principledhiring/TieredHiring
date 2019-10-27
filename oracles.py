import numpy as np

def top_k_utility(util, ind):
    w = 0
    for i in ind:
        w += util[int(i)]
    return w


def top_k_oracle(util, k, A):
    ind = np.argsort(-1*util[A])
    return A[ind[0:k]]

def c_top_k_oracle(util, k, A, A_i):
    ind = np.argsort(-1*util[A])
    return np.append(A_i, A[ind[0:(k-len(A_i))]])

def submodular_max_utility(util, ind, groups, arms_groups):
    add = np.zeros(len(groups))
    for i in groups:
        group_ind = (arms_groups[ind] == i)
        if group_ind.shape[0] > 0:
            add[i] = np.sum(util[ind[group_ind]])
            if add[i] < 0:
                add[i] = 0
    add = np.sqrt(add)
    return np.sum(add)

def submodular_max_oracle(util, k, A, groups, arms_groups):
    indices = set()
    look_set = set(A)

    mx = 0
    mx_ind = -1
    while len(indices) < k:  # Want to make sure we have things to look at
        for i in look_set:
            # if i not in indices:
            val = submodular_max_utility(util, np.array(list(indices) + [i]),
                                         groups, arms_groups)
            if val > mx:
                mx = val
                mx_ind = i
        if mx_ind > -1:
            indices.add(mx_ind)
            look_set.remove(mx_ind)
            mx_ind = -1
        else:
            break
    return np.array(list(indices))

def c_submod_oracle(util,k,A,A_i,groups,arms_groups):
    indices_set = set(A_i)
    look_set = set(A)
    gp_set = {}
    arms_groups = np.array(arms_groups)
    gp_i = {}
    gp_sum = {}
    util = np.array(util)
    for gp in groups:

        tmp = np.array(list(set(A[arms_groups[A] == gp])-indices_set))
        # print(tmp)
        if tmp.shape[0] > 0:
            gp_set[gp] = tmp[np.argsort(util[tmp])]
        else:
            gp_set[gp] = np.array([])
        tmp = np.array(list(indices_set - set(A[arms_groups[A] == gp])))
        if len(tmp) == 0:
            gp_sum[gp] = 0
        else:
            gp_sum[gp] = np.sum(util[tmp])
        gp_i[gp] = -1
    # print (gp_set)

    while len(indices_set) < k:
        mx = 0
        mx_gp = -1
        for gp in groups:
            if np.abs(gp_i[gp]) <= gp_set[gp].shape[0]:
                gp_sum[gp] += util[gp_set[gp][gp_i[gp]]]
                # print(util)
                u = 0
                for gp1 in groups:
                    u += np.sqrt(gp_sum[gp1])
                # print(u)
                if  mx <= u:
                    mx = u
                    mx_gp = gp
                gp_sum[gp] -= util[gp_set[gp][gp_i[gp]]]
        # print(mx)
        if mx_gp == -1:
            break
        else:
            indices_set.add(gp_set[mx_gp][gp_i[mx_gp]])
            gp_sum[mx_gp] += util[gp_set[mx_gp][gp_i[mx_gp]]]
            gp_i[mx_gp] -= 1
    return np.array(list(indices_set))