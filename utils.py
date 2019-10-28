import numpy as np

def pull_strat(true_utility, sigma):
    return np.random.normal(true_utility, sigma)

def get_ks(k, n, m=None, rate=2, type='fraction'):
    """Return k_i's for each round.

    Args:
        k (int): the number of arms wanted at the end
        n (int): the number of arms
        m (int): the total number of rounds. If None, will fit based on type
        rate (float): default 2. 1/rate of remaining arms removed each round if type is fraction
                                 power if type is poly or flipped_poly
        type (string): controls how many arms are kept after each round
            Possible values:
                'fraction' - removes half the arms each round. m=None
                'lin' - k_i's decrease linearly. m required
                'poly' - k_i's decrease polynomially. m, power required
                        pass in power in (0, 1) for convex function from 0 to m
                        pass in power in (1, inf) for concave function from 0 to m
                'flipped_poly' - k_i's as above but rotated diagonally over linear line
    Returns:
        list: a list of k_i's that has a last element K
    """
    K = []
    if type == 'fraction':
        if m is not None:
            raise ValueError('m should not be specified when type is fraction')
        curr = n
        while curr//rate > k:
            K.append(curr//rate)
            curr = curr//rate
        K.append(k)
    elif type == 'lin':
        if m is None:
            raise ValueError('m should be specified when type is poly')
        K = [int(n-(x*((n-k)/m))) for x in range(1, m+1)]
        K[-1] = k
    elif type == 'poly' or type == 'flipped_poly':
        if m is None:
            raise ValueError('m should be specified when type is poly')
        K = [int(n-((((n-k)**(1/rate))/m)*x)**rate) for x in range(1, m+1)]
        K[-1] = k
        if type == 'flipped_poly':
            K.reverse()
            K = [(n-x)+k for x in K][1:] + [k]
    else:
        raise ValueError("Invalid type")
    return K

def get_costs_gains(m, j_rate=1, s_rate=2, type='linear', s_of_j=False,
                    j_of_s=False, predefined=None):
    """Return j_i's and s_i's for each of m rounds.

    Args:
        m (int): number of rounds
        j_rate (int or float): rate that costs grow at
        s_rate (int or float): rate that gains grow at
        type (string): default linear. defines how the rates grow
    Returns:
        list, list: list of j_i's and s_i's respectively
    """
    def enforce_increase(J, S):
        J[0] = 1
        S[0] = 1
        for i in range(1, len(J)):
            if S[i] == J[i]:
                S[i] += 1
            elif S[i] < J[i]:
                raise ValueError('This should never happen, S[i] < J[i]')
        increasing = False
        while not increasing:
            increasing = True
            for i in range(1, len(J)):
                if S[i] <= S[i-1]:
                    S[i] += 1
                    increasing = False
                if J[i] <= J[i-1]:
                    J[i] += 1
                    increasing = False
        for i in range(1, len(J)):
            if S[i] == J[i]:
                S[i] += 1
            elif S[i] < J[i]:
                raise ValueError('This should never happen, S[i] < J[i]')
        return J, S
    # assuming proper usage where only one is true
    # and predefined is an increasing array
    if s_of_j or j_of_s:
        if predefined is None or len(predefined) != m:
            raise ValueError('Need to pass predefined=[m-length array of cost/gains]')
        if s_of_j:
            rate = s_rate
            f = math.ceil
        elif j_of_s:
            rate = j_rate
            f = math.floor
        if type == 'linear':
            arr = [x+rate for x in predefined]
        elif type == 'poly':
            # assuming rate > 1 if s_of_j, else rate in (0, 1)
            arr = [f(x**rate) for x in predefined]
        elif type == 'mult':
            # assuming rate > 1 if s_of_j, else rate in (0, 1)
            arr = [max(f(x*rate), 1) for x in predefined]
        if s_of_j:
            return enforce_increase(predefined, arr)
        else:
            return enforce_increase(arr, predefined)
    if type == 'linear':
        # assuming rates are integers
        if j_rate >= s_rate:
            raise ValueError('j_rate >= s_rate')
        j_rate = int(j_rate)
        s_rate = int(s_rate)
        J = [x for x in range(1, m*j_rate + 1, j_rate)]
        S = [x for x in range(1, m*s_rate + 1, s_rate)]
    elif type == 'poly':
        if j_rate >= s_rate:
            raise ValueError('j_rate >= s_rate')
        J = [int(x**j_rate) for x in range(1, m+1)]
        S = [int(x**s_rate) for x in range(1, m+1)]
        for i in range(1, len(J)):
            if S[i] == J[i]:
                S[i] += 1
            elif S[i] < J[i]:
                raise ValueError('Should not ever get here')
    elif type == 'mult':
        # assuming rates are integers
        J = [1]
        S = [1]
        for i in range(m-1):
            J.append(J[-1]*j_rate)
            S.append(S[-1]*s_rate)
    else:
        raise ValueError('Invalid type')
    return enforce_increase(J, S)