
class CACO(object):

    def __init__(delta, epsilon, sigma, arms, K, S, J, oracle, oracle_utility, oracle_args=[],
        save_data=True, save_decisions=False):
        """Initializes a Caco MAB instance.
        Args:
            arms: A list of arms.
        """