import math
from typing import List, Tuple

def log_sum_exp(log_probs: List[float]) -> float:
    """
    Computes log(sum(exp(log_probs))) safely to prevent underflow/overflow.
    Essential for marginalizing probabilities in continuous log-space updates.
    """
    max_log = max(log_probs)
    # If all log_probs are -inf, the exp sum is 0, so log sum is -inf.
    if math.isinf(max_log) and max_log < 0:
        return float('-inf')
        
    sum_res = 0.0
    for lp in log_probs:
        sum_res += math.exp(lp - max_log)
        
    return max_log + math.log(sum_res)

def log_normal_pdf(x: float, mean: float, std_dev: float) -> float:
    """
    Calculates the log-likelihood of a return given a N(mean, std_dev) distribution.
    Formula: ln( P(x|mu, sigma) ) = -0.5 * ln(2 * pi * sigma^2) - ((x - mu)^2) / (2 * sigma^2)
    """
    variance = std_dev ** 2
    log_scale = -0.5 * math.log(2 * math.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return log_scale + exponent

def compute_posterior_log_space(
    prior_log_probs: Tuple[float, float],
    log_likelihoods: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Executes a single recursive Bayes update step precisely in Log-Space.
    
    Inputs:
    - prior_log_probs: (log_P_Bull, log_P_Bear)  <- Our belief before the tick
    - log_likelihoods: (log_P_Tick_given_Bull, log_P_Tick_given_Bear)
    
    Returns:
    - posterior_log_probs: (log_P_Bull_given_Tick, log_P_Bear_given_Tick)
    """
    # 1. Calculate Unnormalized Numerator: ln(Likelihood) + ln(Prior)
    unnorm_bull = log_likelihoods[0] + prior_log_probs[0]
    unnorm_bear = log_likelihoods[1] + prior_log_probs[1]
    
    # 2. Calculate the Marginal Likelihood (Denominator) using Log-Sum-Exp Trick:
    # ln( P(Data) ) = ln( exp(unnorm_bull) + exp(unnorm_bear) )
    marginal_log_prob = log_sum_exp([unnorm_bull, unnorm_bear])
    
    # 3. Normalize to get Posterior: ln(Posterior) = ln(Numerator) - ln(Denominator)
    posterior_bull = unnorm_bull - marginal_log_prob
    posterior_bear = unnorm_bear - marginal_log_prob
    
    return posterior_bull, posterior_bear
