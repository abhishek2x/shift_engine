import math
from typing import Tuple, List, Optional

from .math_utils import log_normal_pdf, log_sum_exp, compute_posterior_log_space


class RegimeDetector:
    """
    The core engine. Wraps the recursive Bayesian math into a clean interface.

    Accepts market returns one at a time via `update()`, internally manages
    all log-space computations, and returns human-readable probabilities.

    Optionally accepts a Transition Matrix (HMM) to add regime
    stickiness and prevent signal flickering on noisy ticks.

    Usage:
        detector = RegimeDetector(
            bull_mean=0.0005, bull_std=0.01,
            bear_mean=-0.001, bear_std=0.025
        )
        bull_prob, bear_prob = detector.update(-0.03)
        print(f"Bull: {bull_prob:.1%}, Bear: {bear_prob:.1%}")
    """

    def __init__(
        self,
        bull_mean: float,
        bull_std: float,
        bear_mean: float,
        bear_std: float,
        initial_bull_prob: float = 0.5,
        transition_matrix: Optional[List[List[float]]] = None,
    ):
        """
        :param bull_mean: Mean return of the Bull (Low Vol) regime.
        :param bull_std: Std deviation of the Bull regime.
        :param bear_mean: Mean return of the Bear (High Vol) regime.
        :param bear_std: Std deviation of the Bear regime.
        :param initial_bull_prob: Starting belief for Bull (default 0.5 = neutral).
        :param transition_matrix: Optional 2x2 matrix for HMM regime stickiness.
            Format: [[P(Bull→Bull), P(Bull→Bear)],
                      [P(Bear→Bull), P(Bear→Bear)]]
            Example: [[0.95, 0.05], [0.10, 0.90]]
            If None, no transition dampening is applied (pure Bayes).
        """
        # --- Distribution Parameters ---
        self._bull_mean = bull_mean
        self._bull_std = bull_std
        self._bear_mean = bear_mean
        self._bear_std = bear_std

        # --- Internal State (Log-Space) ---
        self._log_prior_bull = math.log(initial_bull_prob)
        self._log_prior_bear = math.log(1.0 - initial_bull_prob)
        self._initial_bull_prob = initial_bull_prob

        # --- Transition Matrix (HMM) ---
        # Pre-compute log of transition probabilities to avoid repeated log() calls
        if transition_matrix is not None:
            self._log_trans = [
                [math.log(transition_matrix[r][c]) for c in range(2)]
                for r in range(2)
            ]
        else:
            self._log_trans = None

        # --- History Tracking ---
        self._history: List[Tuple[float, float]] = []
        self._tick_count: int = 0

    def update(self, market_return: float) -> Tuple[float, float]:
        """
        Process a single market return tick and update regime beliefs.

        Internally performs the full Bayes cycle:
        1. Calculates likelihoods via log_normal_pdf()
        2. (Optional) Applies HMM transition matrix to adjust priors
        3. Computes posterior via compute_posterior_log_space()
        4. Updates internal state for the next tick

        :param market_return: The observed return (e.g., -0.03 for a -3% move).
        :returns: (bull_probability, bear_probability) as floats between 0 and 1.
        """
        # --- Step 1: Likelihoods ---
        # Call log_normal_pdf() twice with different regime parameters
        ll_bull = log_normal_pdf(market_return, self._bull_mean, self._bull_std)
        ll_bear = log_normal_pdf(market_return, self._bear_mean, self._bear_std)

        # --- Step 2: Prepare Priors (with optional Transition Matrix) ---
        if self._log_trans is not None:
            # HMM Prediction Step:
            # P(S_t=Bull) = P(Bull|Bull)*P(Bull_{t-1}) + P(Bull|Bear)*P(Bear_{t-1})
            # P(S_t=Bear) = P(Bear|Bull)*P(Bull_{t-1}) + P(Bear|Bear)*P(Bear_{t-1})
            predicted_bull = log_sum_exp([
                self._log_trans[0][0] + self._log_prior_bull,   # Bull stayed Bull
                self._log_trans[1][0] + self._log_prior_bear,   # Bear switched to Bull
            ])
            predicted_bear = log_sum_exp([
                self._log_trans[0][1] + self._log_prior_bull,   # Bull switched to Bear
                self._log_trans[1][1] + self._log_prior_bear,   # Bear stayed Bear
            ])
            prior = (predicted_bull, predicted_bear)
        else:
            # Pure Bayes (no transition dampening)
            prior = (self._log_prior_bull, self._log_prior_bear)

        # --- Step 3: Posterior (Bayes Update) ---
        post_bull, post_bear = compute_posterior_log_space(
            prior_log_probs=prior,
            log_likelihoods=(ll_bull, ll_bear),
        )

        # --- Step 4: Update Internal State ---
        self._log_prior_bull = post_bull
        self._log_prior_bear = post_bear
        self._tick_count += 1

        # Convert log-probabilities to human-readable decimals
        bull_prob = math.exp(post_bull)
        bear_prob = math.exp(post_bear)

        self._history.append((bull_prob, bear_prob))

        return bull_prob, bear_prob

    def reset(self, initial_bull_prob: Optional[float] = None) -> None:
        """Resets beliefs back to the initial state for re-running backtests."""
        prob = initial_bull_prob if initial_bull_prob is not None else self._initial_bull_prob
        self._log_prior_bull = math.log(prob)
        self._log_prior_bear = math.log(1.0 - prob)
        self._history = []
        self._tick_count = 0

    @property
    def current_belief(self) -> Tuple[float, float]:
        """Returns the current (bull_prob, bear_prob) as human-readable decimals."""
        return math.exp(self._log_prior_bull), math.exp(self._log_prior_bear)

    @property
    def history(self) -> List[Tuple[float, float]]:
        """Returns the full list of (bull_prob, bear_prob) for every processed tick."""
        return self._history

    @property
    def tick_count(self) -> int:
        """Returns the total number of ticks processed so far."""
        return self._tick_count

    def __repr__(self) -> str:
        bull, bear = self.current_belief
        return (
            f"RegimeDetector(ticks={self._tick_count}, "
            f"bull={bull:.1%}, bear={bear:.1%}, "
            f"hmm={'ON' if self._log_trans else 'OFF'})"
        )
