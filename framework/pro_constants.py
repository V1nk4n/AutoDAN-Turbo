"""Shared constants for the PRO red-team pipeline (decoupled from paper-specific jargon)."""

# Tier-1 short-circuit score_loss (empty / short / regex refusal). Must stay aligned
# with refusal-streak logic in ``AutoDANTurboPro._run_request_with_repetitions``.
PRO_TIER1_SHORT_CIRCUIT_SCORE_LOSS = 0.0

# Pattern library round index passed to ``PatternManager`` for this pipeline.
PRO_PATTERN_LIBRARY_ROUND = 1

# How many top strategies from the library feed the generator prompt.
PRO_PATTERN_SELECT_TOP_K = 5

# Structured generator contract: exactly three fields (XML tags or JSON keys).
PRO_GENERATOR_THOUGHT_KEY = "Thought"
PRO_GENERATOR_STRATEGY_KEY = "Strategy"
PRO_GENERATOR_RESPONSE_KEY = "Response"
PRO_GENERATOR_STRUCTURED_KEYS = (
    PRO_GENERATOR_THOUGHT_KEY,
    PRO_GENERATOR_STRATEGY_KEY,
    PRO_GENERATOR_RESPONSE_KEY,
)
