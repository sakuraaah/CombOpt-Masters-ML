MIN_PROBLEM_SIZE = 10
MAX_PROBLEM_SIZE = 100

MIN_RUNTIME = 0.25
MAX_RUNTIME = 6.0

GROWTH_POWER = 2.2


def get_max_runtime(problem_size: int) -> float:
    clamped_problem_size = min(max(problem_size, MIN_PROBLEM_SIZE), MAX_PROBLEM_SIZE)
    normalized_size = (clamped_problem_size - MIN_PROBLEM_SIZE) / (
        MAX_PROBLEM_SIZE - MIN_PROBLEM_SIZE
    )
    scaled_runtime = MIN_RUNTIME + (MAX_RUNTIME - MIN_RUNTIME) * (
        normalized_size**GROWTH_POWER
    )

    return round(scaled_runtime, 2)
