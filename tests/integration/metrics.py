def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    return edit_distance(reference, hypothesis) / max(len(reference), 1)
