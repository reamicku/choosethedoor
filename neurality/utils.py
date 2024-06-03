def generate_pairs(N):
    pairs = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            pairs.append((i, j))
    return pairs


def generate_non_overlapping_pairs(N):
    pairs = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            pairs.append((i, j))
            pairs.append((j, i))
    return pairs
