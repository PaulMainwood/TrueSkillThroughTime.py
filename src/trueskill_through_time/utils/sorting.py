def podium(xs):
    return sortperm(xs)

def sortperm(xs, reverse=False):
    return [i for (v, i) in sorted(
        ((v, i) for (i, v) in enumerate(xs)),
        key=lambda t: t[0],
        reverse=reverse
    )]

