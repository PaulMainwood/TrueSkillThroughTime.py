def dict_diff(old, new):
    step = (0., 0.)
    for a in old:
        step = max_tuple(step, old[a].delta(new[a]))
    return step

def max_tuple(t1, t2):
    return max(t1[0], t2[0]), max(t1[1], t2[1])

def gr_tuple(tup, threshold):
    return (tup[0] > threshold) or (tup[1] > threshold)