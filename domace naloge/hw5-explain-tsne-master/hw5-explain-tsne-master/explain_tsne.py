import yaml


def read_groups(fn):
    with open(fn, 'r', encoding='utf8') as file:
        return yaml.safe_load(file)


def read_xy(fn):
    locs = {}
    with open(fn, 'r', encoding='utf8') as f:
        next(f)  # skip first line
        for l in f:
            l = l.strip()
            if l:
                name, x, y = l.split('\t')
                locs[name] = [float(x), float(y)]
    return locs


def mean_distance_group_allpairs(locs, group):
    pass


def unusual_group_pval(locs, elements):
    pass


if __name__ == '__main__':
    tsne_xy = read_xy('climate-europe.tab')
    groups = read_groups('groups.yaml')

    for group in groups:
        elements = groups[group]
        print(group, elements)
        print(unusual_group_pval(tsne_xy, elements))
