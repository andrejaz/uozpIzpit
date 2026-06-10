import yaml
import random


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
    povp_razdalja = 0
    count = 0
    for i in range(len(group)):
        mesto1 = group[i]
        pos1 = locs[mesto1]
        for j in range(i+1,len(group), 1):
            mesto2 = group[j]
            pos2 = locs[mesto2]
            razdalja = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5
            povp_razdalja += razdalja  
            count +=1

    return povp_razdalja / count


def unusual_group_pval(locs, elements):
    p = 0
    dejansko_povp = mean_distance_group_allpairs(locs, elements)
    for i in range(1000):
        random_group = random.sample(list(locs.keys()), len(elements))
        povp_rand = mean_distance_group_allpairs(locs, random_group)
        
        if povp_rand <= dejansko_povp: 
            p+=1
    
    return p/1000


if __name__ == '__main__':
    tsne_xy = read_xy('climate-europe.tab')
    groups = read_groups('groups.yaml')

    for group in groups:
        elements = groups[group]
        print(group, elements)
        #mean_distance_group_allpairs(tsne_xy,group)
        print(unusual_group_pval(tsne_xy, elements))
