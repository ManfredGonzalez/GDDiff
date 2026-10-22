import pandas as pd, os, random
W = '/home/tico/Desktop/master_research/DeiDDPG/exp/robustness_work'
sub = pd.read_csv(f'{W}/robust_inputs/subset.csv')
random.seed(1)
for s in ['rafd', 'lfw']:
    names = sorted(random.sample(sorted(sub[sub.set == s].img), 50))
    for occ in ['occl_eyes', 'occl_lower', 'occl_patch']:
        d = f'{W}/robust_inputs/{occ}100/{s}'
        os.makedirs(d, exist_ok=True)
        for n in names:
            dst = f'{d}/{n}.jpg'
            if not os.path.exists(dst):
                os.symlink(f'{W}/robust_inputs/{occ}/{s}/{n}.jpg', dst)
