import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold


def ensure_group_disjoint(train_groups,test_groups):
    if set(train_groups)&set(test_groups):
        raise ValueError("Group leakage: train and held-out groups overlap")


def grouped_splits(groups, *, strategy="leave_one_group_out", n_splits=3):
    g=np.asarray(groups,dtype=object)
    if g.ndim!=1 or len(set(g))<2:
        raise ValueError("At least two nonempty source groups required")
    if strategy=="leave_one_group_out":
        splitter=LeaveOneGroupOut()
    elif strategy=="group_kfold":
        if not isinstance(n_splits,int) or isinstance(n_splits,bool) or n_splits<2 or n_splits>len(set(g)):
            raise ValueError("GroupKFold n_splits must be 2..number of training groups")
        splitter=GroupKFold(n_splits)
    else:
        raise ValueError("Only explicitly grouped CV strategies are supported")
    seen=np.zeros(len(g),int)
    splits=[]
    for train,test in splitter.split(np.zeros((len(g),1)),groups=g):
        ensure_group_disjoint(g[train],g[test])
        if not len(train) or not len(test):
            raise ValueError("Empty partition")
        seen[test]+=1
        splits.append((train,test))
    if not (seen==1).all():
        raise ValueError("Outer CV must predict every sample exactly once")
    return splits
