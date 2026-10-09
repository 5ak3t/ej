"""Grouped K-fold: when the training pool has many groups, held-out units become blocks of groups (deterministic sha256
assignment, with a fallback for new groups) instead of single groups; with few groups nothing changes."""
import hashlib

import student_dec as DEC

K = 8
MAX_LOGO = 12
SUB = K // len(DEC.FOLDS)
PREFIX = 'kfold/'


def _h(g):
    return int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)


def block(g):
    """Block label of group g (nested in its encoder fold)."""
    return f'{PREFIX}{SUB * DEC.fold_of(g) + _h(g) % SUB}'


def fold_of(label):
    """Encoder fold of a block label or of a plain group name."""
    return int(label[len(PREFIX):]) // SUB if label.startswith(PREFIX) else DEC.fold_of(label)


def many(groups):
    return len(set(groups)) > MAX_LOGO


def relabel(items):
    """(items with 'group' = held-out unit label, real group per item, unit_of map or None). With <= MAX_LOGO groups: unchanged."""
    real = [i['group'] for i in items]
    if not many(real):
        return items, real, None
    unit = {g: block(g) for g in sorted(set(real))}
    return [dict(i, group=unit[i['group']]) for i in items], real, unit


def _test():
    groups = sorted(set().union(*DEC.FOLDS)) + [f'sgd/service_{n}' for n in range(40)]
    items = [{'group': g, 'y': 0} for g in groups]
    it2, real, unit = relabel(items)
    assert real == groups and unit is not None and len({i['group'] for i in it2}) <= K
    td = [g for g in groups if g.startswith('typed-decisions/')]
    assert len({unit[g] for g in td}) == len(td) == 3  # TD workflows in separate blocks
    assert all(fold_of(unit[g]) == DEC.fold_of(g) for g in groups)  # blocks nest in encoder folds
    small = [{'group': g} for g in sorted(set().union(*DEC.FOLDS))]
    assert relabel(small)[0] is small and relabel(small)[2] is None
    assert fold_of('typed-decisions/customer_service') == 1
    print('student_kf ok', sorted(set(unit.values())))


if __name__ == '__main__':
    _test()
