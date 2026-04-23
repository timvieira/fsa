from fsa import FSA, zero, one
from fsa.fsa import eps

import pytest


MIN_ALGOS = ['min_brzozowski', 'min_fast', 'min_faster']


def run_min(m, algo):
    return getattr(m, algo)()


def assert_is_dfa(m):
    "A minimized DFA should have exactly one start (or be empty) and no nondeterministic arcs."
    # empty languages may have 0 start states
    assert len(m.start) <= 1, f'expected at most one start state, got {len(m.start)}'
    for i in m.nodes:
        for a, targets in m.edges[i].items():
            assert a != eps, f'minimized DFA should have no epsilon arcs, found {i} -ε-> ...'
            assert len(targets) == 1, (
                f'minimized DFA should be deterministic; {i} -{a}-> {targets}'
            )


def assert_iso(a, b, msg=''):
    """Direct DFA isomorphism check.

    Uses `_dfa_isomorphism` instead of `.equal()`, because `.equal()` re-runs
    `min_faster` on both sides — which would hide bugs in the other algorithms
    (min_brzozowski, min_fast) by silently re-minimizing their outputs before
    comparing. We want to verify the raw outputs themselves.
    """
    assert a._dfa_isomorphism(b), f'not isomorphic{(": " + msg) if msg else ""}:\n{a}\nvs\n{b}'


def _check_all(m, expected_size=None):
    "Run m through every minimization algo and check consistency."
    results = {algo: run_min(m, algo) for algo in MIN_ALGOS}

    # each result is a DFA
    for algo, r in results.items():
        assert_is_dfa(r)

    # Cross-check each algorithm against every other using direct isomorphism.
    # If all three outputs are pairwise isomorphic, then:
    #   (a) they recognize the same language, and
    #   (b) they have the same number of states.
    # No call to .equal() / .min() is needed — that would re-run min_faster
    # and mask divergence.
    algos = list(results.keys())
    for i, a in enumerate(algos):
        for b in algos[i+1:]:
            assert_iso(results[a], results[b], msg=f'{a} vs {b} on {m}')

    # Additionally check each algorithm's output is language-equivalent to the
    # original input. .equal() is fine here: it minimizes both sides with
    # min_faster and compares — so this catches bugs in min_faster directly,
    # and for the other algorithms we combine it with the isomorphism check
    # above (if min_faster(input) iso min_X(input), then min_X preserves L).
    for algo, r in results.items():
        assert m.equal(r), f'{algo} changed the language of {m}'

    if expected_size is not None:
        for algo, r in results.items():
            assert len(r.nodes) == expected_size, (
                f'{algo}: expected {expected_size} states, got {len(r.nodes)} for {m}'
            )

    return results


# -------------------------- basic building blocks --------------------------


@pytest.mark.parametrize('algo', MIN_ALGOS)
def test_empty(algo):
    "Minimizing the empty language yields a machine with no final states."
    m = run_min(zero, algo)
    assert_is_dfa(m)
    assert zero.equal(m)
    # empty language: no accepting state reachable
    assert len(m.stop) == 0


@pytest.mark.parametrize('algo', MIN_ALGOS)
def test_one(algo):
    "Minimizing {ε} yields a single-state accepting machine."
    m = run_min(one, algo)
    assert_is_dfa(m)
    assert one.equal(m)
    assert len(m.nodes) == 1
    assert len(m.stop) == 1


@pytest.mark.parametrize('algo', MIN_ALGOS)
def test_single_symbol(algo):
    "A one-symbol machine is already minimal (2 states)."
    a = FSA.lift('a')
    m = run_min(a, algo)
    assert_is_dfa(m)
    assert a.equal(m)
    assert len(m.nodes) == 2


# -------------------------- consistency across algorithms --------------------------


def test_zero_consistency():
    _check_all(zero)


def test_one_consistency():
    _check_all(one, expected_size=1)


def test_single_symbol_consistency():
    a = FSA.lift('a')
    _check_all(a, expected_size=2)


def test_concat_consistency():
    a, b, c = map(FSA.lift, 'abc')
    _check_all(a * b * c, expected_size=4)


def test_union_consistency():
    a, b = map(FSA.lift, 'ab')
    _check_all(a + b, expected_size=2)


def test_star_consistency():
    a = FSA.lift('a')
    _check_all(a.star(), expected_size=1)


def test_plus_consistency():
    a = FSA.lift('a')
    # a+ = a a* has two distinct states (pre and post a)
    _check_all(a.p(), expected_size=2)


def test_alternation_star_consistency():
    a, b = map(FSA.lift, 'ab')
    # (a|b)* is a single accepting state that loops on a and b
    _check_all((a + b).star(), expected_size=1)


def test_fixed_length_three_symbols_consistency():
    "All strings of length 3 over {a,b} — the minimal DFA has 4 states."
    a, b = map(FSA.lift, 'ab')
    m = (a + b) * (a + b) * (a + b)
    _check_all(m, expected_size=4)


def test_union_of_length3_is_same_as_sigma3():
    "Taking the union of every length-3 string over {a,b} equals (a|b)^3."
    a, b = map(FSA.lift, 'ab')
    z = zero
    for x1 in [a, b]:
        for x2 in [a, b]:
            for x3 in [a, b]:
                z += x1 * x2 * x3
    _check_all(z, expected_size=4)


def test_redundant_union_consistency():
    "a|a|a must minimize down to the machine for 'a'."
    a = FSA.lift('a')
    m = a + a + a
    _check_all(m, expected_size=2)


def test_with_dead_states_consistency():
    "Extra unreachable + dead states must be trimmed away during minimization."
    a, b = map(FSA.lift, 'ab')
    m = a * b

    # attach an unreachable junk arc (island disconnected from start)
    m.add(99, 'a', 100)
    m.add(100, 'b', 101)

    # attach a dead branch reachable from start but unable to reach accept
    [s] = m.start
    m.add(s, 'c', 200)
    m.add(200, 'a', 201)   # 201 is not accepting

    _check_all(m, expected_size=3)   # a*b has 3 states


def test_eps_removed_by_minimization():
    "Epsilon transitions should be eliminated by minimization."
    a, b = map(FSA.lift, 'ab')
    m = (a + b).star() * a * b
    # this construction has epsilon arcs from the composition operators
    assert any(arc[1] == eps for arc in m.arcs())
    results = _check_all(m)
    for algo, r in results.items():
        for arc in r.arcs():
            assert arc[1] != eps, f'{algo} left epsilon arcs in minimized DFA'


def test_star_of_concat_consistency():
    "(ab)* — a small cyclic structure."
    a, b = map(FSA.lift, 'ab')
    _check_all((a * b).star(), expected_size=2)


def test_abc_then_anything_consistency():
    "abc followed by (a|b|c)* — a 4-state minimal DFA."
    a, b, c = map(FSA.lift, 'abc')
    m = a * b * c * (a + b + c).star()
    _check_all(m, expected_size=4)


def test_multiple_of_three_as_consistency():
    "Strings of a's whose length is a multiple of 3 — 3 states."
    a = FSA.lift('a')
    m = (a * a * a).star()
    _check_all(m, expected_size=3)


def test_contains_ab_consistency():
    "Strings over {a,b} containing 'ab' as a substring — 3 states."
    a, b = map(FSA.lift, 'ab')
    sigma = a + b
    m = sigma.star() * a * b * sigma.star()
    _check_all(m, expected_size=3)


def test_union_of_prefixes_consistency():
    "{a, ab, abc} — minimization merges nothing here beyond structure."
    a, b, c = map(FSA.lift, 'abc')
    m = a + a * b + a * b * c
    results = _check_all(m)
    # sanity: minimal must equal the constructed union
    for r in results.values():
        assert m.equal(r)


def test_from_strings_consistency():
    "Building from a small language via from_strings and minimizing."
    m = FSA.from_strings(['abc', 'abd', 'ac'])
    _check_all(m)


def test_union_from_strings_many():
    "A larger language built from a list of strings."
    langs = ['a', 'ab', 'abb', 'abbb', 'b', 'ba', 'baa']
    m = FSA.from_strings(langs)
    results = _check_all(m)
    # all the explicit strings must be accepted
    for r in results.values():
        for s in langs:
            assert s in r


def test_complement_consistency():
    "Complement of a DFA then minimized across algorithms."
    a, b = map(FSA.lift, 'ab')
    m = (a + b).star() * a
    comp = m.invert({'a', 'b'})
    _check_all(comp)


def test_intersection_consistency():
    "Intersection of two languages, minimized."
    a, b, c = map(FSA.lift, 'abc')
    m = (a.star() + b.star()) & (b + c)
    _check_all(m)


def test_symmetric_difference_consistency():
    "Symmetric difference: should behave identically across all three algorithms."
    a, b = map(FSA.lift, 'ab')
    x = (a + b).star() * a
    y = (a + b).star() * b
    _check_all(x ^ y)


# -------------------------- idempotence + involutions --------------------------


@pytest.mark.parametrize('algo', MIN_ALGOS)
def test_idempotent(algo):
    "Minimization is idempotent — the second pass must be isomorphic to the first."
    a, b = map(FSA.lift, 'ab')
    m = (a + b).star() * a * b
    once = run_min(m, algo)
    twice = run_min(once, algo)
    # Direct isomorphism check (not .equal() which would re-minimize).
    assert_iso(once, twice, msg=f'{algo} is not idempotent')


@pytest.mark.parametrize('algo', MIN_ALGOS)
def test_min_of_already_minimal(algo):
    "Running minimization on an already-minimal machine returns an isomorphic machine."
    a = FSA.lift('a')
    # a.star() is 1-state minimal
    m1 = run_min(a.star(), algo)
    m2 = run_min(m1, algo)
    assert_iso(m1, m2)


# -------------------------- cross-algorithm agreement sweep --------------------------


def _expressions():
    a, b, c = map(FSA.lift, 'abc')
    sigma = a + b + c
    return [
        zero,
        one,
        a,
        a + b,
        a * b,
        a.star(),
        (a + b).star(),
        (a * b).star(),
        a.p(),
        a * a * a,
        a.star() * b,
        sigma * sigma * sigma,
        sigma.star() * a * b * sigma.star(),    # contains 'ab'
        (a * a * a).star(),                      # multiples of 3 a's
        (a + b + c).star() * a * (a + b + c).star(),
        ((one + a) * a.star() * a.star() * a.star() * b),
        FSA.from_strings(['ab', 'ba', 'aa', 'bb']),
        FSA.from_strings(['a', 'aa', 'aaa']),
    ]


@pytest.mark.parametrize('idx', range(len(_expressions())))
def test_cross_algo_agreement(idx):
    "Every algorithm should produce identical minimal machines (same size, same language)."
    m = _expressions()[idx]
    _check_all(m)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
