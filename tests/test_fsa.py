import shutil
import pytest

from fsa import FSA, zero, one


@pytest.mark.skipif(shutil.which('dot') is None, reason='Graphviz `dot` binary not installed')
def test_visualization():
    a,b = map(FSA.lift, 'ab')
    ((a+b).star())._repr_mimebundle_()


def test_intersection():
    a,b,c = map(FSA.lift, 'abc')
    assert a.equal((a + a) & a)
    assert b.equal((a + b) & (b + c))
    assert b.equal((a.star() + b.star()) & (b + c))
    assert b.star().equal((a+b).star() & (b.star() + c.star()))
    assert ((a*b*c) & (a+b+c).star()).equal(a*b*c)


def test_complement():
    a,b,c = map(FSA.lift, 'abc')

    have = a - a
    want = zero
    assert have.equal(want)

    have = (a+b) - b
    want = a
    assert have.equal(want)

    have = (a+b) - (b+c)
    want = a
    assert have.equal(want)

    have = (a + b.star()) - (b+c)
    want = a + b * b * b.star() + one
    assert have.equal(want)

    # 1 b bb bbb bbbb bbbbb bbbbbb
    # 1   bb     bbbb       bbbbbb
    have = b.star() - (b*b).star()
    want = b * (b * b).star()
    assert have.equal(want)


def test_equality():
    a,b,c = map(FSA.lift, 'abc')

    assert a.equal(a)
    assert not a.equal(b)

    x = ((one + a) * a.star() * a.star() * a.star() * b)
    y = x.min()
    assert x.equal(y)
    assert y.equal(x)

    assert not (a * a.star()).equal(a.star())

    assert (one + a.star()).equal(a.star())
    assert (b * one + b * a.star()).equal(b * a.star())

    assert (a * a.star()).equal(a * a.star() + a * a * a.star())

    assert (a + b) != (b + a)
    assert (a + b + c).equal(b + a + c)

    assert ((a+b).star()).equal((a.star()*b.star()).star())

    assert (a+a).equal(a)   # idempotent

    x = a*a + a
    y = (a*a + a)*(a*a + a) + a

    assert not x.equal(y)


def test_min():
    a,b = map(FSA.lift, 'ab')
    z = zero
    for x1 in [a,b]:
        for x2 in [a,b]:
            for x3 in [a,b]:
                z += (x1 * x2 * x3).min()
    assert len(z.min().nodes) == 4
    print(z.min())

    assert len(((a + b)*(a + b)*(a + b)).min().nodes) == 4

    assert len(((one + a) * a.star() * a.star() * a.star() * b).min().nodes) == 2


def test_derivative():
    a, b, x, y = map(FSA.lift, 'abxy')

    # D_a({ax, by}) = {x}  (NOT {x, by} — the non-a-prefixed branch must be dropped)
    L = a*x + b*y
    assert L.D('a').equal(x)

    # D_a({a, ab}) = {ε, b}
    L = a + a*b
    assert L.D('a').equal(one + b)

    # D_a({ax}) = {x}
    L = a*x
    assert L.D('a').equal(x)

    # D over a symbol that doesn't start any word: empty derivative.
    L = a*x
    assert L.D('b').equal(zero)

    # D_a(a*) = a*  (once you consume one 'a', any number of 'a's still works)
    assert a.star().D('a').equal(a.star())

    # D_a(ε) = ∅
    assert one.D('a').equal(zero)


def test_contains():
    a = FSA.lift('a')

    # exact membership
    m = FSA.from_strings(['ab'])
    assert 'ab' in m
    assert 'abc' not in m     # regression: prefix 'ab' is accepted but 'abc' is not
    assert 'abcd' not in m
    assert 'a' not in m       # proper prefix of an accepted string
    assert '' not in m

    # language with multiple strings
    m = FSA.from_strings(['ab', 'abc'])
    assert 'ab' in m
    assert 'abc' in m
    assert 'abcd' not in m    # extends past an accepted string
    assert 'abx' not in m     # diverges after an accepted prefix

    # empty string
    assert '' in one
    assert '' not in a

    # Kleene star accepts ε and repetitions
    assert '' in a.star()
    assert 'aaa' in a.star()
    assert 'aab' not in a.star()


#def test_fsa_to_regex():
#    from semirings.regex import Symbol
#    a, b = map(FSA.lift, 'ab')
#    m = b * a.star()
#    m = m.min()
#
#    A, B = map(Symbol, 'ab')
#    assert m.to_regex() == B + B * A.star() * A


def test_quotient():
    a, b, c, d = map(FSA.lift, 'abcd')

    q = (a * b) // a
    assert q.equal(b)

    q = (a * b * c + c * d) // a
    assert q.equal(b * c + zero * c * d)

    q = (a * b * c * d + c * a * b * d) // (a * b)
    assert q.equal(c * d)

    q = (a * b) / b
    assert q.equal(a)

    q = ((a * b) / b) / a
    assert q.equal(one)

    q = (a * b) / (a * b * c)
    assert q.equal(zero)

    q = (a.star() * b) // a
    assert q.equal(a.star() * b)

    q = (a.star() * b) // a.star()
    print(q)
    print(q.min().renumber())
    assert q.equal(a.star() * b)   # is this correct?


    # L1\L2 = {y | ∃x ∈ L2: xy ∈ L1}
    # L1/L2 = {x | ∃y ∈ L2: xy ∈ L1}

    L1 = (a.star() * b)
    L2 = a.star()

    def checker(y): return (L2 * y) <= L1

    assert checker(a * b)
    assert checker(a.star() * b)
    assert not checker(c)


def test_arcs_filtering():
    "arcs() has three call modes: no-arg, source-only, and (source, label)."
    m = FSA()
    m.add(0, 'a', 1).add(0, 'b', 2).add(1, 'a', 1).add(1, 'c', 2)

    # no-arg: all triples
    assert set(m.arcs()) == {(0, 'a', 1), (0, 'b', 2), (1, 'a', 1), (1, 'c', 2)}

    # source-only: (label, target) pairs from state 0
    assert set(m.arcs(0)) == {('a', 1), ('b', 2)}

    # (source, label): iter of targets
    assert set(m.arcs(1, 'a')) == {1}
    assert set(m.arcs(1, 'c')) == {2}
    # no arcs from (0, 'c')
    assert set(m.arcs(0, 'c')) == set()


def test_strict_subset():
    a, b = map(FSA.lift, 'ab')

    # a ⊂ (a | b): strict subset
    assert a < (a + b)
    # (a | b) is not a strict subset of itself
    assert not ((a + b) < (a + b))
    # b is not a subset of a
    assert not (b < a)


def test_from_string():
    "Single-string classmethod (distinct from from_strings which takes a list)."
    m = FSA.from_string('hello')
    assert 'hello' in m
    assert 'hell' not in m
    assert 'helloo' not in m
    assert '' not in m
    assert m.equal(FSA.from_strings(['hello']))


def test_merge():
    "merge() folds a set of states into a single representative."
    a, b = map(FSA.lift, 'ab')
    # Build (a|b) — two parallel branches sharing start and stop
    m = a + b

    # Pick two distinct non-start states and merge them. The language shouldn't
    # shrink; in general it may grow (merges only add edges).
    before = set(m.arcs())
    S = set(list(m.stop)[:2]) if len(m.stop) >= 2 else set(list(m.nodes)[:2])
    merged = m.merge(S)
    # language preservation: merged language contains original
    assert m <= merged
    # fewer states (or same) post-merge
    assert len(merged.nodes) <= len(m.nodes)
    # original unchanged
    assert set(m.arcs()) == before


def test_isomorphism_conflict():
    "Two DFAs that have the same state count but are not isomorphic."
    a, b = map(FSA.lift, 'ab')
    sigma = a + b
    # Both "ends in aa" and "ends in ab" minimize to 3-state DFAs, but they're
    # non-isomorphic — walking in lockstep from the starts will try to bind the
    # same self-state to two different other-states.
    L_aa = sigma.star() * a * a
    L_ab = sigma.star() * a * b
    assert not L_aa.equal(L_ab)
    assert not L_ab.equal(L_aa)


try:
    import semirings  # noqa: F401
    HAS_SEMIRINGS = True
except ImportError:
    HAS_SEMIRINGS = False


@pytest.mark.skipif(not HAS_SEMIRINGS, reason='optional semirings package not installed')
def test_to_regex():
    from semirings.regex import Symbol
    a = FSA.lift('a')
    # L(a) = {a}, regex is just 'a'
    r = a.to_regex()
    A = Symbol('a')
    assert r == A

    # Kleene star: L(a.star()) = a*
    r = a.star().min().to_regex()
    # Exact form depends on elimination ordering, but must equal A.star()
    assert r == A.star()


@pytest.mark.skipif(not HAS_SEMIRINGS, reason='optional semirings package not installed')
def test_to_regex_with_eps_arcs():
    "to_regex must accept machines that still contain ε-transitions."
    from fsa.fsa import eps
    a, b = map(FSA.lift, 'ab')
    m = a * b  # concatenation injects eps arcs between the two pieces
    assert any(arc[1] == eps for arc in m.arcs())
    # Shouldn't raise; the resulting regex should at minimum recognize 'ab'.
    m.to_regex()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
