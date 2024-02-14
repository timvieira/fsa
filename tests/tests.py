from fsa import FSA, zero, one


def test_visualization():
    a,b = map(FSA.lift, 'ab')
    ((a+b).star())._repr_html_()


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


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
