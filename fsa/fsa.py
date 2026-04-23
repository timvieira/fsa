from arsenal import Integerizer
from collections import defaultdict
from functools import lru_cache
from graphviz import Digraph


def dfs(Ps, arcs):
    "Subgraph reachable from seeds `Ps` under the transition callable `arcs(P) -> iter[(label, Q)]`."
    stack = list(Ps)
    m = FSA()
    for P in Ps: m.add_start(P)
    while stack:
        P = stack.pop()
        for a, Q in arcs(P):
            if Q not in m.nodes:
                stack.append(Q)
                m.nodes.add(Q)
            m.add(P, a, Q)
    return m


_frozenset = frozenset
class frozenset(_frozenset):
    "frozenset subclass with a stable, sorted repr."
    def __repr__(self):
        return '{%s}' % (','.join(str(x) for x in sorted_robust(self)))



def sorted_robust(xs):
    "sorted() that tolerates heterogeneous types by grouping by type name first."
    return sorted(xs, key=lambda x: (type(x).__name__, x))


class FSA:

    def __init__(self):
        self.start = set()
        self.edges = defaultdict(lambda: defaultdict(set))
        self.nodes = set()
        self.stop = set()
        self.syms = set()

    def as_tuple(self):
        "Canonical hashable representation; backs __hash__ and __eq__."
        return (frozenset(self.nodes),
                frozenset(self.start),
                frozenset(self.stop),
                frozenset(self.arcs()))

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()

    def __repr__(self):
        try:
            return repr(self.to_regex())
        except ImportError:
            # to_regex needs the optional `semirings` extra.
            return f'<{type(self).__name__} states={len(self.nodes)} starts={len(self.start)} stops={len(self.stop)} arcs={sum(1 for _ in self.arcs())}>'

    def __str__(self):
        x = ['{']
        for s in sorted_robust(self.nodes):
            ss = f'{s}'
            if s in self.start:
                ss = f'^{ss}'
            if s in self.stop:
                ss = f'{ss}$'
            x.append(f'  {ss}:')
            for a, t in sorted_robust(self.arcs(s)):
                x.append(f'    {a} -> {t}')
        x.append('}')
        return '\n'.join(x)

    def _repr_mimebundle_(self, *args, **kwargs):
        "Jupyter rich-display hook; delegates to Graphviz."
        return self.graphviz()._repr_mimebundle_(*args, **kwargs)

    def graphviz(self, show_label=True):
        "Graphviz rendering: starts drawn as incoming arrows, accept states as double circles."
        import html
        g = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='10',
                height='.05', width='.05',
                margin="0,0"
            ),
            edge_attr=dict(
                arrowsize='0.3',
                fontname='Monospace',
                fontsize='9'
            ),
        )
        f = Integerizer()

        # '<start>' sentinel can't collide: every real node is labeled by
        # Integerizer, which always yields a decimal-digit string.
        start = '<start>'
        g.node(start, label='', shape='point', height='0', width='0')
        for i in self.start:
            g.edge(start, str(f(i)), label='')

        for i in sorted_robust(self.nodes):
            shape = 'circle'
            if i in self.stop: shape = 'doublecircle'
            label = str(i) if show_label else ''
            g.node(str(f(i)), label=label, shape=shape)
            for a, j in sorted_robust(self.arcs(i)):
                g.edge(str(f(i)), str(f(j)), label=html.escape(str(a).replace(' ', '␣')))

        return g

    def D(self, x):
        "Left quotient (Brzozowski derivative) by symbol `x`: { y : x·y ∈ L(self) }."
        e = self.epsremoval()
        m = FSA()

        for i, a, j in e.arcs():
            m.add(i, a, j)
        for j in e.stop:
            m.add_stop(j)
        # New starts = x-successors of the old starts. The old starts are no
        # longer start states, so their non-x outgoing arcs are no longer
        # reachable (hence do not contaminate the derivative).
        for i in e.start:
            for j in e.edges[i][x]:
                m.add_start(j)

        return m

    def add(self, i, a, j):
        self.edges[i][a].add(j)
        self.nodes.add(i); self.syms.add(a); self.nodes.add(j)
        return self

    def add_start(self, i):
        self.start.add(i)
        self.nodes.add(i)
        return self

    def add_stop(self, i):
        self.stop.add(i)
        self.nodes.add(i)
        return self

    def arcs(self, i=None, a=None):
        "Iterate the transition relation, optionally restricted by source state `i` and/or label `a`."
        if i is None and a is None:

            for i in self.edges:
                for a in self.edges[i]:
                    for j in self.edges[i][a]:
                        yield (i,a,j)

        elif i is not None and a is None:

            for a in self.edges[i]:
                for j in self.edges[i][a]:
                    yield (a,j)

        elif i is not None and a is not None:

            for j in self.edges[i][a]:
                yield j

        else:
            raise NotImplementedError()

    def reverse(self):
        "Machine for the reversed language: L(self.reverse()) = { w[::-1] : w ∈ L(self) }."
        m = FSA()
        for i in self.start:
            m.add_stop(i)
        for i in self.stop:
            m.add_start(i)
        for i, a, j in self.arcs():
            m.add(j, a, i)     # pylint: disable=W1114
        return m

    def _accessible(self, start):
        return dfs(start, self.arcs).nodes

    def accessible(self):
        "States that lie on at least one path from a start state."
        return self._accessible(self.start)

    @lru_cache(None)
    def trim(self):
        "Equivalent machine with useless (unreachable or dead-end) states removed."
        c = self.accessible() & self.reverse().accessible()
        m = FSA()
        for i in self.start & c:
            m.add_start(i)
        for i in self.stop & c:
            m.add_stop(i)
        for i,a,j in self.arcs():
            if i in c and j in c:
                m.add(i,a,j)
        return m

    def renumber(self):
        "Canonicalize state labels to consecutive integers."
        return self.rename(Integerizer())

    def rename(self, f):
        "Equivalent machine with every state label `i` replaced by `f(i)`; non-injective `f` merges states."
        m = FSA()
        for i in self.start:
            m.add_start(f(i))
        for i in self.stop:
            m.add_stop(f(i))
        for i, a, j in self.arcs():
            m.add(f(i), a, f(j))
        return m

    def rename_apart(self, other):
        "Relabel `self` and `other` so their state sets are disjoint."
        f = Integerizer()
        self = self.rename(lambda i: f((0, i)))
        other = other.rename(lambda i: f((1, i)))
        assert self.nodes.isdisjoint(other.nodes)
        return (self, other)

    def __mul__(self, other):
        "Concatenation: L(self) · L(other)."
        m = FSA()
        self, other = self.rename_apart(other)
        m.start = self.start
        m.stop = other.stop
        for i in self.stop:
            for j in other.start:
                m.add(i,eps,j)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def __add__(self, other):
        "Union: L(self) ∪ L(other)."
        m = FSA()
        [self, other] = self.rename_apart(other)
        m.start = self.start | other.start
        m.stop = self.stop | other.stop
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i,a,j in other.arcs():
            m.add(i,a,j)
        return m

    def p(self):
        "Kleene plus: L(self)+ (one or more repetitions)."
        m = FSA()
        m.start = set(self.start)
        m.stop = set(self.stop)
        for i,a,j in self.arcs():
            m.add(i,a,j)
        for i in self.stop:
            m.add_stop(i)
            for j in self.start:
                m.add(i, eps, j)
        return m

    def star(self):
        "Kleene star: L(self)* (zero or more repetitions)."
        return one + self.p()

    @lru_cache(None)
    def epsremoval(self):
        "Equivalent machine with all ε-transitions eliminated."
        eps_m = FSA()
        for i,a,j in self.arcs():
            if a == eps:
                eps_m.add(i,a,j)

        @lru_cache
        def eps_accessible(i):
            return eps_m._accessible({i})

        m = FSA()

        for i,a,j in self.arcs():
            if a == eps: continue
            m.add(i, a, j)
            for k in eps_accessible(j):
                m.add(i, a, k)

        for i in self.start:
            m.add_start(i)
            for k in eps_accessible(i):
                m.add_start(k)

        for i in self.stop:
            m.add_stop(i)

        return m

    @lru_cache(None)
    def det(self):
        "Equivalent DFA via the subset (powerset) construction; states are frozensets of the original NFA states."
        self = self.epsremoval()

        def powerarcs(Q):
            for a in self.syms:
                yield a, frozenset({j for i in Q for j in self.edges[i][a]})

        m = dfs([frozenset(self.start)], powerarcs)

        for powerstate in m.nodes:
            if powerstate & self.stop:
                m.add_stop(powerstate)

        return m

    def min_brzozowski(self):
        "Brzozowski's minimization algorithm"
        # https://en.wikipedia.org/wiki/DFA_minimization#Brzozowski's_algorithm

        # Proof of correctness:
        #
        # Let M' = M.r.d.r
        # Clearly,  [[M']] = [[M]]
        #
        # In M', there are no two states that can accept the same suffix
        # language because the reverse of M' is deterministic.
        #
        # The determinization of M' then creates powerstates, where every pair
        # of distinct powerstates R and S, there exists by construction at least
        # one state q of M' where q \in R and q \notin S. Such a q contributes
        # at least one word w \in [[q]] to the suffix language of q in [[R]]
        # that is not present in [[S]], since this word is unique to q (i.e., no
        # other state accepts it).  Thus, all pairs of states in M'.d are
        # distinguishable.
        #
        # Thus, after trimming of M'.d, we have a DFA with no indistinguishable
        # or unreachable states, which must be minimal.

        return self.reverse().det().reverse().det().trim()

    def min_fast(self):
        "Minimal equivalent DFA via Hopcroft-style partition refinement (rescans every block each step; see `min_faster` for the indexed version)."
        self = self.det().renumber()

        # calculate inverse of transition function (i.e., reverse arcs)
        inv = defaultdict(set)
        for i,a,j in self.arcs():
            inv[j,a].add(i)

        final = self.stop
        nonfinal = self.nodes - final

        P = [final, nonfinal]
        # Hopcroft: only the smaller side needs to seed W — splits witnessed
        # by the larger side are redundant.
        W = [final if len(final) <= len(nonfinal) else nonfinal]

        while W:
            A = W.pop()
            for a in self.syms:
                X = {i for j in A for i in inv[j,a]}
                R = []
                for Y in P:
                    if X.isdisjoint(Y) or X >= Y:
                        R.append(Y)
                    else:
                        YX = Y & X
                        Y_X = Y - X
                        R.append(YX)
                        R.append(Y_X)
                        W.append(YX if len(YX) < len(Y_X) else Y_X)
                P = R

        # create new equivalence classes of states
        minstates = {}
        for i, qs in enumerate(P):
            for q in qs:
                minstates[q] = i

        return self.rename(lambda i: minstates[i]).trim()

    def min_faster(self):
        "Minimal equivalent DFA via Hopcroft's algorithm, with a block-index (`find`) so only affected partitions are revisited."
        self = self.det().renumber()

        # calculate inverse of transition function (i.e., reverse arcs)
        inv = defaultdict(set)
        for i,a,j in self.arcs():
            inv[j,a].add(i)

        final = self.stop
        nonfinal = self.nodes - final

        P = [final, nonfinal]
        # Hopcroft: only the smaller side needs to seed W — splits witnessed
        # by the larger side are redundant.
        W = [final if len(final) <= len(nonfinal) else nonfinal]

        find = {i: block for block, elements in enumerate(P) for i in elements}

        while W:

            S = W.pop()
            for a in self.syms:

                # Group pre-images by their current block; this lets us
                # replace the O(|Y|) superset check `X >= Y` with an
                # O(1) length comparison.
                block_members = defaultdict(set)
                for j in S:
                    for i in inv[j, a]:
                        block_members[find[i]].add(i)

                for block, YX in block_members.items():
                    Y = P[block]

                    if len(YX) == len(Y): continue

                    Y_X = Y - YX

                    # we will replace block with the intersection case (no
                    # need to update `find` index for YX elements)
                    P[block] = YX

                    new_block = len(P)
                    for i in Y_X:
                        find[i] = new_block

                    P.append(Y_X)
                    W.append(YX if len(YX) < len(Y_X) else Y_X)

        return self.rename(lambda i: find[i]).trim()

    min = lru_cache(None)(min_faster)

    def equal(self, other):
        "Language equality: L(self) = L(other)."
        return self.min()._dfa_isomorphism(other.min())

    def _dfa_isomorphism(self, other):
        "True iff `self` and `other` are isomorphic as DFAs. Assumes both are minimal."

        # Minimal DFAs have at most one start state; a DFA's per-symbol
        # out-degree is at most one. Walk both from their starts in lockstep,
        # building a candidate bijection on states.
        if len(self.nodes) != len(other.nodes): return False
        if len(self.start) == 0: return len(other.start) == 0

        assert len(self.start) == 1 and len(other.start) == 1

        [p] = self.start
        [q] = other.start

        stack = [(p, q)]
        iso = {p: q}

        syms = self.syms | other.syms

        done = set()
        while stack:
            (p, q) = stack.pop()
            done.add((p,q))
            for a in syms:

                # presence of the arc must match on both sides
                if (a in self.edges[p]) != (a in other.edges[q]):
                    return False

                if a not in self.edges[p]:
                    continue

                # machines are assumed deterministic
                [r] = self.edges[p][a]
                [s] = other.edges[q][a]

                if r in iso and iso[r] != s:
                    return False

                iso[r] = s
                if (r,s) not in done:
                    stack.append((r,s))

        return self.rename(iso.get) == other

    def to_regex(self):
        "Equivalent regular expression."
        import numpy as np
        from semirings.regex import Symbol
        from semirings.kleene import kleene

        n = len(self.nodes)

        A = np.full((n,n), Symbol.zero)
        start = np.full(n, Symbol.zero)
        stop = np.full(n, Symbol.zero)

        ix = Integerizer(list(self.nodes))

        for i in self.nodes:
            for a, j in self.arcs(i):
                if a == eps:
                    A[ix(i),ix(j)] += Symbol.one
                else:
                    A[ix(i),ix(j)] += Symbol(a)

        for i in self.start:
            start[ix(i)] += Symbol.one

        for i in self.stop:
            stop[ix(i)] += Symbol.one

        return start @ kleene(A, Symbol) @ stop

    def __and__(self, other):
        "Intersection: L(self) ∩ L(other). States are pairs (q1, q2) from the product construction."

        self = self.epsremoval().renumber()
        other = other.epsremoval().renumber()

        def product_arcs(Q):
            (q1, q2) = Q
            for a, j1 in self.arcs(q1):
                for j2 in other.edges[q2][a]:
                    yield a, (j1,j2)

        m = dfs({(q1, q2) for q1 in self.start for q2 in other.start},
                product_arcs)

        # final states
        for q1 in self.stop:
            for q2 in other.stop:
                m.add_stop((q1, q2))

        return m

    def add_sink(self, syms):
        "Equivalent machine made total over alphabet `syms` by routing missing transitions to a fresh non-accepting sink state."

        syms = set(syms) - {eps}   # the alphabet of a total DFA cannot contain ε

        self = self.renumber()

        sink = len(self.nodes)
        for a in syms:
            self.add(sink, a, sink)

        for q in self.nodes:
            if q == sink: continue
            for a in syms - set(self.edges[q]):
                self.add(q, a, sink)

        return self

    def __sub__(self, other):
        "Language difference: L(self) \\ L(other)."
        return self & other.invert(self.syms | other.syms)

    __or__ = __add__

    def __xor__(self, other):
        "Symmetric difference: (L(self) ∪ L(other)) \\ (L(self) ∩ L(other))."
        return (self | other) - (self & other)

    def invert(self, syms):
        "Complement of the language with respect to alphabet `syms`."

        self = self.det().add_sink(syms)

        m = FSA()

        for i in self.nodes:
            for a, j in self.arcs(i):
                m.add(i, a, j)

        for q in self.start:
            m.add_start(q)

        for q in self.nodes - self.stop:
            m.add_stop(q)

        return m

    def __floordiv__(self, other):
        "left quotient self//other ≐ {y | ∃x ∈ other: x⋅y ∈ self}"

        self = self.epsremoval()
        other = other.epsremoval()

        # quotient arcs are very similar to product arcs except that the common
        # string is "erased" in the new machine.
        def quotient_arcs(Q):
            (q1, q2) = Q
            for a, j1 in self.arcs(q1):
                for j2 in other.edges[q2][a]:
                    yield eps, (j1, j2)

        m = dfs({(q1, q2) for q1 in self.start for q2 in other.start},
                quotient_arcs)

        # If we have managed to reach a final state of q2 then we can move into
        # the post-prefix set of states
        for (q1,q2) in set(m.nodes):
            if q2 in other.stop:
                m.add((q1, q2), eps, (q1,))

        # business as usual
        for q1 in self.nodes:
            for a, j1 in self.arcs(q1):
                m.add((q1,), a, (j1,))
        for q1 in self.stop:
            m.add_stop((q1,))

        return m

    def __truediv__(self, other):
        "right quotient self/other ≐ {x | ∃y ∈ other: x⋅y ∈ self}"
        return (self.reverse() // other.reverse()).reverse()   # reduce to left quotient on reversed languages

    def __lt__(self, other):
        "self ⊂ other"
        if self.equal(other): return False
        return (self & other).equal(self)

    def __le__(self, other):
        "self ⊆ other"
        return (self & other).equal(self)

    @classmethod
    def lift(cls, x):
        "Single-symbol machine: accepts exactly the one-symbol string `x`."
        m = cls()
        m.add_start(0); m.add_stop(1); m.add(0,x,1)
        return m

    @classmethod
    def from_string(cls, xs):
        "Linear machine that accepts exactly the string `xs`."
        m = cls()
        m.add_start(xs[:0])
        for i in range(len(xs)):
            m.add(xs[:i], xs[i], xs[:i+1])
        m.add_stop(xs)
        return m

    @classmethod
    def from_strings(cls, Xs):
        "Prefix-tree machine that accepts exactly the strings in `Xs`."
        m = cls()
        for xs in Xs:
            m.add_start(xs[:0])
            for i in range(len(xs)):
                m.add(xs[:i], xs[i], xs[:i+1])
            m.add_stop(xs)
        return m

    def __contains__(self, xs):
        "Membership test: True iff `xs` is accepted."
        d = self.det()
        [s] = d.start
        for x in xs:
            t = d.edges[s][x]
            if not t: return False
            [s] = t
        return (s in d.stop)

    def merge(self, S, name=None):
        "merge states in `S` into a single state."
        if name is None: name = min(S)
        def f(s):
            return name if s in S else s
        m = FSA()
        for x in self.start:
            m.add_start(f(x))
        for x,a,y in self.arcs():
            m.add(f(x),a,f(y))
        for x in self.stop:
            m.add_stop(f(x))
        return m

eps = 'ε'

FSA.one = one = FSA()
one.add_start(0); one.add_stop(0)

FSA.zero = zero = FSA()
