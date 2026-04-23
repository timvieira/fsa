"""Render a worked minimization example (Brzozowski & Hopcroft).

Run from the repo root: `python docs/render_minimization_example.py`.
"""
from pathlib import Path

from fsa import FSA


OUT = Path(__file__).parent / 'img'
OUT.mkdir(exist_ok=True)


def save(name, fsa, fmt='png'):
    g = fsa.graphviz()
    g.format = fmt
    g.render(str(OUT / name), cleanup=True)
    return OUT / f'{name}.{fmt}'


# Input machine, transcribed from the picture.
# States: 0, 1, 2, 3.   Start: 0.   Accept: 0, 1 (double circles).
#
# Transitions (source -[symbols]-> target):
#   0 -[2,5]->    2
#   2 -[0,2]->    2     (self-loop)
#   2 -[0,4]->    1
#   2 -[3,4]->    0
#   2 -[1,5]->    3
#   3 -[2,5]->    2
#   3 -[1]->      1
#   3 -[1,4]->    0
#   3 -[0,4]->    0
#   1 -[1,2,3,5]-> 3
#
# Note: a few labels overlap on the same source (e.g. 2 on input 4 reaches
# both 1 and 0; 3 on input 1 reaches both 1 and 0), so this is an NFA.
# Both minimization algorithms first determinize, then collapse equivalent
# states, so this is a fair test of the full pipeline.
EDGES = [
    (0, [2, 5],        2),
    (2, [0, 2],        2),
    (2, [0, 4],        1),
    (2, [3, 4],        0),
    (2, [1, 5],        3),
    (3, [2, 5],        2),
    (3, [1],           1),
    (3, [1, 4],        0),
    (3, [0, 4],        0),
    (1, [1, 2, 3, 5],  3),
]

m = FSA()
m.add_start(0)
m.add_stop(0)
m.add_stop(1)
for src, syms, dst in EDGES:
    for a in syms:
        m.add(src, a, dst)


# Original as drawn
save('min_example_input', m)

# Intermediate: determinized (subset construction). Brzozowski and Hopcroft
# both start from the DFA; showing it helps explain what they operate on.
save('min_example_det', m.det().renumber())

# Brzozowski: reverse, determinize, reverse, determinize, trim.
save('min_example_brzozowski', m.min_brzozowski().renumber())

# Hopcroft (partition refinement). The library ships two variants; `min_fast`
# is the classical formulation, `min_faster` adds an index. Both produce the
# same minimal DFA (up to state relabeling).
save('min_example_hopcroft', m.min_fast().renumber())


# Sanity: both should accept the same language as the input, and be
# isomorphic to each other.
assert m.min_brzozowski().equal(m.min_fast())
assert m.equal(m.min_brzozowski())
assert m.equal(m.min_fast())
print('OK — minimizations agree and preserve the language.')
print(f'  input states:      {len(m.nodes)}')
print(f'  det states:        {len(m.det().nodes)}')
print(f'  brzozowski states: {len(m.min_brzozowski().nodes)}')
print(f'  hopcroft states:   {len(m.min_fast().nodes)}')
