"""Render the README figures. Run from the repo root: `python docs/render.py`."""
from pathlib import Path

from fsa import FSA


OUT = Path(__file__).parent / 'img'
OUT.mkdir(exist_ok=True)


def save(name, fsa):
    g = fsa.graphviz()
    g.format = 'svg'
    g.render(str(OUT / name), cleanup=True)


a, b, c = map(FSA.lift, 'abc')

save('abc',      a * b * c)
save('abc_star', (a + b).star() * c)
save('abc_star_min', ((a + b).star() * c).min())

trie = FSA.from_strings(['cat', 'car', 'dog'])
save('trie',     trie)
save('trie_min', trie.min())
