from setuptools import setup

setup(
    name='fsa',
    version='0.1',
    description=(
        "Finite-state automata"
    ),
    project_url = 'https://github.com/timvieira/fsa',
    install_requires = [
        'numpy',
        'pytest',
        'graphviz',   # for notebook visualization of left-recursion graph
        'arsenal @ git+https://github.com/timvieira/arsenal',
    ],
    authors = [
        'Tim Vieira',
    ],
    readme=open('README.md').read(),
    scripts=[],
    packages=['fsa'],
)
