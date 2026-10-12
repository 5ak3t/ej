# Contributing to ej

Issues and pull requests are welcome. Issues labelled `good first issue` are a good place to start.

## Set up

```bash
git clone https://github.com/5ak3t/ej && cd ej
pip install -e '.[test]'
```

## Before you open a pull request

```bash
python scripts/check_file_length.py                          # every file <= 300 lines
python -m pytest -q -rs                                       # what CI runs
EJ_WEIGHTS=/path/to/model.ejpack python -m pytest -q -rs      # optional: the prediction tests
```

Every change follows the rules in [AGENTS.md](AGENTS.md): tests for new behaviour, no data or weights in git, every
number backed by the command that produced it, `ej/_runtime/` left unchanged, and code copied only from permissively
licensed sources.

## Reporting a problem

Open an issue with the ej version (`pip show ejai`), your Python and torch versions, a minimal record that shows the
problem, and what you expected.

## Licence

By contributing you agree that your code is licensed under GPL-3.0-only ([LICENSE](LICENSE)).
