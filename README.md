# Evolutor

A simple package with very minimal functionality to handle the analytical evolution of beam parameters from the effect of Intra Beam Scattering.

## Installing

The package being a prototype, it is only deployed on `test-pypi`.
It can be installed with `pip` by specifying the test index:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ evolutor
```

To install with `uv`:

```bash
uv pip install --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match evolutor
```
