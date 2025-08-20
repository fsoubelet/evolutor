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

## Usage

```bash
python -m evolutor examples/lhcb1.json seconds \
    --formalism nagaitsev \
    --rf-voltage 4000000.0 \
    --harmonic-number 34640 \
    --bunch-intensity 192000000000.0 \
    --nemitt-x 2.2e-06 \
    --nemitt-y 2.2e-06 \
    --sigma-z 0.08993773646299315 \
    --nseconds 7200 \
    --recompute-step 50 \
    --export data_nag_secs.npz
```

## License

This package is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
