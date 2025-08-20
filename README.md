# Evolutor

A simple package with very minimal functionality to handle the analytical evolution of beam parameters from the effect of Intra Beam Scattering.

## Installing

> [!WARNING]
> An `evolutor` package exists on PyPI, but it is not this one.

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

For default behavior and quick simulations, a command line interface is provided.
Simply passing the JSON file of your lattice, beam and formalism parameters will run the main evolution loop.

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

For finer control, using the Python API is recommended.
The package exposes a structure to hold results and compute beam parameters' evolution, but more elements need to be set up manually.
It does allow controlling every aspect of the evolution process, including additional customizations.

```python
from evolutor import Records

# set up parameters
nturns = 10_000
harmonic_number = 34640
# sigma_e = ...

# prepare records
results = Records(dt=line.twiss().T_rev0, nsteps=nturns)
# ...

# Run the loop yourself
# Now this loop handles everything
for step in range(1, nturns):
    # Potentially recompute growth rates
    if (step % recompute_step == 0) or (step == 1):
        rates = twiss.get_ibs_growth_rates(...)

    # Compute the new emittances etc and update
    results.update_with_ibs_at_next_step(...)
```

Detailed examples are provided for each mode in the `examples` directory.
They can be run isolated with `uv`:

```bash
uv run examples/seconds.py
```

## License

This package is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
