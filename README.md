# Evolutor

A simple package with very minimal functionality to handle the analytical evolution of beam parameters from the effect of Intra Beam Scattering, and potentially also Synchrotron Radiation.

## Installing

> [!WARNING]
> An `evolutor` package exists on PyPI, but it is not this one.

The package being a prototype, it is only deployed on `test-pypi`.
It can be installed with `pip` by specifying the test index:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ evolutor
```

To install with `uv pip`:

```bash
uv pip install --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match evolutor
```

## Usage

### Command Line

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

Detailed usage of the CLI goes as:

```bash
Usage: python -m evolutor [OPTIONS] SEQUENCE [MODE]

Command line tool to run the IBS evolutor.

Provided with a sequence file and required parameters, this tool runs the IBS evolutor simulation either per
seconds or per turns, depending on the mode specified. The results can be exported to a .npz file if requested.

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    sequence      FILE    Path to the sequence file. [required]                                                  │
│      mode          [MODE]  Simulation mode, either per 'seconds' or 'turns'. [default: seconds]                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                           │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.    │
│ --help                        Show this message and exit.                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ IBS growth rates computing ──────────────────────────────────────────────────────────────────────────────────────╮
│ *  --formalism                          [b&m|bjorken-mtingwa|nagaitsev]  IBS formalism to use for growth rates.   │
│                                                                          [default: None]                          │
│                                                                          [required]                               │
│ *  --rf-voltage                         FLOAT RANGE [x>=0.0]             RF voltage in [V]. [default: None]       │
│                                                                          [required]                               │
│ *  --harmonic-number                    INTEGER RANGE [x>=1]             Harmonic number of the ring.             │
│                                                                          [default: None]                          │
│                                                                          [required]                               │
│ *  --bunch-intensity                    FLOAT RANGE [x>=0.0]             Bunch intensity in [ppb] (particles per  │
│                                                                          bunch).                                  │
│                                                                          [default: None]                          │
│                                                                          [required]                               │
│ *  --nemitt-x                           FLOAT RANGE [x>=0.0]             Normalized emittance in the horizontal   │
│                                                                          plane in [m].                            │
│                                                                          [default: None]                          │
│                                                                          [required]                               │
│ *  --nemitt-y                           FLOAT RANGE [x>=0.0]             Normalized emittance in the vertical     │
│                                                                          plane in [m].                            │
│                                                                          [default: None]                          │
│                                                                          [required]                               │
│ *  --sigma-z                            FLOAT RANGE [x>=0.0]             Bunch length in [m]. [default: None]     │
│                                                                          [required]                               │
│    --bunched            --no-bunched                                     Whether the beam is bunched or not. If   │
│                                                                          False, the IBS growth rates are computed │
│                                                                          for a coasting beam.                     │
│                                                                          [default: bunched]                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Global simulation parameters ────────────────────────────────────────────────────────────────────────────────────╮
│    --nseconds              INTEGER RANGE [x>=0]  Number of seconds to simulate. If mode is not 'seconds', this is │
│                                                  rejected.                                                        │
│                                                  [default: None]                                                  │
│    --nturns                INTEGER RANGE [x>=1]  Number of turns to simulate. If mode is not 'turns', this is     │
│                                                  rejected.                                                        │
│                                                  [default: None]                                                  │
│    --dt                    INTEGER RANGE [x>=0]  The time step in [s] between two data points. If mode is         │
│                                                  'seconds', this defaults to 1s. If mode is 'turns', this         │
│                                                  defaults to the revolution time.                                 │
│                                                  [default: None]                                                  │
│ *  --recompute-step        INTEGER RANGE [x>=1]  Re-compute the IBS growth rates every this many seconds or       │
│                                                  turns.                                                           │
│                                                  [default: None]                                                  │
│                                                  [required]                                                       │
│    --export                FILE                  If provided, export the results to a .npz file with the given    │
│                                                  name.                                                            │
│                                                  [default: None]                                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Python API

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

## Examples

Some detailed examples are provided for each mode in the [examples](./examples) directory.
The individual scripts follow `PEP 723` and can be run isolated with `uv`:

```bash
uv run examples/seconds.py
```

## License

This package is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
