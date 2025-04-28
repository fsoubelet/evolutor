# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "setuptools",
#   "xtrack>=0.80.0",
#   "xfields>=0.23.0",
#   "xpart>=0.22.0",
#   "evolutor",
# ]
# [tool.uv.sources]
# evolutor = { index = "testpypi" }
# [[tool.uv.index]]
# name = "testpypi"
# url = "https://test.pypi.org/simple/"
# ///

"""
A quick example on how to call the records class and run a
simple evolution, including the effect of Synchrotron Radiation,
then visualize it. This script uses seconds as steps.
"""

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from evolutor import Records, energy_spread

# Get a line and Twiss with radiation
line = xt.Line.from_json("chrom-corr_DR.newlattice_2GHz.json")
line.build_tracker()
line.configure_radiation(model='mean')
twiss = line.twiss(eneloss_and_damping=True)
formalism = "nagaitsev"  # faster (use if there is no Dy)

# Parameters computed once and needed later
circumference = line.get_length()
total_energy = np.sqrt(line.particle_ref.p0c[0] ** 2 + line.particle_ref.mass0**2)  # in [eV]
slip_factor = twiss.slip_factor
beta_rel = twiss.beta0
reference_charge = line.particle_ref.q0
sr_eq_sigma_delta = (twiss.eq_gemitt_zeta / twiss.bets0) ** 0.5

# Parameters to be set by the user and needed later
harmonic_number = 2851  # provide your own
RF_voltage = 4.5e6  # in [V] - provide your own
bunch_intensity = 4.07e9  # provide your own
nemitt_x = 5.6644e-07  # in [m] - provide your own
nemitt_y = 3.7033e-09  # in [m] - provide your own
sigma_z = 1.58e-3  # in [m] - provide your own
sigma_e = energy_spread(
    circumference,
    harmonic_number=harmonic_number,
    total_energy=total_energy,
    slip_factor=slip_factor,
    bunch_length=sigma_z,
    beta_rel=beta_rel,
    rf_voltage=RF_voltage,
    reference_charge=reference_charge,
)
sigma_delta = sigma_e / (beta_rel**2)

# SET THESE TO DETERMINE THE SIMULATION - do not make recompute_step too high
nturns = 1000  # number of turns to simulate
recompute_step = 1  # re-compute growth rates every [turns]

# Prepare records
dt = twiss.T_rev0  # time step in [s] (revolution time)
turns = np.arange(nturns, dtype=int)
results = Records(nsteps=nturns)
# Insert the initial values
results.update_at_step(
    0,
    epsx=nemitt_x,
    epsy=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=sigma_z,
)


# Now this is an ugly for loop
for turn in range(1, nturns):
    # Potentially recompute growth rates
    if (turn % recompute_step == 0) or (turn == 1):
        rates = twiss.get_ibs_growth_rates(
            formalism=formalism,
            total_beam_intensity=bunch_intensity,
            nemitt_x=results.epsx[turn - 1],
            nemitt_y=results.epsy[turn - 1],
            sigma_delta=results.sigma_delta[turn - 1],
            bunch_length=results.bunch_length[turn - 1],
            # bunched=True,  # by default
        )
        print(f"Turn {turn:d}: re-computed IBS rates - {rates}")

    # Compute the new emittances etc and update
    results.update_with_ibs_and_sr_at_next_step(
        dt=dt,
        ibs_rates=rates,
        sr_eq_epsx=twiss.eq_nemitt_x,
        sr_eq_epsy=twiss.eq_nemitt_y,
        sr_eq_sigma_delta=sr_eq_sigma_delta,
        sr_taux=twiss.damping_constants_s[0],
        sr_tauy=twiss.damping_constants_s[1],
        sr_tauz=twiss.damping_constants_s[2],
        circumference=circumference,
        harmonic_number=harmonic_number,
        total_energy=total_energy,
        slip_factor=slip_factor,
        beta_rel=beta_rel,
        rf_voltage=RF_voltage,
        reference_charge=reference_charge,
    )


# Some quick plot
fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(10, 7))

axs["epsx"].plot(turns, 1e7 * results.epsx, lw=2)
axs["epsy"].plot(turns, 1e9 * results.epsy, lw=2)
axs["sigd"].plot(turns, 1e3 * results.sigma_delta, lw=2)
axs["bl"].plot(turns, 1e3 * results.bunch_length, lw=2)

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_{x}^{n}$ [$10^{-7}$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_{y}^{n}$ [$10^{-9}$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
axs["bl"].set_ylabel(r"Bunch length [mm]")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Turn Number")

for axis in axs.values():
    axis.yaxis.set_major_locator(plt.MaxNLocator(3))

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))

plt.tight_layout()
plt.show()
