# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "setuptools",
#   "xsuite>=0.29.0",
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
simple evolution, then visualize it. This script shows how
to run the script and show results by seconds, the default.

This time the Synchrotron Radiation is included in the evolution.
"""

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from scipy.constants import c

from evolutor import Records, energy_spread

# Get LHC line and Twiss
line = xt.Line.from_json("lhcb1.json")
line.build_tracker()
line.configure_radiation(model="mean")
twiss = line.twiss(eneloss_and_damping=True)
formalism = "bjorken-mtingwa"  # or "nagaitsev" (faster) if there is no Dy

# Parameters computed once and needed later
circumference = line.get_length()
total_energy = np.sqrt(line.particle_ref.p0c[0] ** 2 + line.particle_ref.mass0**2)  # in [eV]
slip_factor = twiss.slip_factor
beta_rel = twiss.beta0
reference_charge = line.particle_ref.q0
sr_eq_sigma_delta = (twiss.eq_gemitt_zeta / twiss.bets0) ** 0.5

# -------------------------------- #
# Parameters to be set by the user #
# -------------------------------- #
harmonic_number = 34640
RF_voltage = 4000000.0  # in [V]
bunch_intensity = 2.3e11  # in [ppb]
nemitt_x = 2.1e-06  # in [m]
nemitt_y = 2.1e-06  # in [m]
bl = 1.3e-9  # bunch length in [s]
sigma_z = bl / 4.0*(c*beta_rel)   # in [m]
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

# We hardcode these SR times as for the
# LHC twiss struggles to compute them
sr_tau_x = 56 * 3600  # 56h in [s]
# sr_tau_x = twiss./damping_times[0]  # in [s]
sr_tau_y = 56 * 3600  # 56h in [s]
# sr_tau_y = twiss./damping_times[1]  # in [s]
sr_tau_z = 28 * 3600  # 28h in [s]
# sr_tau_z = twiss./damping_times[2]  # in [s]

# SET THESE TO DETERMINE THE SIMULATION - do not make recompute_step too high
nseconds = 8 * 3_600  # this is 8h of beam time simulated
recompute_step = 5 * 60  # re-compute growth rates every [s] (this is 5min)

# Prepare records
dt = 60  # time step in [s]
nsteps = int(nseconds / dt)  # nsteps = nseconds / step_size
results = Records(dt=dt, nsteps=nsteps)
# Insert the initial values
results.update_at_step(
    0,
    epsx=nemitt_x,
    epsy=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=sigma_z,
)


# Now this loop handles everything
for step in range(1, nsteps):
    # Potentially recompute growth rates
    if (results.times[step] % recompute_step == 0) or (step == 1):
        rates = twiss.get_ibs_growth_rates(
            formalism=formalism,
            total_beam_intensity=bunch_intensity,
            nemitt_x=results.epsx[step - 1],
            nemitt_y=results.epsy[step - 1],
            sigma_delta=results.sigma_delta[step - 1],
            bunch_length=results.bunch_length[step - 1],
            # bunched=True,  # by default
        )
        print(f"At {results.times[step]:d}s: re-computed IBS rates - {rates}")

    # Compute the new emittances etc and update
    results.update_with_ibs_and_sr_at_next_step(
        ibs_rates=rates,
        sr_eq_epsx=twiss.eq_nemitt_x,  # nemitt everywhere
        sr_eq_epsy=twiss.eq_nemitt_y,  # nemitt everywhere
        sr_eq_sigma_delta=sr_eq_sigma_delta,
        sr_taux=sr_tau_x,
        sr_tauy=sr_tau_y,
        sr_tauz=sr_tau_z,
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

# Divide times by 3600 to get xaxis in [h]
axs["epsx"].plot(results.times / 3600, 1e6 * results.epsx, lw=2)
axs["epsy"].plot(results.times / 3600, 1e6 * results.epsy, lw=2)
axs["sigd"].plot(results.times / 3600, 1e3 * results.sigma_delta, lw=2)
axs["bl"].plot(results.times / 3600, 1e2 * results.bunch_length, lw=2)

# Axes parameters
axs["epsx"].set_ylabel(r"$\varepsilon_{x}^{n}$ [$10^{-6}$m]")
axs["epsy"].set_ylabel(r"$\varepsilon_{y}^{n}$ [$10^{-6}$m]")
axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
axs["bl"].set_ylabel(r"Bunch length [cm]")

for axis in (axs["epsy"], axs["bl"]):
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()

for axis in (axs["sigd"], axs["bl"]):
    axis.set_xlabel("Duration [h]")

for axis in axs.values():
    axis.xaxis.set_major_locator(plt.MaxNLocator(5))
    axis.yaxis.set_major_locator(plt.MaxNLocator(4))

fig.align_ylabels((axs["epsx"], axs["sigd"]))
fig.align_ylabels((axs["epsy"], axs["bl"]))

plt.tight_layout()
plt.show()
