"""
Converters
----------

Provides simple functions to convert bunch length and energy
spread from various ring and distribution parameters.
"""

from __future__ import annotations

import numpy as np

from evolutor._jit import maybe_jit

# These two functions below have been copied from Michalis
# (they also work for protons). Originally copied from his Github
# (MichZampetakis/IBS_for_Xsuite/blob/main/lib/general_functions.py)


@maybe_jit
def bunch_length(
    Circumference: float,
    Harmonic_Num: int,
    Energy_total: float,
    SlipF: float,
    Sigma_E: float,
    beta_rel: float,
    RF_Voltage: float,
    Z: float,
) -> float:
    """Get bunch length from the energy spread. I removed Energy_loss from the function
    of Michalis as it was not used (and Sofia was passing 0.0).

    Circumference : float
        The ring circumference in [m].
    Harmonic_Num : int
        The harmonic number.
    Energy_total : float
        The total energy of the particles in [eV] (np.sqrt(particles.p0c[0]**2 + particles.mass0**2)).
    SlipF : float
        The slip factor (as gotten from xsuite Twiss).
    Sigma_E : float
        The energy spread. (it is sigma_delta * beta_rel^2)
    beta_rel : float
        The relativistic beta.
    RF_Voltage : float
        The RF voltage in [V]. (config['V0max'] * 1e6))
    Z : float
        The total charge (this is the xp.Particles.q0)
    """
    return (
        Circumference
        / (2.0 * np.pi * Harmonic_Num)
        * np.arccos(
            1
            - (Sigma_E**2 * Energy_total * abs(SlipF) * Harmonic_Num * np.pi)
            / (beta_rel**2 * Z * RF_Voltage)
        )
    )


@maybe_jit
def energy_spread(
    Circumference: float,
    Harmonic_Num: int,
    Energy_total: float,
    SlipF: float,
    BL: float,
    beta_rel: float,
    RF_Voltage: float,
    Z: float,
) -> float:
    """Get the energy spread from bunch length. I removed Energy_loss from the function
    of Michalis as it was not used (and Sofia was passing 0.0).

    Circumference : float
        The ring circumference in [m].
    Harmonic_Num : int
        The harmonic number.
    Energy_total : float
        The total energy of the particles in [eV].
    SlipF : float
        The slip factor (as gotten from xsuite Twiss).
    BL : float
        The bunch length in [m].
    beta_rel : float
        The relativistic beta.
    RF_Voltage : float
        The RF voltage in [V]. (config['V0max'] * 1e6))
    Z : float
        The total charge (this is the xp.Particles.q0)
    """
    tau_phi = 2 * np.pi * Harmonic_Num * BL / Circumference  # bunch length in rad
    return np.sqrt(
        beta_rel**2
        * Z
        * RF_Voltage
        * (-(np.cos(tau_phi) - 1))
        / (Energy_total * abs(SlipF) * Harmonic_Num * np.pi)
    )
