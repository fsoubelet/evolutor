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
    circumference: float,
    harmonic_number: int,
    total_energy: float,
    slip_factor: float,
    sigma_e: float,
    beta_rel: float,
    rf_voltage: float,
    reference_charge: float,
) -> float:
    """
    Get the bunch length from the energy spread. I have removed
    Energy_loss from the function of Michalis as it was not used
    (and Sofia was passing 0.0).

    Parameters
    ----------
    circumference : float
        The ring circumference in [m].
    harmonic_number : int
        The harmonic number.
    total_energy : float
        The total energy of the particles in [eV].
    slip_factor : float
        The slip factor (as from xsuite Twiss).
    sigma_e : float
        The energy spread (sigma_delta * beta_rel^2).
    beta_rel : float
        The relativistic beta.
    rf_voltage : float
        The RF voltage in [V] (config['V0max'] * 1e6).
    reference_charge : float
        The reference charge (this is xt.Particles.q0)

    Returns
    -------
    float
        The bunch length in [m].
    """
    return (
        circumference
        / (2.0 * np.pi * harmonic_number)
        * np.arccos(
            1
            - (sigma_e**2 * total_energy * abs(slip_factor) * harmonic_number * np.pi)
            / (beta_rel**2 * reference_charge * rf_voltage)
        )
    )


@maybe_jit
def energy_spread(
    circumference: float,
    harmonic_number: int,
    total_energy: float,
    slip_factor: float,
    bunch_length: float,
    beta_rel: float,
    rf_voltage: float,
    reference_charge: float,
) -> float:
    """
    Get the energy spread from bunch length. I have removed
    Energy_loss from the function of Michalis as it was not
    used (and Sofia was passing 0.0).

    Parameters
    ----------
    circumference : float
        The ring circumference in [m].
    harmonic_number : int
        The harmonic number.
    total_energy : float
        The total energy of the particles in [eV].
    slip_factor : float
        The slip factor (as from xsuite Twiss).
    bunch_length : float
        The bunch length in [m].
    beta_rel : float
        The relativistic beta.
    rf_voltage : float
        The RF voltage in [V] (config['V0max'] * 1e6).
    reference_charge : float
        The reference charge (this is xt.Particles.q0)

    Returns
    -------
    float
        The energy spread in [-].
    """
    tau_phi = 2 * np.pi * harmonic_number * bunch_length / circumference
    return np.sqrt(
        beta_rel**2
        * reference_charge
        * rf_voltage
        * (-(np.cos(tau_phi) - 1))
        / (total_energy * abs(slip_factor) * harmonic_number * np.pi)
    )
