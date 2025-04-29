"""
Evolution
---------

Provides functionality to compute new emittance values due
to Intra-Beam Scattering from analytical growth rates and
formulae, over a given time interval.
"""

from __future__ import annotations

import numpy as np
from cycler import K

from evolutor._converters import bunch_length
from evolutor._jit import maybe_jit


@maybe_jit
def ibs_evolution(
    epsx: float,
    epsy: float,
    sigma_delta: float,
    dt: float,
    Kx: float,
    Ky: float,
    Kz: float,
    # -----------------------------
    # Parameters needed for sigma_delta -> bunch_length conversion
    circumference: float,
    harmonic_number: int,
    total_energy: float,
    slip_factor: float,
    beta_rel: float,
    rf_voltage: float,
    reference_charge: float,
) -> tuple[float, float, float, float]:
    """
    Return new emittances from IBS growth/decay over the interval dt.
    The new bunch length is computed via the new sigma delta (via the
    corresponding energy spread).

    Parameters
    ----------
    epsx : float
        Horizontal emittance, in [m].
    epsy : float
        Vertical emittance, in [m].
    sigma_delta : float
        Momentum spread, in [-].
    bunch_length : float
        Longitudinal bunch length, in [m].
    dt : float
        The time interval over which to compute
        the emittances' evolution in [s].
    Kx : float
        The horizontal IBS growth rate in
        amplitude convention, in [1/s].
    Ky : float
        The vertical IBS growth rate in
        amplitude convention, in [1/s].
    Kz : float
        The longitudinal IBS growth rate in
        amplitude convention, in [1/s].
    ----- Bunch length from sigma delta parameters -----
    circumference : float
        The ring circumference in [m].
    harmonic_number : int
        The harmonic number.
    total_energy : float
        The total energy of the particles in [eV].
        (np.sqrt(particles.p0c[0]**2 + particles.mass0**2)).
    slip_factor : float
        The slip factor (as gotten from xsuite Twiss).
    beta_rel : float
        The relativistic beta.
    rf_voltage : float
        The RF voltage in [V] (config['V0max'] * 1e6).
    reference_charge : float
        The reference charge (this is xt.Particles.q0)

    Returns
    -------
    tuple[float, float, float, float]
        The new transverse emittances, sigma delta
        and bunch length after the time interval dt.
    """
    # ----------------------------------------------
    # Compute new transverse emittances and sigma_delta
    new_epsx: float = epsx * np.exp(dt * 2 * Kx)
    new_epsy: float = epsy * np.exp(dt * 2 * Ky)
    new_sigma_delta: float = sigma_delta * np.exp(dt * Kz)
    # ----------------------------------------------
    # Get new bunch length value from sigma delta (not exponential growth)
    sigma_E: float = new_sigma_delta * beta_rel**2  # get energy spread from sigma delta
    new_bunch_length: float = bunch_length(
        circumference,
        harmonic_number,
        total_energy,
        slip_factor,
        sigma_E,
        beta_rel,
        rf_voltage,
        reference_charge,
    )
    return new_epsx, new_epsy, new_sigma_delta, new_bunch_length


@maybe_jit
def ibs_and_sr_evolution(
    epsx: float,
    epsy: float,
    sigma_delta: float,
    dt: float,
    Kx: float,
    Ky: float,
    Kz: float,
    # -----------------------------
    # Parameters for the SR properties
    sr_eq_epsx: float,
    sr_eq_epsy: float,
    sr_eq_sigma_delta: float,
    sr_taux: float,
    sr_tauy: float,
    sr_tauz: float,
    # -----------------------------
    # Parameters needed for sigma_delta -> bunch_length conversion
    circumference: float,
    harmonic_number: int,
    total_energy: float,
    slip_factor: float,
    beta_rel: float,
    rf_voltage: float,
    reference_charge: float,
) -> tuple[float, float, float, float]:
    """
    Return new emittances from IBS growth/decay AND Synchrotron Radiation
    over the interval dt. The new bunch length is computed via the new sigma
    delta (via the corresponding energy spread).

    Note
    ----
        It is important that the Synchrotron Radiation equilibrium
        emittances are given in the same convention as the input
        emittances: provide either geometric or normalized for both.

    Parameters
    ----------
    epsx : float
        Horizontal emittance, in [m].
    epsy : float
        Vertical emittance, in [m].
    sigma_delta : float
        Momentum spread, in [-].
    bunch_length : float
        Longitudinal bunch length, in [m].
    dt : float
        The time interval over which to compute
        the emittances' evolution in [s].
    Kx : float
        The horizontal IBS growth rate in
        amplitude convention, in [1/s].
    Ky : float
        The vertical IBS growth rate in
        amplitude convention, in [1/s].
    Kz : float
        The longitudinal IBS growth rate in
        amplitude convention, in [1/s].
    ----- Synchrotron Radiation parameters -----
    sr_eq_epsx : float
        Horizontal emittance at equilibrium from SR, in [m].
    sr_eq_epsy : float
        Vertical emittance at equilibrium from SR, in [m].
    sr_eq_sigma_delta : float
        Momentum spread at equilibrium from SR, in [-]. As the
        twiss results give 'eq_gemitt_zeta' one can convert
        with sigma_delta = (gemitt_zeta / twiss.bets0) ** 0.5
    sr_taux : float
        Horizontal damping time from SR, in [s]. This is
        the first value of 'tw.damping_times_s'.
    sr_tauy : float
        Vertical damping time from SR, in [s]. This is
        the second value of 'tw.damping_times_s'.
    sr_tauz : float
        Longitudinal damping time from SR, in [s]. This is
        the last value of 'tw.damping_times_s'.
    ----- Bunch length from sigma delta parameters -----
    circumference : float
        The ring circumference in [m].
    harmonic_number : int
        The harmonic number.
    total_energy : float
        The total energy of the particles in [eV].
        (np.sqrt(particles.p0c[0]**2 + particles.mass0**2)).
    slip_factor : float
        The slip factor (as gotten from xsuite Twiss).
    beta_rel : float
        The relativistic beta.
    rf_voltage : float
        The RF voltage in [V] (config['V0max'] * 1e6).
    reference_charge : float
        The reference charge (this is xt.Particles.q0)

    Returns
    -------
    tuple[float, float, float, float]
        The new transverse emittances, sigma delta
        and bunch length after the time interval dt.
    """
    # ----------------------------------------------
    # Compute new transverse emittances and sigma_delta
    new_epsx: float = (
        -sr_eq_epsx
        + (sr_eq_epsx + epsx * (Kx * sr_taux - 1.0)) * np.exp(2 * dt * (Kx - 1 / sr_taux))
    ) / (Kx * sr_taux - 1)
    new_epsy: float = (
        -sr_eq_epsy
        + (sr_eq_epsy + epsy * (Ky * sr_tauy - 1.0)) * np.exp(2 * dt * (Ky - 1 / sr_tauy))
    ) / (Ky * sr_tauy - 1)
    new_sigma_delta: float = (
        -sr_eq_sigma_delta
        + (sr_eq_sigma_delta + sigma_delta * (Kz * sr_tauz - 1.0)) * np.exp(dt * (Kz - 1 / sr_tauz))
    ) / (Kz * sr_tauz - 1.0)
    # ----------------------------------------------
    # Get new bunch length value from sigma delta (not exponential growth)
    sigma_E: float = new_sigma_delta * beta_rel**2  # get energy spread from sigma delta
    new_bunch_length: float = bunch_length(
        circumference,
        harmonic_number,
        total_energy,
        slip_factor,
        sigma_E,
        beta_rel,
        rf_voltage,
        reference_charge,
    )
    return new_epsx, new_epsy, new_sigma_delta, new_bunch_length
