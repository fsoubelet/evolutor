"""
Record
------

Provides a convenient class to store intermediate
results and perform the evolution of these parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from evolutor._evolution import ibs_and_sr_evolution, ibs_evolution

if TYPE_CHECKING:
    from xfields.ibs import IBSAmplitudeGrowthRates


@dataclass
class Records:
    """
    Dataclass to store and compute intermediate step emittance values.
    """

    nsteps: int  # number of steps to compute
    epsx: np.ndarray = field(init=False)  # horizontal emittance
    epsy: np.ndarray = field(init=False)  # vertical emittance
    sigma_delta: np.ndarray = field(init=False)  # momentum spread
    bunch_length: np.ndarray = field(init=False)  # bunch length
    current_step: int = 0  # to know where we're at

    def __post_init__(self):
        # We know the steps, we can preallocate the arrays
        self.epsx = np.zeros(self.nsteps, dtype=np.float64)
        self.epsy = np.zeros(self.nsteps, dtype=np.float64)
        self.sigma_delta = np.zeros(self.nsteps, dtype=np.float64)
        self.bunch_length = np.zeros(self.nsteps, dtype=np.float64)

    @property
    def values_at_current_step(self) -> tuple[float, float, float, float]:
        """Return the values at the current step."""
        return (
            self.epsx[self.current_step],
            self.epsy[self.current_step],
            self.sigma_delta[self.current_step],
            self.bunch_length[self.current_step],
        )

    def update_at_step(
        self, step: int, epsx: float, epsy: float, sigma_delta: float, bunch_length: float
    ) -> None:
        """Update the internal records with values for a given step (index)."""
        self.epsx[step] = epsx
        self.epsy[step] = epsy
        self.sigma_delta[step] = sigma_delta
        self.bunch_length[step] = bunch_length

    def update_with_ibs_at_next_step(
        self,
        dt: float,
        ibs_rates: IBSAmplitudeGrowthRates,
        circumference: float,
        harmonic_number: int,
        total_energy: float,
        slip_factor: float,
        beta_rel: float,
        rf_voltage: float,
        reference_charge: float,
    ) -> None:
        """
        Update the records for the next step, provided with the
        time delta between the steps and the IBS growth rates.
        Note that this assumes growth rates to be in amplitude
        convention (since xfields 0.23.0 and xtrack 0.80.0).
        """
        epsx, epsy, sigma_delta, _ = self.values_at_current_step
        # All is handled in this function right there
        new_epsx, new_epsy, new_sigma_delta, new_bunch_length = ibs_evolution(
            epsx=epsx,
            epsy=epsy,
            sigma_delta=sigma_delta,
            dt=dt,
            Kx=ibs_rates.Kx,
            Ky=ibs_rates.Ky,
            Kz=ibs_rates.Kz,
            circumference=circumference,
            harmonic_number=harmonic_number,
            total_energy=total_energy,
            slip_factor=slip_factor,
            beta_rel=beta_rel,
            rf_voltage=rf_voltage,
            reference_charge=reference_charge,
        )
        # Update the records - we took values at current step
        # to determine the values at current step + 1
        self.update_at_step(
            self.current_step + 1, new_epsx, new_epsy, new_sigma_delta, new_bunch_length
        )
        self.current_step += 1

    def update_with_ibsand_sr_at_next_step(
        self,
        dt: float,
        ibs_rates: IBSAmplitudeGrowthRates,
        sr_eq_epsx: float,
        sr_eq_epsy: float,
        sr_eq_sigma_delta: float,
        sr_taux: float,
        sr_tauy: float,
        sr_tauz: float,
        circumference: float,
        harmonic_number: int,
        total_energy: float,
        slip_factor: float,
        beta_rel: float,
        rf_voltage: float,
        reference_charge: float,
    ) -> None:
        """
        Update the records for the next step, provided with the
        time delta between the steps, the IBS growth rates and
        the synchrotron radiation equilibrium values as well as
        damping times. Note that this assumes growth rates to be
        in amplitude convention (since xfields 0.23.0 and xtrack
        0.80.0).
        """
        epsx, epsy, sigma_delta, _ = self.values_at_current_step
        # All is handled in this function right there
        new_epsx, new_epsy, new_sigma_delta, new_bunch_length = ibs_and_sr_evolution(
            epsx=epsx,
            epsy=epsy,
            sigma_delta=sigma_delta,
            dt=dt,
            Kx=ibs_rates.Kx,
            Ky=ibs_rates.Ky,
            Kz=ibs_rates.Kz,
            sr_eq_epsx=sr_eq_epsx,
            sr_eq_epsy=sr_eq_epsy,
            sr_eq_sigma_delta=sr_eq_sigma_delta,
            sr_taux=sr_taux,
            sr_tauy=sr_tauy,
            sr_tauz=sr_tauz,
            circumference=circumference,
            harmonic_number=harmonic_number,
            total_energy=total_energy,
            slip_factor=slip_factor,
            beta_rel=beta_rel,
            rf_voltage=rf_voltage,
            reference_charge=reference_charge,
        )
        # Update the records - we took values at current step
        # to determine the values at current step + 1
        self.update_at_step(
            self.current_step + 1, new_epsx, new_epsy, new_sigma_delta, new_bunch_length
        )
        self.current_step += 1
