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
    from pathlib import Path
    from typing import BinaryIO

    from xfields.ibs import IBSAmplitudeGrowthRates


@dataclass
class Records:
    """
    Dataclass to store and compute intermediate step emittance values.
    """

    dt: float  # time step in [s]
    nsteps: int  # number of steps to compute
    times: np.ndarray = field(init=False)  # time at each step, in [s]
    epsx: np.ndarray = field(init=False)  # horizontal emittance
    epsy: np.ndarray = field(init=False)  # vertical emittance
    sigma_delta: np.ndarray = field(init=False)  # momentum spread
    bunch_length: np.ndarray = field(init=False)  # bunch length
    current_step: int = 0  # to know where we're at

    def __post_init__(self):
        # We know the steps, we can preallocate the arrays
        self.times = np.arange(self.nsteps, dtype=int) * self.dt
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

    def savez(self, file: str | Path | BinaryIO) -> None:
        """
        Save the record's data to disk using numpy's savez function.

        Parameters
        ----------
        file : str | Path | BinaryIO
            The file to save the data to. This can be a string
            (filename), a Path object, or a binary file object.
        """
        np.savez(
            file,
            epsx=self.epsx,
            epsy=self.epsy,
            sigma_delta=self.sigma_delta,
            bunch_length=self.bunch_length,
        )

    @classmethod
    def load(cls, file: str | Path | BinaryIO) -> Records:
        """
        Load the record's data from disk using numpy's load function.

        Parameters
        ----------
        file : str | Path | BinaryIO
            The file to load the data from. This can be a string
            (filename), a Path object, or a binary file object.

        Returns
        -------
        Records
            An instance of the Records class with the loaded data.
        """
        with np.load(file) as data:
            instance = cls(nsteps=data["epsx"].size)
            instance.epsx = data["epsx"]
            instance.epsy = data["epsy"]
            instance.sigma_delta = data["sigma_delta"]
            instance.bunch_length = data["bunch_length"]
        return instance

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
        ibs_rates: IBSAmplitudeGrowthRates,
        # -----------------------------
        # Parameters needed for sigma_delta -> bunch_length conversion
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
        IBS growth rates. The instance will use its internal
        (provided at init time) time step. Note that this assumes
        growth rates to be in amplitude convention (since xfields
        0.23.0 and xtrack 0.80.0), same as SR quantities.

        Parameters
        ----------
        dt : float
            The time interval over which to compute
            the emittances' evolution in [s].
        ibs_rates : IBSAmplitudeGrowthRates
            The IBS growth rates in amplitude convention,
            in [1/s].
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
        """
        epsx, epsy, sigma_delta, _ = self.values_at_current_step
        # All is handled in this function right there
        new_epsx, new_epsy, new_sigma_delta, new_bunch_length = ibs_evolution(
            epsx=epsx,
            epsy=epsy,
            sigma_delta=sigma_delta,
            dt=self.dt,
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
            self.current_step + 1,
            new_epsx,
            new_epsy,
            new_sigma_delta,
            new_bunch_length,
        )
        self.current_step += 1

    def update_with_ibs_and_sr_at_next_step(
        self,
        ibs_rates: IBSAmplitudeGrowthRates,
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
    ) -> None:
        """
        Update the records for the next step, provided with the
        IBS growth rates and the synchrotron radiation equilibrium
        values as well as damping times. The instance will use its
        internal (provided at init time) time step. Note that this
        assumes growth rates to be in amplitude convention (since
        xfields 0.23.0 and xtrack 0.80.0), same as SR quantities.

        Note
        ----
            It is important that the Synchrotron Radiation equilibrium
            emittances are given in the same convention as the input
            emittances: provide either geometric or normalized for both.

        Parameters
        ----------
        dt : float
            The time interval over which to compute
            the emittances' evolution in [s].
        ibs_rates : IBSAmplitudeGrowthRates
            The IBS growth rates in amplitude convention,
            in [1/s].
        ----- Synchrotron Radiation parameters -----
        sr_eq_epsx : float
            Horizontal emittance at equilibrium from SR, in [m].
        sr_eq_epsy : float
            Vertical emittance at equilibrium from SR, in [m].
        sr_eq_sigma_delta : float
            Momentum spread at equilibrium from SR, in [-]. As
            twiss results give 'eq_gemitt_zeta' one can convert
            with sigma_delta = (gemitt_zeta / tw.bets0) ** 0.5
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
        """
        epsx, epsy, sigma_delta, _ = self.values_at_current_step
        # All is handled in this function right there
        new_epsx, new_epsy, new_sigma_delta, new_bunch_length = ibs_and_sr_evolution(
            epsx=epsx,
            epsy=epsy,
            sigma_delta=sigma_delta,
            dt=self.dt,
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
            self.current_step + 1,
            new_epsx,
            new_epsy,
            new_sigma_delta,
            new_bunch_length,
        )
        self.current_step += 1
