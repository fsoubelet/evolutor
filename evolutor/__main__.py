from __future__ import annotations

import logging

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from matplotlib.ticker import MaxNLocator
from rich.logging import RichHandler
from rich.progress import Progress
from typer import Abort, Argument, Option, Typer

from evolutor import Records, energy_spread
from evolutor._constants import Formalisms, Modes

logging.basicConfig(level="DEBUG", handlers=[RichHandler(level="NOTSET")])
logging.getLogger("numba").setLevel(logging.CRITICAL)
logging.getLogger("xdeps").setLevel(logging.CRITICAL)
logging.getLogger("xfields").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
LOGGER = logging.getLogger("evolutor")

app = Typer(no_args_is_help=True)


@app.command()
def main(
    # ----- Arguments ------ #
    sequence: Path = Argument(
        file_okay=True,
        dir_okay=False,
        exists=True,
        resolve_path=True,
        show_default=False,  # required anyway
        help="Path to the sequence file.",
    ),
    mode: str = Argument(
        default=Modes.seconds,
        show_choices=True,
        help="Simulation mode, either per 'seconds' or 'turns'.",
    ),
    # ----- Options ------ #
    formalism: Formalisms = Option(
        default=...,
        show_choices=True,
        help="IBS formalism to use for growth rates.",
        rich_help_panel="IBS growth rates computing",
    ),
    rf_voltage: float = Option(
        min=0.0,
        default=...,
        help="RF voltage in [V].",
        rich_help_panel="IBS growth rates computing",
    ),
    harmonic_number: int = Option(
        min=1,
        default=...,
        help="Harmonic number of the ring.",
        rich_help_panel="IBS growth rates computing",
    ),
    bunch_intensity: float = Option(
        min=0.0,
        default=...,
        help="Bunch intensity in [ppb] (particles per bunch).",
        rich_help_panel="IBS growth rates computing",
    ),
    nemitt_x: float = Option(
        min=0.0,
        default=...,
        help="Normalized emittance in the horizontal plane in [m].",
        rich_help_panel="IBS growth rates computing",
    ),
    nemitt_y: float = Option(
        min=0.0,
        default=...,
        help="Normalized emittance in the vertical plane in [m].",
        rich_help_panel="IBS growth rates computing",
    ),
    sigma_z: float = Option(
        min=0.0,
        default=...,
        help="Bunch length in [m].",
        rich_help_panel="IBS growth rates computing",
    ),
    bunched: bool = Option(
        default=True,
        show_choices=True,
        help="Whether the beam is bunched or not. "
        "If False, the IBS growth rates are computed for a coasting beam.",
        rich_help_panel="IBS growth rates computing",
    ),
    nseconds: int = Option(
        min=0,
        default=None,
        help="Number of seconds to simulate. If mode is not 'seconds', this is rejected.",
        rich_help_panel="Global simulation parameters",
    ),
    nturns: int = Option(
        min=1,
        default=None,
        help="Number of turns to simulate. If mode is not 'turns', this is rejected.",
        rich_help_panel="Global simulation parameters",
    ),
    dt: int = Option(
        min=0,
        default=None,
        help="The time step in [s] between two data points. "
        "If mode is 'seconds', this defaults to 1s. If mode is 'turns', this defaults to the revolution time.",
        rich_help_panel="Global simulation parameters",
    ),
    recompute_step: int = Option(
        min=1,
        default=...,
        help="Re-compute the IBS growth rates every this many seconds or turns. ",
        rich_help_panel="Global simulation parameters",
    ),
    export: Path = Option(
        file_okay=True,
        dir_okay=False,
        exists=False,
        resolve_path=True,
        default=None,
        help="If provided, export the results to a .npz file with the given name.",
        rich_help_panel="Global simulation parameters",
    ),
) -> None:
    """Command line tool to run the IBS evolutor.

    Provided with a sequence file and required parameters, this tool runs the
    IBS evolutor simulation either per seconds or per turns, depending on the
    mode specified. The results can be exported to a .npz file if requested.
    """
    sequence = Path(sequence)

    if mode == Modes.seconds:
        # We make quick cheks: need nseconds, need to not provide nturns
        if nseconds is None:
            LOGGER.error("Please provide 'nseconds' in this mode.")
            raise Abort()
        if nturns is not None:
            LOGGER.error("Invalid option 'nturns' in 'seconds' mode.")
            raise Abort()

        line = xt.Line.from_json(sequence)
        if dt is None:
            LOGGER.info("Using default time step of 1s.")
            dt = 1  # default time step in [s]

        # And we run the simulation per seconds
        LOGGER.info("Starting simulation, per seconds mode.")
        handle_per_seconds(
            line=line,
            formalism=formalism,
            harmonic_number=harmonic_number,
            rf_voltage=rf_voltage,
            bunch_intensity=bunch_intensity,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            sigma_z=sigma_z,
            nseconds=nseconds,
            dt=dt,
            recompute_step=recompute_step,
            bunched=bunched,
            export=export,
        )

    elif mode == Modes.turns:
        # We make quick checks: need nturns, need to not provide nseconds
        if nturns is None:
            LOGGER.error("Please provide 'nturns' in this mode.")
            raise Abort(code=1)
        if nseconds is not None:
            LOGGER.error("Invalid option 'nseconds' in 'turns' mode.")
            raise Abort()

        line = xt.Line.from_json(sequence)
        if dt is None:
            LOGGER.info("Using default time step of the line's revolution time.")
            dt = line.twiss4d().T_rev0  # default time step in [s] (revolution time)

        # And we run the simulation per turns
        LOGGER.info("Starting simulation, per turns mode.")
        handle_per_turns(
            line=line,
            formalism=formalism,
            harmonic_number=harmonic_number,
            rf_voltage=rf_voltage,
            bunch_intensity=bunch_intensity,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            sigma_z=sigma_z,
            nturns=nturns,
            dt=dt,
            recompute_step=recompute_step,
            bunched=bunched,
            export=export,
        )

    LOGGER.info("Simulation completed successfully.")


# ----- Helper Functions ----- #


def handle_per_seconds(
    line: xt.Line,
    formalism: str,
    harmonic_number: int,
    rf_voltage: float,
    bunch_intensity: float,
    nemitt_x: float,
    nemitt_y: float,
    sigma_z: float,
    nseconds: int,
    dt: int,
    recompute_step: int,
    bunched: bool = True,
    export: str | None = None,
) -> None:
    """
    Run the IBS evolutor for a given line over a specified number of seconds.

    Parameters
    ----------
    line : xt.Line
        The line to simulate.
    formalism : str
        The IBS formalism to use for growth rates.
    harmonic_number : int
        The harmonic number of the ring.
    rf_voltage : float
        The RF voltage in [V].
    bunch_intensity : float
        The bunch intensity in [ppb] (particles per bunch).
    nemitt_x : float
        The normalized emittance in the horizontal plane in [m].
    nemitt_y : float
        The normalized emittance in the vertical plane in [m].
    sigma_z : float
        The bunch length in [m].
    nseconds : int
        The number of seconds to simulate.
    dt : int
        The time step in [s] between two data points.
    recompute_step : int
        Re-compute the IBS growth rates every this many seconds.
    bunched : bool, optional
        Whether the beam is bunched or not. If False, the IBS growth rates are computed for a coasting beam.
    export : str | None, optional
        If provided, export the results to a .npz file with the given name.
    """
    twiss = line.twiss4d()
    circumference = line.get_length()
    total_energy = np.sqrt(line.particle_ref.p0c[0] ** 2 + line.particle_ref.mass0**2)  # in [eV]
    slip_factor = twiss.slip_factor
    beta_rel = twiss.beta0
    reference_charge = line.particle_ref.q0

    # Computed from parameters provided by the user
    sigma_e = energy_spread(
        circumference,
        harmonic_number=harmonic_number,
        total_energy=total_energy,
        slip_factor=slip_factor,
        bunch_length=sigma_z,
        beta_rel=beta_rel,
        rf_voltage=rf_voltage,
        reference_charge=reference_charge,
    )
    sigma_delta = sigma_e / (beta_rel**2)

    # Prepare records & insert initial values
    nsteps = int(nseconds / dt)  # nsteps = nseconds / step_size
    results = Records(dt=dt, nsteps=nsteps)
    results.update_at_step(
        0, epsx=nemitt_x, epsy=nemitt_y, sigma_delta=sigma_delta, bunch_length=sigma_z
    )

    # Run the simulation
    with Progress(refresh_per_second=20, expand=True) as progress:
        task = progress.add_task("Running simulation...", total=nsteps)
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
                    bunched=bunched,
                )
                LOGGER.debug(f"At {results.times[step]:d}s: re-computed IBS rates - {rates}")

            # Compute the new emittances etc and update
            results.update_with_ibs_at_next_step(
                ibs_rates=rates,  # type: ignore
                circumference=circumference,
                harmonic_number=harmonic_number,
                total_energy=total_energy,
                slip_factor=slip_factor,
                beta_rel=beta_rel,
                rf_voltage=rf_voltage,
                reference_charge=reference_charge,
            )
            progress.update(task, advance=1)

    # Export results if requested
    if export is not None:
        LOGGER.info(f"Exporting results in 'npz' format at {export}.")
        np.savez(
            export,
            nemitt_x=results.epsx,
            nemitt_y=results.epsy,
            sigma_delta=results.sigma_delta,
            bunch_length=results.bunch_length,
            time_s=results.times,
        )

    # Make a plot to show the user
    fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(10, 7))

    # Potentially divide times by 3600 to get xaxis in [h]
    times = results.times / 3600 if np.max(results.times) > 3600 else results.times
    time_unit = "h" if np.max(results.times) > 3600 else "s"
    axs["epsx"].plot(times, 1e6 * results.epsx, lw=2)
    axs["epsy"].plot(times, 1e6 * results.epsy, lw=2)
    axs["sigd"].plot(times, 1e3 * results.sigma_delta, lw=2)
    axs["bl"].plot(times, 1e2 * results.bunch_length, lw=2)

    # Axes parameters
    axs["epsx"].set_ylabel(r"$\varepsilon_{x}^{n}$ [$10^{-6}$m]")
    axs["epsy"].set_ylabel(r"$\varepsilon_{y}^{n}$ [$10^{-6}$m]")
    axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
    axs["bl"].set_ylabel(r"Bunch length [cm]")

    for axis in (axs["epsy"], axs["bl"]):
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()

    for axis in (axs["sigd"], axs["bl"]):
        axis.set_xlabel(f"Duration [{time_unit}]")

    for axis in axs.values():
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(4))

    fig.align_ylabels((axs["epsx"], axs["sigd"]))
    fig.align_ylabels((axs["epsy"], axs["bl"]))

    plt.tight_layout()
    plt.show()


def handle_per_turns(
    line: xt.Line,
    formalism: str,
    harmonic_number: int,
    rf_voltage: float,
    bunch_intensity: float,
    nemitt_x: float,
    nemitt_y: float,
    sigma_z: float,
    nturns: int,
    dt: int,
    recompute_step: int,
    bunched: bool = True,
    export: str | None = None,
) -> None:
    """
    Run the IBS evolutor for a given line over a specified number of turns.

    Parameters
    ----------
    line : xt.Line
        The line to simulate.
    formalism : str
        The IBS formalism to use for growth rates.
    harmonic_number : int
        The harmonic number of the ring.
    rf_voltage : float
        The RF voltage in [V].
    bunch_intensity : float
        The bunch intensity in [ppb] (particles per bunch).
    nemitt_x : float
        The normalized emittance in the horizontal plane in [m].
    nemitt_y : float
        The normalized emittance in the vertical plane in [m].
    sigma_z : float
        The bunch length in [m].
    nturns : int
        The number of turns to simulate.
    dt : int
        The time step in [s] between two data points.
    recompute_step : int
        Re-compute the IBS growth rates every this many turns.
    bunched : bool, optional
        Whether the beam is bunched or not. If False, the IBS growth rates are computed for a coasting beam.
    export : str | None, optional
        If provided, export the results to a .npz file with the given name.
    """
    twiss = line.twiss4d()
    circumference = line.get_length()
    total_energy = np.sqrt(line.particle_ref.p0c[0] ** 2 + line.particle_ref.mass0**2)  # in [eV]
    slip_factor = twiss.slip_factor
    beta_rel = twiss.beta0
    reference_charge = line.particle_ref.q0

    # Computed from parameters provided by the user
    sigma_e = energy_spread(
        circumference,
        harmonic_number=harmonic_number,
        total_energy=total_energy,
        slip_factor=slip_factor,
        bunch_length=sigma_z,
        beta_rel=beta_rel,
        rf_voltage=rf_voltage,
        reference_charge=reference_charge,
    )
    sigma_delta = sigma_e / (beta_rel**2)

    # Prepare records & insert initial values
    turns = np.arange(nturns, dtype=int)
    results = Records(dt=dt, nsteps=nturns)
    results.update_at_step(
        0, epsx=nemitt_x, epsy=nemitt_y, sigma_delta=sigma_delta, bunch_length=sigma_z
    )

    # Run the simulation
    with Progress(refresh_per_second=20, expand=True) as progress:
        task = progress.add_task("Running simulation...", total=nturns)
        for step in range(1, nturns):
            # Potentially recompute growth rates
            if (step % recompute_step == 0) or (step == 1):
                rates = twiss.get_ibs_growth_rates(
                    formalism=formalism,
                    total_beam_intensity=bunch_intensity,
                    nemitt_x=results.epsx[step - 1],
                    nemitt_y=results.epsy[step - 1],
                    sigma_delta=results.sigma_delta[step - 1],
                    bunch_length=results.bunch_length[step - 1],
                    bunched=bunched,
                )
                LOGGER.debug(f"Turn {step:d}: re-computed IBS rates - {rates}")

            # Compute the new emittances etc and update
            results.update_with_ibs_at_next_step(
                ibs_rates=rates,  # type: ignore
                circumference=circumference,
                harmonic_number=harmonic_number,
                total_energy=total_energy,
                slip_factor=slip_factor,
                beta_rel=beta_rel,
                rf_voltage=rf_voltage,
                reference_charge=reference_charge,
            )
            progress.update(task, advance=1)

    # Export results if requested
    if export is not None:
        LOGGER.info(f"Exporting results in 'npz' format at {export}.")
        np.savez(
            export,
            nemitt_x=results.epsx,
            nemitt_y=results.epsy,
            sigma_delta=results.sigma_delta,
            bunch_length=results.bunch_length,
            turn=turns,
        )

    # Make a plot to show the user
    fig, axs = plt.subplot_mosaic([["epsx", "epsy"], ["sigd", "bl"]], sharex=True, figsize=(10, 7))

    axs["epsx"].plot(turns, 1e6 * results.epsx, lw=2)
    axs["epsy"].plot(turns, 1e6 * results.epsy, lw=2)
    axs["sigd"].plot(turns, 1e3 * results.sigma_delta, lw=2)
    axs["bl"].plot(turns, 1e2 * results.bunch_length, lw=2)

    # Axes parameters
    axs["epsx"].set_ylabel(r"$\varepsilon_{x}^{n}$ [$10^{-6}$m]")
    axs["epsy"].set_ylabel(r"$\varepsilon_{y}^{n}$ [$10^{-6}$m]")
    axs["sigd"].set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
    axs["bl"].set_ylabel(r"Bunch length [cm]")

    for axis in (axs["epsy"], axs["bl"]):
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()

    for axis in (axs["sigd"], axs["bl"]):
        axis.set_xlabel("Turn Number")

    for axis in axs.values():
        axis.yaxis.set_major_locator(MaxNLocator(3))

    fig.align_ylabels((axs["epsx"], axs["sigd"]))
    fig.align_ylabels((axs["epsy"], axs["bl"]))

    plt.tight_layout()
    plt.show()


# ----- Entrypoint ----- #

if __name__ == "__main__":
    app()
