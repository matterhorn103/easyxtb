# SPDX-FileCopyrightText: 2024 Matthew J. Milner <matterhorn103@proton.me>
# SPDX-License-Identifier: BSD-3-Clause

"""
Provides a simpler function-based calculation API.

The function-based API returns just the values of interest for a given geometry in a
quick and intuitive fashion, without the user having to worry about `Calculation`
objects.
"""

import logging
from collections.abc import Callable

from .calc import Calculation
from .geometry import Geometry

logger = logging.getLogger(__name__)


def energy(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> float:
    """Calculate the energy in hartree of the given geometry."""

    calc = Calculation.sp(
        input_geometry,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        molden=False,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.energy


def optimize(
    input_geometry: Geometry,
    level: str | None = None,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> Geometry:
    """Optimize the geometry, starting from the provided initial geometry, and return
    the optimized geometry."""

    calc = Calculation.opt(
        input_geometry,
        level=level,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    # Check for convergence
    # TODO
    # Will need to look for "FAILED TO CONVERGE"
    return calc.output_geometry


def frequencies(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Calculate vibrational frequencies and return results as a list of dicts."""

    calc = Calculation.hess(
        input_geometry,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.frequencies


def smartopt_calculation(
    input_geometry: Geometry,
    level: str | None = None,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
    max_restarts: int | None = None,
) -> Calculation:
    """Optimize the geometry repeatedly until a minimum is reached, and return the
    final `Calculation`.

    If xtb is run with the `ohess` runtype (an optimization and subsequent frequency
    calculation) and the program detects one or more imaginary frequencies in the
    result, it distorts the optimized geometry away from the stationary point and
    recommends to restart the calculation from that distorted geometry.

    This function takes advantage of this and does that restart automatically, and will
    continue to restart the optimizations until a minimum is reached.

    Unlike `smartopt()`, which returns only the optimized geometry, this returns the
    `Calculation` that produced it, so that the energy, partial charges and vibrational
    frequencies of the final minimum are available as well. Those come for free with
    the run that has just been paid for, and recovering them otherwise would mean
    repeating the frequency calculation.

    Parameters
    ----------
    max_restarts
        Maximum number of times to restart from a distorted geometry, or `None` for no
        limit. xtb usually needs at most a couple of restarts, but nothing guarantees
        that it ever reaches a minimum, so a caller that cannot afford to wait forever
        — a graphical interface, for instance — should set a limit. When the limit is
        reached the last calculation is returned as it is, so the caller should check
        its frequencies if it needs to know whether a genuine minimum was found.
    """

    calc = Calculation.ohess(
        input_geometry=input_geometry,
        solvation=solvation,
        method=method,
        level=level,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)

    # Check for distorted geometry
    # (Generated automatically by xtb if result had negative frequency)
    # If found, rerun
    distorted_geom_file = calc.output_file.with_name("xtbhess.xyz")
    restarts = 0
    while distorted_geom_file.exists():
        if max_restarts is not None and restarts >= max_restarts:
            logger.warning(f"Still not at a minimum after {restarts} restarts, giving up")
            break
        restarts += 1
        logger.debug(f"Imaginary frequency found, restart {restarts}")
        calc = Calculation.ohess(
            # Loaded before the new Calculation runs, as running it empties the
            # directory that this file is in.
            input_geometry=Geometry.load_file(
                distorted_geom_file, charge=input_geometry.charge, spin=input_geometry.spin
            ),
            solvation=solvation,
            method=method,
            level=level,
            n_proc=n_proc,
            options=options,
        )
        calc.run(on_output_line=on_output_line)

    return calc


def smartopt(
    input_geometry: Geometry,
    level: str | None = None,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
    max_restarts: int | None = None,
) -> Geometry:
    """Optimize the geometry repeatedly until a minimum is reached.

    See `smartopt_calculation()`, which this wraps, for the details of the restarting
    and for access to the energy and frequencies of the final geometry.
    """

    return smartopt_calculation(
        input_geometry=input_geometry,
        level=level,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        options=options,
        on_output_line=on_output_line,
        max_restarts=max_restarts,
    ).output_geometry


def orbitals(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> str:
    """Calculate molecular orbitals for given geometry.

    Returns a string of the Molden-format output file, which contains principally the
    GTO and MO information.
    """

    calc = Calculation.sp(
        input_geometry=input_geometry,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        molden=True,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.output_molden


def conformers(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    ewin: int | float = 6,
    hess: bool = False,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Simulate a conformer ensemble and return set of conformer Geometries and energies.

    The returned conformers are ordered from lowest to highest energy.

    All conformers within <ewin> kcal/mol are kept.
    If hess=True, vibrational frequencies are calculated and the conformers reordered by Gibbs energy.
    """

    calc = Calculation.v3(
        input_geometry=input_geometry,
        solvation=solvation,
        method=method,
        ewin=ewin,
        hess=hess,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.conformers


def tautomerize(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Sample prototropic tautomers and return set of tautomer Geometries and energies.

    The returned tautomers are ordered from lowest to highest energy.
    """

    calc = Calculation.tautomerize(
        input_geometry=input_geometry,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.tautomers


def protonate(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Screen possible protonation sites and return set of tautomer Geometries and energies.

    The returned tautomers are ordered from lowest to highest energy.
    """

    calc = Calculation.protonate(
        input_geometry=input_geometry,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.tautomers


def deprotonate(
    input_geometry: Geometry,
    solvation: str | None = None,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> list[dict]:
    """Screen possible deprotonation sites and return set of tautomer Geometries and energies.

    The returned tautomers are ordered from lowest to highest energy.
    """

    calc = Calculation.deprotonate(
        input_geometry=input_geometry,
        solvation=solvation,
        method=method,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.tautomers


def solvate(
    solute_geometry: Geometry,
    solvent_geometry: Geometry,
    nsolv: int,
    method: int | None = None,
    n_proc: int | None = None,
    options: dict | None = None,
    on_output_line: Callable[[str], None] | None = None,
) -> Geometry:
    """Grow a solvent shell around a solute for a total of `nsolv` solvent molecules.

    Note that non-zero charge and spin on the solvent `Geometry` will not be passed to
    CREST.
    """

    calc = Calculation.qcg(
        solute_geometry=solute_geometry,
        solvent_geometry=solvent_geometry,
        nsolv=nsolv,
        method=method,
        n_proc=n_proc,
        options=options,
    )
    calc.run(on_output_line=on_output_line)
    return calc.output_geometry
