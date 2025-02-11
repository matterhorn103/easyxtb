# SPDX-FileCopyrightText: 2024 Matthew J. Milner <matterhorn103@proton.me>
# SPDX-License-Identifier: BSD-3-Clause

"""
Provides the key `Calculation` class.

The `Calculation` object provides the infrastructure and methods to launch a calculation
on the command line and is used by all other calculation options.
"""

import logging
import os
import subprocess
from pathlib import Path
from shutil import rmtree

from .configuration import config, TEMP_DIR
from .geometry import Geometry
from .parse import parse_energy, parse_g98_frequencies, parse_mulliken_charges
from .program import Program, XTB, CREST


logger = logging.getLogger(__name__)


class Calculation:
    def __init__(
        self,
        program: Program,
        runtype: str | None = None,
        runtype_args: list | None = None,
        options: dict | None = None,
        command: list | None = None,
        input_geometry: Geometry | None = None,
        calc_dir: os.PathLike | None = None,
    ):
        """A convenience object to prepare, launch, and hold the results of
        calculations.

        `options` are the flags passed to `xtb` or `crest` and should be given
        without preceding minuses, as the appropriate number will be added
        automatically.
        To use a flag that takes no argument, set the value to `True`.
        Flags with values of `False` or `None` will not be passed.

        `Geometry` objects passed as items in `runtype_args` or `command` or as values
        in `options` will be saved as XYZ files and the path to the file will be put in
        the command in its place.

        Note that any contents of `calc_dir` will be removed when the calculation
        begins.
        """
        self.program = program
        if runtype:
            self.runtype = runtype
        elif options:
            # Runtype not specified but options were passed, so default runtype (single
            # point) is being requested
            self.runtype = None
        elif command:
            # User has passed full command themselves
            first_flag = command.split()[0].lstrip("-")
            if first_flag in self.program.runtypes:
                self.runtype = first_flag
        else:
            # Just a simple single point with no special options or flags or anything
            self.runtype = None
        self.runtype_args = runtype_args if runtype_args else []
        self.options = options if options else {}
        self.command = command
        self.input_geometry = input_geometry
        self.calc_dir = Path(calc_dir) if calc_dir else TEMP_DIR
        logger.info(f"New Calculation created for runtype {self.runtype} with {self.program.name}")

    def _build_xtb_command(self, geom_file):
        # Build command line args
        # "xtb"
        command = [self.program.path]
        # Charge and spin from the initial Geometry
        if "chrg" not in self.options and self.input_geometry.charge != 0:
            command.extend(["--chrg", self.input_geometry.charge])
        if "uhf" not in self.options and self.input_geometry.spin != 0:
            command.extend(["--uhf", self.input_geometry.spin])
        # Any other options
        for flag, value in self.options.items():
            # Add appropriate number of minuses to flags
            if len(flag) == 1:
                flag = "-" + flag
            else:
                flag = "--" + flag
            if value is True:
                command.append(flag)
            elif value is False or value is None:
                continue
            elif isinstance(value, Geometry):
                command.extend([flag, value])
            else:
                command.extend([flag, value])
        # Which calculation to run
        if self.runtype is None:
            # Simple single point
            pass
        else:
            command.extend(
                [
                    "--" + self.runtype,
                    *self.runtype_args,
                ]
            )
        # Path to the input geometry file, preceded by a -- to ensure that it is not
        # parsed as an argument to the runtype
        command.extend(["--", geom_file])
        return command

    def _build_crest_command(self, geom_file):
        # Build command line args
        # "crest"
        command = [self.program.path]
        # Path to the input geometry file
        command.append(geom_file)
        # Which calculation to run
        if self.runtype is None:
            # v3 iMTD-GC sampling
            pass
        else:
            command.extend(
                [
                    "--" + self.runtype,
                    *self.runtype_args,
                ]
            )
        # Charge and spin from the initial Geometry
        if "chrg" not in self.options and self.input_geometry.charge != 0:
            command.extend(["--chrg", self.input_geometry.charge])
        if "uhf" not in self.options and self.input_geometry.spin != 0:
            command.extend(["--uhf", self.input_geometry.spin])
        # Any other options
        for flag, value in self.options.items():
            # Add appropriate number of minuses to flags
            if len(flag) == 1:
                flag = "-" + flag
            else:
                flag = "--" + flag
            if value is True:
                command.append(flag)
            elif value is False or value is None:
                continue
            else:
                command.extend([flag, value])
        return command
    
    def preview_command(self):
        """Return the command that will be used to run xtb or crest based on the current
        attributes of the `Calculation`."""
        if self.command:
            # If arguments were passed by the user, use them as is
            command = self.command
        elif self.program.name == "crest":
            # CREST parses command line input in a slightly different way to xtb
            command = self._build_crest_command(self.input_geometry)
        else:
            command = self._build_xtb_command(self.input_geometry)
        return command

    def run(self):
        """Run calculation with xtb or crest, storing the output, the saved output file,
        and the parsed energy, as well as the `subprocess` object."""

        logger.debug(f"Calculation of runtype {self.runtype} has been asked to run")
        logger.debug(f"The calculation will be run in the following directory: {self.calc_dir}")

        # Make sure directory nominated for calculation exists and is empty
        if self.calc_dir.exists():
            for x in self.calc_dir.iterdir():
                if x.is_file():
                    x.unlink()
                elif x.is_dir():
                    rmtree(x)
        else:
            self.calc_dir.mkdir(parents=True)

        # Save geometry to file
        geom_file = self.calc_dir / "input.xyz"
        logger.debug(f"Saving input geometry to {geom_file}")
        self.input_geometry.write_xyz(geom_file)
        # We are using proper paths for pretty much everything so it shouldn't be
        # necessary to change the working dir
        # But we do anyway to be absolutely that xtb runs correctly and puts all the
        # output here
        os.chdir(self.calc_dir)
        logger.debug(f"Working directory changed to {Path.cwd()}")
        self.output_file = geom_file.with_name("output.out")

        # Build command line args
        if self.command:
            # If arguments were passed by the user, use them as is
            command = self.command
        elif self.program.name == "crest":
            # CREST parses command line input in a slightly different way to xtb
            command = self._build_crest_command(geom_file)
        else:
            command = self._build_xtb_command(geom_file)
        # Replace any Geometry objects with paths to files
        aux_count = 0
        for i, arg in enumerate(command):
            if isinstance(arg, Geometry):
                aux_count += 1
                aux_file = arg.write_xyz(self.calc_dir / f"aux{aux_count}.xyz")
                command[i] = aux_file
        # Sanitize everything to strings
        command = [x if isinstance(x, str) else str(x) for x in command]
        logger.debug(f"Calculation will be run with the command: {' '.join(command)}")

        # Run xtb or crest from command line
        logger.debug("Running calculation in new subprocess...")
        subproc = subprocess.run(command, capture_output=True, encoding="utf-8")
        logger.debug("...calculation complete.")

        # Store output
        self.output = subproc.stdout
        # Also save it to file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(self.output)
            logger.debug(f"Calculation output saved to {self.output_file}")

        # Store the subprocess.CompletedProcess object too
        self.subproc = subproc

        # Do all post-calculation processing according to which program was run
        if self.program.name == "crest":
            self.process_crest()
        else:
            self.process_xtb()

    def process_xtb(self):
        # First do generic operations that apply to many calculation types
        # Extract energy from output (if not found, returns None)
        self.energy = parse_energy(self.output)
        logger.debug(f"Final energy parsed from output file: {self.energy}")
        # If there's an output geometry, read it
        if self.output_file.with_name("xtbopt.xyz").exists():
            logger.debug(f"Output geometry found at {self.output_file.with_name('xtbopt.xyz')}")
            self.output_geometry = Geometry.load_file(
                self.output_file.with_name("xtbopt.xyz"),
                charge=self.input_geometry.charge,
                spin=self.input_geometry.spin,
            )
            logger.debug("Read output geometry")
        else:
            # Assume geom was the same at end of calc as at beginning
            logger.debug("No output geometry found - final geometry assumed to be same as initial")
            self.output_geometry = self.input_geometry
        # Mulliken atomic charges
        if self.output_file.with_name("charges").exists():
            logger.debug(
                f"Partial charges printout found at {self.output_file.with_name('charges')}"
            )
            with open(self.output_file.with_name("charges"), encoding="utf-8") as f:
                charges_string = f.read()
            mulliken_charges = parse_mulliken_charges(charges_string)
            self.partial_charges = {"mulliken": mulliken_charges}
            logger.debug("Mulliken partial charges were found and read")
        # Orbital info
        # If Molden output was requested, read the file
        if self.options.get("molden", False):
            logger.debug("Molden output was requested, so checking for file")
            with open(self.output_file.with_name("molden.input"), encoding="utf-8") as f:
                molden_string = f.read()
            self.output_molden = molden_string
            logger.debug("Molden output was found and read")

        # Now do more specific operations based on calculation type
        match self.runtype:
            case "hess" | "ohess":
                logger.debug("Frequencies were requested, so checking for file")
                # Read the Gaussian output file with frequencies
                with open(self.output_file.with_name("g98.out"), encoding="utf-8") as f:
                    g98_string = f.read()
                self.frequencies = parse_g98_frequencies(g98_string)
            case _:
                pass

    def process_crest(self):
        match self.runtype:
            case "v1" | "v2" | "v2i" | "v3" | "v4" | None:
                # Conformer search
                logger.debug("A conformer search was requested, so checking for files")
                # Get energy and geom of lowest conformer
                best = Geometry.load_file(
                    self.output_file.with_name("crest_best.xyz"),
                    charge=self.input_geometry.charge,
                    spin=self.input_geometry.spin,
                )
                logger.debug(
                    f"Geometry of lowest energy conformer read from {self.output_file.with_name('crest_best.xyz')}"
                )
                self.output_geometry = best
                self.energy = float(best._comment)
                logger.debug(f"Energy of lowest energy conformer: {self.energy}")
                # Get set of conformers
                conformer_geoms = Geometry.load_file(
                    self.output_file.with_name("crest_conformers.xyz"),
                    multi=True,
                    charge=self.input_geometry.charge,
                    spin=self.input_geometry.spin,
                )
                logger.debug(
                    f"Geometries of conformers read from {self.output_file.with_name('crest_conformers.xyz')}"
                )
                self.conformers = [
                    {"geometry": geom, "energy": float(geom._comment)} for geom in conformer_geoms
                ]
                logger.debug(f"Found {len(self.conformers)} conformers in the ensemble")
                # By convention the first in the file is the lowest but make sure anyway
                self.conformers = sorted(self.conformers, key=lambda x: x["energy"])
            case "tautomerize" | "protonate" | "deprotonate":
                # Protomer screening with +/- 1 or 0 protons
                logger.debug("A protomer screening was requested, so checking for files")
                match self.runtype:
                    case "tautomerize":
                        result_filename = "tautomers.xyz"
                        result_charge = self.input_geometry.charge
                    case "protonate":
                        result_filename = "protonated.xyz"
                        result_charge = self.input_geometry.charge + 1
                    case "deprotonate":
                        result_filename = "deprotonated.xyz"
                        result_charge = self.input_geometry.charge - 1
                # Get set of tautomers
                tautomer_geoms = Geometry.load_file(
                    self.output_file.with_name(result_filename),
                    multi=True,
                    charge=result_charge,
                    spin=self.input_geometry.spin,
                )
                logger.debug(
                    f"Geometries of tautomers read from {self.output_file.with_name(result_filename)}"
                )
                self.tautomers = [
                    {"geometry": geom, "energy": float(geom._comment)} for geom in tautomer_geoms
                ]
                logger.debug(f"Found {len(self.tautomers)} tautomers in the ensemble")
                # By convention the first in the file is the lowest but make sure anyway
                self.tautomers = sorted(self.tautomers, key=lambda x: x["energy"])
                # Get energy and geom of lowest tautomer
                best = self.tautomers[0]
                self.output_geometry = best["geometry"]
                self.energy = best["energy"]
                logger.debug(f"Energy of lowest energy tautomer: {self.energy}")
            case "qcg":
                # Explicit solvent shell growing
                logger.debug(
                    "Growing of a solvent shell was requested, so checking for generated cluster geometry"
                )
                # Get final cluster geom
                self.output_geometry = Geometry.load_file(
                    self.calc_dir / "grow/cluster.xyz",
                    charge=self.input_geometry.charge,
                    spin=self.input_geometry.spin,
                )
                logger.debug(f'Cluster geometry read from {self.calc_dir/"grow/cluster.xyz"}')
            case _:
                pass

    # Provide some convenient constructors for frequently used calculation types

    @classmethod
    def sp(
        cls,
        input_geometry: Geometry,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        molden: bool = False,
        options: dict | None = None,
    ):
        """Construct a Calculation for an `xtb` single-point energy calculation.
        
        If `molden` is `True`, a print-out of the molecular orbitals in Molden format
        will be requested.
        
        Parameters
        ----------
        input_geometry
        solvation
            Sets the implicit solvent via the `--alpb` command line option.
        method
            Sets the parameterization of GFN-xTB to use via the `--gfn` command line
            option.
        n_proc
            Indicates the number of parallel threads the program should use via the `-P`
            command line option.
        molden
            Whether to request a Molden-format MO print-out.
        options
            Further command line options (see `Calculation`).
        """

        options = options if options else {}
        return cls(
            program=XTB,
            input_geometry=input_geometry,
            options={
                "gfn": method if method else config["method"],
                "alpb": solvation if solvation else config["solvent"],
                "P": n_proc if n_proc else config["n_proc"],
                "molden": molden,
            }
            | options,
        )

    @classmethod
    def opt(
        cls,
        input_geometry: Geometry,
        level: str | None = None,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Construct a Calculation for an `xtb` geometry optimization.
        
        Corresponds to the `--opt` runtype.

        Parameters
        ----------
        input_geometry
        level
            Sets the optimization level.
        solvation
            Sets the implicit solvent via the `--alpb` command line option.
        method
            Sets the parameterization of GFN-xTB to use via the `--gfn` command line
            option.
        n_proc
            Indicates the number of parallel threads the program should use via the `-P`
            command line option.
        options
            Further command line options (see `Calculation`).
        """

        options = options if options else {}
        return cls(
            program=XTB,
            input_geometry=input_geometry,
            runtype="opt",
            runtype_args=[level] if level else [config["opt_lvl"]],
            options={
                "gfn": method if method else config["method"],
                "alpb": solvation if solvation else config["solvent"],
                "P": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )

    @classmethod
    def hess(
        cls,
        input_geometry: Geometry,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Construct a Calculation for an `xtb` vibrational frequency calculation.
        
        Corresponds to the `--hess` runtype.
        """

        options = options if options else {}
        return cls(
            program=XTB,
            input_geometry=input_geometry,
            runtype="hess",
            options={
                "gfn": method if method else config["method"],
                "alpb": solvation if solvation else config["solvent"],
                "P": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )

    @classmethod
    def ohess(
        cls,
        input_geometry: Geometry,
        level: str | None = None,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Construct a Calculation for a combined geometry optimization and frequency
        calculation with `xtb`.

        Corresponds to the `--ohess` runtype.
        """

        options = options if options else {}
        options = {
            "gfn": method if method else config["method"],
            "alpb": solvation if solvation else config["solvent"],
            "P": n_proc if n_proc else config["n_proc"],
        } | options
        return cls(
            program=XTB,
            input_geometry=input_geometry,
            runtype="ohess",
            runtype_args=[level] if level else [config["opt_lvl"]],
            options=options,
        )

    @classmethod
    def v3(
        cls,
        input_geometry: Geometry,
        solvation: str | None = None,
        method: int | None = None,
        ewin: int | float = 6,
        hess: bool = False,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Construct a Calculation for a conformer ensemble generation with `crest`.

        Corresponds to the default `--v3` runtype.

        All conformers within `ewin` kcal/mol are kept.
        If hess=True, vibrational frequencies are calculated and the conformers reordered by Gibbs energy.
        """
        method_flag = f"gfn{method}" if method else f'gfn{config["method"]}'
        options = options if options else {}
        return cls(
            program=CREST,
            input_geometry=input_geometry,
            runtype="v3",
            options={
                "xnam": XTB.path,
                method_flag: True,
                "alpb": solvation if solvation else config["solvent"],
                "ewin": ewin,
                "prop": "hess" if hess else False,
                "T": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )

    @classmethod
    def tautomerize(
        cls,
        input_geometry: Geometry,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Sample prototropic tautomers and return set of tautomer Geometries and energies.

        The returned tautomers are ordered from lowest to highest energy.
        """
        method_flag = f"gfn{method}" if method else f'gfn{config["method"]}'
        options = options if options else {}
        return cls(
            program=CREST,
            input_geometry=input_geometry,
            runtype="tautomerize",
            options={
                "xnam": XTB.path,
                method_flag: True,
                "alpb": solvation if solvation else config["solvent"],
                "T": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )

    @classmethod
    def protonate(
        cls,
        input_geometry: Geometry,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Screen possible protonation sites and return set of tautomer Geometries and energies.

        The returned tautomers are ordered from lowest to highest energy.
        """
        method_flag = f"gfn{method}" if method else f'gfn{config["method"]}'
        options = options if options else {}
        return cls(
            program=CREST,
            input_geometry=input_geometry,
            runtype="protonate",
            options={
                "xnam": XTB.path,
                method_flag: True,
                "alpb": solvation if solvation else config["solvent"],
                "T": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )

    @classmethod
    def deprotonate(
        cls,
        input_geometry: Geometry,
        solvation: str | None = None,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Screen possible deprotonation sites and return set of tautomer Geometries and energies.

        The returned tautomers are ordered from lowest to highest energy.
        """
        method_flag = f"gfn{method}" if method else f'gfn{config["method"]}'
        options = options if options else {}
        return cls(
            program=CREST,
            input_geometry=input_geometry,
            runtype="deprotonate",
            options={
                "xnam": XTB.path,
                method_flag: True,
                "alpb": solvation if solvation else config["solvent"],
                "T": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )

    @classmethod
    def qcg(
        cls,
        solute_geometry: Geometry,
        solvent_geometry: Geometry,
        nsolv: int,
        method: int | None = None,
        n_proc: int | None = None,
        options: dict | None = None,
    ):
        """Grow a solvent shell around a solute for a total of `nsolv` solvent molecules.

        Note that non-zero charge and spin on the solvent `Geometry` will not be passed to
        CREST.
        """
        method_flag = f"gfn{method}" if method else f'gfn{config["method"]}'
        options = options if options else {}
        return cls(
            program=CREST,
            input_geometry=solute_geometry,
            runtype="qcg",
            runtype_args=[solvent_geometry],
            options={
                "grow": True,
                "nsolv": nsolv,
                "xnam": XTB.path,
                method_flag: True,
                "T": n_proc if n_proc else config["n_proc"],
            }
            | options,
        )
