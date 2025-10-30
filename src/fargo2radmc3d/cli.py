# src/fargo2radmc3d/cli.py
from __future__ import annotations
import argparse
import sys
import os
import shutil
import subprocess
from pathlib import Path
import sys

def _import_legacy_modules():
    from fargo2radmc3d.core import par as par_mod
    from fargo2radmc3d.core import radmc_inputs as radmc_inputs_mod

    from fargo2radmc3d.dust import dust_opacities as dust_opacities_mod
    from fargo2radmc3d.dust import dust_density as dust_density_mod
    from fargo2radmc3d.dust import dust_temperature as dust_temperature_mod

    from fargo2radmc3d.gas import gas_density as gas_density_mod
    from fargo2radmc3d.gas import gas_temperature as gas_temperature_mod
    from fargo2radmc3d.gas import gas_velocity as gas_velocity_mod

    from fargo2radmc3d.imaging import radmc_to_fits as radmc_to_fits_mod
    from fargo2radmc3d.imaging import final_image as final_image_mod

    return {
        'par': par_mod,
        'radmc_inputs': radmc_inputs_mod,
        'dust_opacities': dust_opacities_mod,
        'dust_density': dust_density_mod,
        'dust_temperature': dust_temperature_mod,
        'gas_density': gas_density_mod,
        'gas_temperature': gas_temperature_mod,
        'gas_velocity': gas_velocity_mod,
        'radmc_to_fits': radmc_to_fits_mod,
        'final_image': final_image_mod,
    }

def _eprint(*a, **k):  # tiny helper for stderr prints
    print(*a, file=sys.stderr, **k)

def _run(cmd: list[str], cwd: str | Path | None = None):
    _eprint("→", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def _legacy_run(workdir: Path) -> None:
    os.chdir(workdir)

    mods = _import_legacy_modules()
    par = mods['par']
    R = mods['radmc_inputs']
    DOP = mods['dust_opacities']
    DD = mods['dust_density']
    DT = mods['dust_temperature']
    GD = mods['gas_density']
    GT = mods['gas_temperature']
    GV = mods['gas_velocity']
    FIT = mods['radmc_to_fits']
    IMG = mods['final_image']

    # Determine gas/dust coupling flag for radmc3d.inp
    tgas_eq_tdust = 0
    if par.RTdust_or_gas == 'both':
        if (par.Tdust_eq_Thydro == 'Yes') or (par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'Yes'):
            tgas_eq_tdust = 1

    # --- Dust branch ---
    if par.RTdust_or_gas == 'dust':
        if par.Tdust_eq_Thydro == 'Yes':
            if par.recalc_dust_temperature == 'Yes':
                GT.compute_hydro_temperature()
            if par.plot_dust_quantities == 'Yes':
                GT.plot_gas_temperature()
        if par.recalc_dust_density == 'Yes':
            DD.compute_dust_mass_volume_density()
        if par.plot_dust_quantities == 'Yes':
            if par.Tdust_eq_Thydro == 'No' and par.dustsublimation == 'Yes':
                DD.plot_dust_density('before')
            else:
                DD.plot_dust_density('')
        if par.recalc_opac == 'Yes':
            DOP.compute_dust_opacities()
        if par.plot_dust_quantities == 'Yes':
            DOP.plot_opacities(species=par.species, amin=par.amin, amax=par.amax, nbin=par.nbin, lbda1=par.wavelength*1e3)
        DOP.write_dustopac(par.species, par.nbin)

    # --- Gas branch ---
    if par.RTdust_or_gas == 'gas':
        if (par.recalc_gas_quantities == 'Yes') or (par.plot_gas_quantities == 'Yes'):
            if (par.Tdust_eq_Thydro == 'Yes') or (par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No'):
                GT.compute_hydro_temperature()
                if par.plot_gas_quantities == 'Yes':
                    GT.plot_gas_temperature()
            GD.compute_gas_mass_volume_density()
            GV.write_gas_microturb()
            GV.compute_gas_velocity()

    # --- Both branch ---
    if par.RTdust_or_gas == 'both':
        if (par.recalc_gas_quantities == 'Yes') or (par.plot_gas_quantities == 'Yes'):
            if (par.Tdust_eq_Thydro == 'Yes') or (par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No'):
                GT.compute_hydro_temperature()
                if par.plot_gas_quantities == 'Yes':
                    GT.plot_gas_temperature()
            GD.compute_gas_mass_volume_density()
            GV.write_gas_microturb()
            GV.compute_gas_velocity()
        if par.recalc_dust_density == 'Yes':
            DD.compute_dust_mass_volume_density()
        if par.plot_dust_quantities == 'Yes':
            if par.Tdust_eq_Thydro == 'No' and par.dustsublimation == 'Yes':
                DD.plot_dust_density('before')
            else:
                DD.plot_dust_density('')
                DD.plot_dust_to_gas_density()
        if par.recalc_opac == 'Yes':
            DOP.compute_dust_opacities()
        if par.plot_dust_quantities == 'Yes':
            DOP.plot_opacities(species=par.species, amin=par.amin, amax=par.amax, nbin=par.nbin, lbda1=par.wavelength*1e3)
        DOP.write_dustopac(par.species, par.nbin)

    # Write script and optionally run RADMC-3D
    R.write_radmc3d_script()

    if par.recalc_radmc == 'Yes':
        R.write_wavelength()
        R.write_stars(Rstar=par.rstar, Tstar=par.teff)
        R.write_AMRgrid(par.gas, Plot=False)
        if par.RTdust_or_gas in ('gas', 'both'):
            R.write_lines(str(par.gasspecies), par.lines_mode)
        R.write_radmc3dinp(
            incl_dust=par.incl_dust,
            incl_lines=par.incl_lines,
            lines_mode=par.lines_mode,
            nphot=par.nb_photons,
            nphot_scat=par.nb_photons_scat,
            rto_style=3,
            tgas_eq_tdust=tgas_eq_tdust,
            modified_random_walk=1,
            scattering_mode_max=par.scat_mode,
            setthreads=par.nbcores,
        )

        if (par.RTdust_or_gas in ('dust', 'both')) and par.Tdust_eq_Thydro == 'No':
            _run(['radmc3d', 'mctherm'], cwd=workdir)
            if par.plot_dust_quantities == 'Yes':
                DT.plot_dust_temperature('' if par.dustsublimation == 'No' else 'before')
            if par.dustsublimation == 'Yes':
                DD.recompute_dust_mass_volume_density()
                if par.plot_dust_quantities == 'Yes':
                    DD.plot_dust_density('after')
                _run(['radmc3d', 'mctherm'], cwd=workdir)
                if par.plot_dust_quantities == 'Yes':
                    DT.plot_dust_temperature('after')
            if par.freezeout == 'Yes':
                GD.recompute_gas_mass_volume_density()

        _run(['./script_radmc'], cwd=workdir)

    # Post-processing
    if par.recalc_rawfits == 'Yes':
        FIT.exportfits()
    if par.recalc_fluxmap == 'Yes':
        IMG.produce_final_image('')
        if (par.RTdust_or_gas == 'both') and (par.subtract_continuum == 'Yes'):
            par.RTdust_or_gas = 'dust'
            IMG.produce_final_image('dust')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='fargo2radmc3d')
    sub = p.add_subparsers(dest='cmd', required=False)

    run = sub.add_parser('run', help='Run legacy pipeline in a model directory (expects params.dat).')
    run.add_argument('-C', '--chdir', type=Path, default=Path('.'), help='Working directory containing params.dat')
    run.set_defaults(func=lambda a: _legacy_run(a.chdir))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'cmd', None):
        # Default: run in current directory
        args = parser.parse_args(['run'])
        args.chdir = Path('.')
        args.func = lambda a: _legacy_run(a.chdir)
    args.func(args)
    return 0

