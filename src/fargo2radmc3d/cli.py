# src/fargo2radmc3d/cli.py
from __future__ import annotations
import argparse
import sys
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

def _eprint(*a, **k):  # tiny helper for stderr prints
    print(*a, file=sys.stderr, **k)

def _run(cmd: list[str], cwd: str | Path | None = None):
    _eprint("→", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def _legacy_run(workdir: Path) -> None:
    os.chdir(workdir)
    # Import modules after chdir so they read params.dat in the workdir
    from fargo2radmc3d.core import par
    from fargo2radmc3d.core import radmc_inputs
    from fargo2radmc3d.dust import dust_opacities, dust_density, dust_temperature
    from fargo2radmc3d.gas import gas_density, gas_temperature, gas_velocity
    from fargo2radmc3d.chemistry import pinte_co
    from fargo2radmc3d.imaging import radmc_to_fits, final_image

    # Determine gas/dust coupling flag for radmc3d.inp
    tgas_eq_tdust = 0
    if par.RTdust_or_gas == 'both':
        if (par.Tdust_eq_Thydro == 'Yes') or (par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'Yes'):
            tgas_eq_tdust = 1

    # --- Dust branch ---
    if par.RTdust_or_gas == 'dust':
        if par.Tdust_eq_Thydro == 'Yes':
            if par.recalc_dust_temperature == 'Yes':
                gas_temperature.compute_hydro_temperature()
            if par.plot_dust_quantities == 'Yes':
                gas_temperature.plot_gas_temperature()
        if par.recalc_dust_density == 'Yes':
            dust_density.compute_dust_mass_volume_density()
        if par.plot_dust_quantities == 'Yes':
            if par.Tdust_eq_Thydro == 'No' and par.dustsublimation == 'Yes':
                dust_density.plot_dust_density('before')
            else:
                dust_density.plot_dust_density('')
        if par.recalc_opac == 'Yes':
            dust_opacities.compute_dust_opacities()
        if par.plot_dust_quantities == 'Yes':
            dust_opacities.plot_opacities(species=par.species, amin=par.amin, amax=par.amax, nbin=par.nbin, lbda1=par.wavelength*1e3)
        dust_opacities.write_dustopac(par.species, par.nbin)

    # --- Gas branch ---
    if par.RTdust_or_gas == 'gas':
        if (par.recalc_gas_quantities == 'Yes') or (par.plot_gas_quantities == 'Yes'):
            if (par.Tdust_eq_Thydro == 'Yes') or (par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No'):
                gas_temperature.compute_hydro_temperature()
                if par.plot_gas_quantities == 'Yes':
                    gas_temperature.plot_gas_temperature()
            gas_density.compute_gas_mass_volume_density()
            gas_velocity.write_gas_microturb()
            gas_velocity.compute_gas_velocity()

    # --- Both branch ---
    if par.RTdust_or_gas == 'both':
        if (par.recalc_gas_quantities == 'Yes') or (par.plot_gas_quantities == 'Yes'):
            if (par.Tdust_eq_Thydro == 'Yes') or (par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No'):
                gas_temperature.compute_hydro_temperature()
                if par.plot_gas_quantities == 'Yes':
                    gas_temperature.plot_gas_temperature()
            gas_density.compute_gas_mass_volume_density()
            gas_velocity.write_gas_microturb()
            gas_velocity.compute_gas_velocity()
        if par.recalc_dust_density == 'Yes':
            dust_density.compute_dust_mass_volume_density()
        if par.plot_dust_quantities == 'Yes':
            if par.Tdust_eq_Thydro == 'No' and par.dustsublimation == 'Yes':
                dust_density.plot_dust_density('before')
            else:
                dust_density.plot_dust_density('')
                dust_density.plot_dust_to_gas_density()
        if par.recalc_opac == 'Yes':
            dust_opacities.compute_dust_opacities()
        if par.plot_dust_quantities == 'Yes':
            dust_opacities.plot_opacities(species=par.species, amin=par.amin, amax=par.amax, nbin=par.nbin, lbda1=par.wavelength*1e3)
        dust_opacities.write_dustopac(par.species, par.nbin)

    # Write script and optionally run RADMC-3D
    radmc_inputs.write_radmc3d_script()

    if par.recalc_radmc == 'Yes':
        radmc_inputs.write_wavelength()
        radmc_inputs.write_stars(Rstar=par.rstar, Tstar=par.teff)
        radmc_inputs.write_AMRgrid(par.gas, Plot=False)
        if par.RTdust_or_gas in ('gas', 'both'):
            radmc_inputs.write_lines(str(par.gasspecies), par.lines_mode)
        radmc_inputs.write_radmc3dinp(
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

        # Optional: run mcmono for mean intensity at working wavelength
        if str(getattr(par, 'write_meanint', 'No')).lower() == 'yes':
            try:
                radmc_inputs.write_mcmono_wavelengths([float(par.wavelength)*1e3])
                _run(['radmc3d', 'mcmono'], cwd=workdir)
            except Exception as e:
                _eprint(f"[mcmono] single-wavelength run skipped: {e}")

        # Optional: UV mean intensity for CO photodissociation chemistry, then run chemistry
        if str(getattr(par, 'photodissociation', 'No')).lower() in ('pinte2018', 'pinte'):
            try:
                lam = np.linspace(0.0912, 0.205, 100)
                radmc_inputs.write_mcmono_wavelengths(lam)
                _run(['radmc3d', 'mcmono'], cwd=workdir)
                pinte_co.run(model_dir='.', mol=str(par.gasspecies), use_dust_temperature=(par.Tdust_eq_Thydro == 'No'))
            except Exception as e:
                _eprint(f"[chemistry] Pinte2018 step skipped: {e}")

        if (par.RTdust_or_gas in ('dust', 'both')) and par.Tdust_eq_Thydro == 'No' and par.recalc_dust_temperature == 'Yes':
            _run(['radmc3d', 'mctherm'], cwd=workdir)
            if par.plot_dust_quantities == 'Yes':
                dust_temperature.plot_dust_temperature('' if par.dustsublimation == 'No' else 'before')
            if par.dustsublimation == 'Yes':
                dust_density.recompute_dust_mass_volume_density()
                if par.plot_dust_quantities == 'Yes':
                    dust_density.plot_dust_density('after')
                _run(['radmc3d', 'mctherm'], cwd=workdir)
                if par.plot_dust_quantities == 'Yes':
                    dust_temperature.plot_dust_temperature('after')
            if par.freezeout == 'Yes':
                gas_density.recompute_gas_mass_volume_density()

        _run(['./script_radmc'], cwd=workdir)

    # Post-processing
    if par.recalc_rawfits == 'Yes':
        radmc_to_fits.exportfits()
    if par.recalc_fluxmap == 'Yes':
        final_image.produce_final_image('')
        if (par.RTdust_or_gas == 'both') and (par.subtract_continuum == 'Yes'):
            par.RTdust_or_gas = 'dust'
            final_image.produce_final_image('dust')


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

