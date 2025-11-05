#!/usr/bin/env python3
import os
from fargo2radmc3d.core import par
from fargo2radmc3d.core import radmc_inputs as R
from fargo2radmc3d.dust import dust_opacities as DOP
from fargo2radmc3d.dust import dust_density as DD

if (par.recalc_dust_density == 'Yes') or (not (os.path.exists('dust_density.binp') or os.path.exists('dust_density.inp'))):
    DD.compute_dust_mass_volume_density()
DOP.ensure_opacity_files(folder='dust')
DOP.write_dustopac(par.species, par.nbin)
R.write_wavelength()
R.write_stars(Rstar=par.rstar, Tstar=par.teff)
R.write_AMRgrid(par.gas, Plot=False)
R.write_radmc3dinp(incl_dust=1, incl_lines=0, lines_mode=par.lines_mode,
                   nphot=par.nb_photons, nphot_scat=par.nb_photons_scat,
                   tgas_eq_tdust=0, modified_random_walk=1,
                   scattering_mode_max=par.scat_mode, setthreads=par.nbcores, rto_style=3)
R.write_mctherm_script()
