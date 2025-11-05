#!/usr/bin/env python3
import os
from fargo2radmc3d.core import par
from fargo2radmc3d.core import radmc_inputs as R
from fargo2radmc3d.gas import gas_density as GD
from fargo2radmc3d.gas import gas_velocity as GV

GD.compute_gas_mass_volume_density()
GV.write_gas_microturb()
GV.compute_gas_velocity()
R.write_wavelength()
R.write_stars(Rstar=par.rstar, Tstar=par.teff)
R.write_AMRgrid(par.gas, Plot=False)
R.write_lines(str(par.gasspecies), par.lines_mode)
R.write_radmc3dinp(incl_dust=par.incl_dust, incl_lines=par.incl_lines,
                   lines_mode=par.lines_mode, nphot=par.nb_photons,
                   nphot_scat=par.nb_photons_scat, tgas_eq_tdust=1,
                   modified_random_walk=1, scattering_mode_max=par.scat_mode,
                   setthreads=par.nbcores, rto_style=3)
R.write_radmc3d_script()
os.rename('script_radmc','image_lines.sh')
print('Stage 2 ready: ./image_lines.sh')
