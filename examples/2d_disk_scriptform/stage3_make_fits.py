#!/usr/bin/env python3
from fargo2radmc3d.core import par
from fargo2radmc3d.imaging import radmc_to_fits as FIT
from fargo2radmc3d.imaging import final_image as IMG

FIT.exportfits()
IMG.produce_final_image('')
if (par.RTdust_or_gas == 'both') and (par.subtract_continuum == 'Yes'):
    par.RTdust_or_gas = 'dust'
    IMG.produce_final_image('dust')
print('Stage 3 ready. FITS files produced.')
