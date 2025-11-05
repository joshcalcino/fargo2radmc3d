# import global variables
from . import par

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# Physical constants (cgs)
SIGMA_SB = 5.670374419e-5  # Stefan–Boltzmann constant [erg cm^-2 s^-1 K^-4]


# -------------------------
# script calling RADMC3D
# -------------------------
def write_radmc3d_script():
    
    # RT in the dust continuum
    if par.RTdust_or_gas == 'dust':
        command ='radmc3d image lambda '+str(par.wavelength*1e3)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
        if par.plot_tau == 'Yes':
            command ='radmc3d image tracetau lambda '+str(par.wavelength*1e3)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
        if par.polarized_scat == 'Yes':
            command=command+' stokes'

    # RT in gas lines
    if par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both':
        if par.widthkms == 0.0:
            command='radmc3d image iline '+str(par.iline)+' vkms '+str(par.vkms)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
        else:
            command='radmc3d image iline '+str(par.iline)+' widthkms '+str(par.widthkms)+' linenlam '+str(par.linenlam)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
            if par.plot_tau == 'Yes':
                command='radmc3d image tracetau iline '+str(par.iline)+' widthkms '+str(par.widthkms)+' linenlam '+str(par.linenlam)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
                #command='radmc3d tausurf 1.0 iline '+str(iline)+' widthkms '+str(widthkms)+' linenlam '+str(linenlam)+' npix '+str(nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)

    # optional: second-order ray tracing
    if par.secondorder == 'Yes':
        command=command+' secondorder'

    # write execution script
    if par.verbose == 'Yes':
        print(command)
    SCRIPT = open('script_radmc','w')
    SCRIPT.write('#!/usr/bin/env bash\n')
    '''
    if par.Tdust_eq_Thydro == 'No':
        SCRIPT.write('radmc3d mctherm; '+command)
    else:
        SCRIPT.write(command)        
    '''
    SCRIPT.write(command + '\n')
    SCRIPT.close()
    os.system('chmod a+x script_radmc')


def write_mctherm_script():
    """Write a simple mctherm runner script (mctherm.sh) and chmod +x.
    If write_meanint == Yes, also prepare mcmono wavelengths at the working
    wavelength and append a mcmono call to the script.
    """
    if par.verbose == 'Yes':
        print('writing mctherm.sh')
    with open('mctherm.sh', 'w') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write('radmc3d mctherm\n')
        if str(getattr(par, 'write_meanint', 'No')).lower() == 'yes':
            try:
                # working wavelength is in mm in params; convert to micron for mcmono
                write_mcmono_wavelengths([float(par.wavelength) * 1e3])
                f.write('radmc3d mcmono\n')
            except Exception:
                pass
        # automatic UV mcmono for Pinte photodissociation chemistry
        try:
            phot = str(getattr(par, 'photodissociation', 'No')).lower()
            if phot in ('pinte2018', 'pinte'):
                lam = np.linspace(0.0912, 0.205, 100)
                write_mcmono_wavelengths(lam)
                f.write('radmc3d mcmono\n')
        except Exception:
            pass
    os.chmod('mctherm.sh', 0o755)

# ---------------------------------------
# write spatial grid in file amr_grid.inp
# ---------------------------------------
def write_AMRgrid(F, R_Scaling=1, Plot=False):

    if par.verbose == 'Yes':
        print("writing spatial grid")
    path_grid='amr_grid.inp'

    grid=open(path_grid,'w')

    grid.write('1 \n')              # iformat/ format number = 1
    grid.write('0 \n')              # Grid style (regular = 0)
    grid.write('101 \n')            # coordsystem: 100 < spherical < 200 
    grid.write('0 \n')              # gridinfo
    grid.write('1 \t 1 \t 1 \n')    # incl x, incl y, incl z

    # spherical radius, colatitude, azimuth
    grid.write(str(F.nrad)+ '\t'+ str(F.ncol)+'\t'+ str(F.nsec)+'\n') 

    # nrad+1 dimension as we need to enter the coordinates of the cells edges
    for i in range(F.nrad + 1):  
        grid.write(str(F.redge[i]*F.culength*1e2)+'\t') # with unit conversion in cm
    grid.write('\n')

    # colatitude
    for i in range(F.ncol + 1):
        grid.write(str(F.tedge[i])+'\t')
    grid.write('\n')

    # azimuth
    for i in range(F.nsec + 1):
        grid.write(str(F.pedge[i])+'\t')
    grid.write('\n')
    
    grid.close()


# -----------------------
# writing out wavelength 
# -----------------------
def _build_wavelength_grid(wmin=0.1, wmax=10000.0, Nw=150):
    Pw = (wmax/wmin)**(1.0/(Nw-1))
    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in range(1, Nw):
        waves[i] = wmin * Pw**i
    return waves


def write_wavelength(wmin=0.1, wmax=10000.0):
    Nw = 150
    waves = _build_wavelength_grid(wmin=wmin, wmax=wmax, Nw=Nw)

    if par.verbose == 'Yes':
        print('writing wavelength_micron.inp')

    path = 'wavelength_micron.inp'
    wave = open(path,'w')
    wave.write(str(Nw)+'\n')
    for i in range(Nw):
        wave.write(str(waves[i])+'\n')
    wave.close()

# -----------------------------------------
# write mcmono_wavelength_micron.inp (single helper)
# -----------------------------------------
def write_mcmono_wavelengths(waves_um):
    """Write RADMC-3D mcmono_wavelength_micron.inp from an iterable of wavelengths in micron."""
    arr = np.array(list(waves_um), dtype=float)
    with open('mcmono_wavelength_micron.inp', 'w') as f:
        f.write(f"{arr.size}\n")
        for w in arr:
            f.write(f"{w}\n")


def _B_lambda(lam_cm: np.ndarray, T: float) -> np.ndarray:
    """Planck function B_lambda [erg s^-1 cm^-2 sr^-1 cm^-1]."""
    h = par.h
    c = par.c
    kB = par.kB
    x = (h*c)/(lam_cm*kB*T)
    x = np.clip(x, 1e-10, 1e3)
    return (2.0*h*c*c)/(lam_cm**5) / (np.expm1(x))


def _source_scale(R_cm: float) -> float:
    """Geometric scale factor pi R^2 / d^2 using current par.distance."""
    d_cm = float(par.distance) * par.pc
    return np.pi * (R_cm*R_cm) / (d_cm*d_cm)


def compute_stellar_accretion_sed(lam_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute F_lambda for star, accretion, and total on lam_um using the same
    source assembly as write_stars(). Returns (F_star_um, F_acc_um, F_tot_um)."""
    lam_cm = lam_um * 1e-4
    stars = _assemble_star_sources(par.rstar, par.teff)
    F_star_cm = np.zeros_like(lam_cm)
    F_acc_cm = np.zeros_like(lam_cm)
    for s in stars:
        scale = _source_scale(s['R_cm'])
        F_cm = scale * _B_lambda(lam_cm, s['T_K'])
        if s.get('kind') == 'accretion':
            F_acc_cm += F_cm
        else:
            F_star_cm += F_cm
    F_star_um = F_star_cm * 1e-4
    F_acc_um = F_acc_cm * 1e-4
    F_tot_um = F_star_um + F_acc_um
    return F_star_um, F_acc_um, F_tot_um


def read_wavelength_grid_or_default() -> np.ndarray:
    """Read wavelength_micron.inp if present; else build the default grid.
    Returns wavelengths in micron.
    """
    if os.path.isfile('wavelength_micron.inp'):
        with open('wavelength_micron.inp', 'r') as f:
            N = int(f.readline().strip())
            return np.array([float(f.readline().strip()) for _ in range(N)])
    return _build_wavelength_grid()


def plot_stellar_accretion_sed():
    try:
        # 1) Wavelength grid in micron (shared reader)
        lam_um = read_wavelength_grid_or_default()

        # 2) Compute SED using shared function
        F_star_um, F_acc_um, F_tot_um = compute_stellar_accretion_sed(lam_um)

        # 5) Plot
        matplotlib.rcParams.update({'font.size': 14})
        plt.figure(figsize=(7.5,5.5))
        plt.loglog(lam_um, F_star_um, label='Star')
        if np.any(F_acc_um > 0):
            plt.loglog(lam_um, F_acc_um, label='Accretion')
        plt.loglog(lam_um, F_tot_um, label='Total', linestyle='--')
        plt.xlabel('Wavelength [μm]')
        plt.ylabel('Flux density [erg s$^{-1}$ cm$^{-2}$ μm$^{-1}$]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('stellar_accretion_sed.png', dpi=160)
        plt.close()
    except Exception as _e:
        pass


def compute_stellar_accretion_luminosities() -> dict:
    """Compute and return stellar/accretion luminosities using shared sources.
    Returns dict with keys: T_star, L_star, T_acc (optional), L_acc, L_tot [cgs].
    """
    stars = _assemble_star_sources(par.rstar, par.teff)
    L_star = 0.0
    L_acc = 0.0
    T_star = None
    T_acc = None
    for s in stars:
        L = 4.0 * np.pi * SIGMA_SB * s['R_cm']*s['R_cm'] * (s['T_K']**4)
        if s.get('kind') == 'accretion':
            L_acc += L
            T_acc = s['T_K']
        else:
            L_star += L
            T_star = s['T_K']
    return {
        'T_star': T_star,
        'L_star': L_star,
        'T_acc': T_acc,
        'L_acc': L_acc,
        'L_tot': L_star + L_acc,
    }


def print_stellar_accretion_diagnostics() -> None:
    try:
        d = compute_stellar_accretion_luminosities()
        print(f"[SED] Star:    T={d['T_star']:.2f} K   L={d['L_star']:.3e} erg/s")
        if d['T_acc'] is not None and d['L_acc'] > 0.0:
            print(f"[SED] Accr.:   T={d['T_acc']:.2f} K   L={d['L_acc']:.3e} erg/s")
        print(f"[SED] Total:              L={d['L_tot']:.3e} erg/s")
    except Exception:
        pass


def _assemble_star_sources(Rstar, Tstar):
    stars = []
    r_cm = Rstar * par.R_Sun
    mstar_g = getattr(par, 'mstar', 1.0) * par.M_Sun
    stars.append({'R_cm': r_cm, 'M_g': mstar_g, 'T_K': float(Tstar), 'kind': 'star'})
    mdot_msun_per_yr = float(getattr(par, 'mdot', 0.0))
    include_flag = str(getattr(par, 'include_accretion_lum', 'No')).lower() == 'yes'
    if mdot_msun_per_yr > 0.0 or include_flag:
        if mdot_msun_per_yr > 0.0:
            # Accretion filling factor (fraction of stellar surface emitting)
            f_fill = float(getattr(par, 'accretion_fill_factor', getattr(par, 'f', 0.01)))
            f_fill = max(min(f_fill, 1.0), 1e-6)
            mdot_g_per_s = mdot_msun_per_yr * par.M_Sun / (365.25*24.0*3600.0)
            Lacc = par.G * mstar_g * mdot_g_per_s / r_cm
            # Hotspot radius such that area = f * 4 pi R_*^2
            r_acc = (f_fill**0.5) * r_cm
            # Temperature over hotspot area so that sigma T^4 * area = Lacc
            Tacc = (Lacc / (4.0*np.pi*SIGMA_SB*r_acc*r_acc))**0.25
            if Tacc > 0.0:
                stars.append({'R_cm': r_acc, 'M_g': mstar_g, 'T_K': float(Tacc), 'kind': 'accretion'})
    return stars


def write_stars(Rstar = 1, Tstar = 6000, wmin=0.1, wmax=10000.0, Nw = 150):
    waves = _build_wavelength_grid(wmin=wmin, wmax=wmax, Nw=Nw)

    if par.verbose == 'Yes':
        print('writing stars.inp')

    path = 'stars.inp'
    wave = open(path,'w')

    stars = _assemble_star_sources(Rstar, Tstar)

    # Write header and star descriptors
    wave.write('\t 2\n')
    wave.write(f"{len(stars)} \t{Nw}\n")
    for s in stars:
        wave.write(f"{s['R_cm']}\t{s['M_g']}\t 0 \t 0 \t 0 \n")
    for i in range(Nw):
        wave.write('\t'+str(waves[i])+'\n')
    # For each star, write a negative temperature to indicate blackbody emitter
    for s in stars:
        wave.write('\t -'+str(s['T_K'])+'\n')
    wave.close()


# --------------------
# writing radmc3d.inp
# --------------------
def write_radmc3dinp(incl_dust = 1,
                     incl_lines = 0,
                     lines_mode = 1,
                     nphot = 1000000,
                     nphot_scat = 1000000,
                     nphot_spec = 1000000,
                     nphot_mono = 1000000,
                     istar_sphere = 0,
                     scattering_mode_max = 0,
                     tgas_eq_tdust = 1,
                     modified_random_walk = 0,
                     itempdecoup=1,
                     setthreads=2,
                     rto_style=3 ):

    if par.verbose == 'Yes':
        print('writing radmc3d.inp')

    RADMCINP = open('radmc3d.inp','w')
    inplines = ["incl_dust = "+str(int(incl_dust))+"\n",
                "incl_lines = "+str(int(incl_lines))+"\n",
                "lines_mode = "+str(int(lines_mode))+"\n",
                "nphot = "+str(int(nphot))+"\n",
                "nphot_scat = "+str(int(nphot_scat))+"\n",
                "nphot_spec = "+str(int(nphot_spec))+"\n",
                "nphot_mono = "+str(int(nphot_mono))+"\n",
                "istar_sphere = "+str(int(istar_sphere))+"\n",
                "scattering_mode_max = "+str(int(scattering_mode_max))+"\n",
                "tgas_eq_tdust = "+str(int(tgas_eq_tdust))+"\n",
                "modified_random_walk = "+str(int(modified_random_walk))+"\n",
                "itempdecoup = "+str(int(itempdecoup))+"\n",
                "setthreads="+str(int(setthreads))+"\n",
                "rto_style="+str(int(rto_style))+"\n"]

    RADMCINP.writelines(inplines)
    RADMCINP.close()

    
# --------------------
# writing lines.inp
# --------------------
def write_lines(specie,lines_mode):

    if par.verbose == 'Yes':
        print("writing lines.inp")
    path_lines='lines.inp'

    lines=open(path_lines,'w')

    lines.write('2 \n')              # <=== Put this to 2
    lines.write('1 \n')              # Nr of molecular or atomic species to be modeled
    # LTE calculations
    if lines_mode == 1:
        lines.write('%s    leiden    0    0    0'%specie)    # incl x, incl y, incl z
    else:
    # non-LTE calculations
        lines.write('%s    leiden    0    0    1\n'%specie)    # incl x, incl y, incl z
        lines.write('h2')
    lines.close()

    # Get molecular data file
    molecular_file = 'molecule_'+str(par.gasspecies)+'.inp'

    datafile = str(par.gasspecies)
    if par.gasspecies == 'hco+':
        datafile = 'hco+@xpol'
    if par.gasspecies == 'so':
        datafile = 'so@lique'
    if par.gasspecies == 'cs':
        datafile = 'cs'
    dat_file = datafile+'.dat'

    if (os.path.isfile(molecular_file) == False) and (os.path.isfile(dat_file) == False):

        # ---
        # check if curl is installed
        from shutil import which
        if which('curl') is None:
            sys.exit('curl is not installed on your system! I cannot download the molecular data file. Please install curl and restart!')
        # ---
    
        if par.verbose == 'Yes':
            print('--------- Downloading molecular data file ----------')
            
        command = 'curl -k -O https://home.strw.leidenuniv.nl/~moldata/datafiles/'+dat_file
        print(command)
        os.system(command)
        command = 'mv '+datafile+'.dat molecule_'+str(par.gasspecies)+'.inp'
        os.system(command)
