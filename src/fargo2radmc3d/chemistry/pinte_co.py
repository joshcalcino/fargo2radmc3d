#!/usr/bin/env python3
"""
Pinte+2018-like parametric CO chemistry for RADMC-3D models integrated in fargo2radmc3d.

Usage (from a RADMC-3D model directory containing params.dat):
  python -m fargo2radmc3d.chemistry.pinte_co --model . --mol co

It will try to read grid and parameters from fargo2radmc3d.core.par (params.dat),
then fetch temperature and a proxy for the UV field chi. It writes number density
files for the requested molecule using the Appendix B switches from Pinte et al. (2018):

- Photodissociation: set X_CO = 0 where log10(chi/nH) > -6
- Photodesorption escape: skip freeze-out where log10(chi/nH) > -7
- Freeze-out: where T < 21 K and not escaped, multiply X by eps = 8e-5

Outputs:
  - numberdens_<mol>.binp (binary, preferred by RADMC-3D)
  - numberdens_<mol>.inp  (ascii, also accepted by RADMC-3D)

Notes:
  - nH is per H nucleus; we derive it from either H2 number density or a gas mass density proxy.
  - chi is in Draine units. Exact computation from J_lambda is not wired by default
    because typical pipelines do not write mean_intensity_lambda.*; a fast proxy is used instead.
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
from ..core.plot_utils import rz_edges_au, plot_rz_scalar_field
from . import freezeout as FO

# Physical constants (cgs)
C_LIGHT = 2.99792458e10
M_H     = 1.6735575e-24
MU_H    = 1.4  # mass per H nucleus in units of m_H
SIGMA_SB = 5.670374419e-5

# Default Pinte+18 thresholds
T_FRZ_DEFAULT = 21.0
EPS_DEFAULT   = 8e-5
LOG_CHI_OVER_NH_PDISS = -6.0
LOG_CHI_OVER_NH_PDES  = -7.0

def _load_mean_intensity_lambda(model: Path):
    """Try common RADMC-3D names for wavelength-resolved mean intensity.
    Returns (lam_cm [nw], J_lambda [ncol,nrad,nsec,nw]) or (None, None).
    """
    # Candidates: adjust if your RADMC build uses different names
    for cand in (
        'mean_intensity_lambda.binp',
        'mean_intensity_lambda.out',
        'meanint_lambda.binp',
        'meanint_lambda.out',
    ):
        p = model / cand
        if not p.exists():
            continue
        if p.suffix == '.binp':
            with p.open('rb') as f:
                # Expect: int32 iformat, then 4 int32 dims, then wavelengths (float64), then data
                try:
                    hdr_i = np.fromfile(f, dtype='>i4', count=1)
                    if hdr_i.size == 0:
                        continue
                    _iformat = int(hdr_i[0])
                    dims = np.fromfile(f, dtype='>i4', count=4)
                    nx, ny, nz, nw = map(int, dims)
                    lam = np.fromfile(f, dtype='>f8', count=nw)  # micron
                    data = np.fromfile(f, dtype='>f8', count=nx*ny*nz*nw)
                except Exception:
                    continue
            try:
                J = data.reshape((nx, ny, nz, nw))
            except Exception:
                continue
            # Our repo convention: (ncol,nrad,nsec)
            return lam*1e-4, J  # convert to cm
        else:
            try:
                with p.open() as f:
                    head = f.readline().split()
                    nx, ny, nz, nw = map(int, head)
                    lam = np.fromfile(f, count=nw, sep='\n') * 1e-4
                    data = np.fromfile(f, count=nx*ny*nz*nw, sep='\n')
                J = data.reshape((nx, ny, nz, nw))
            except Exception:
                continue
            return lam, J
    return None, None

def _compute_chi_from_Jlambda(lam_cm: np.ndarray, Jlam: np.ndarray) -> np.ndarray:
    """Compute chi (Draine units) by integrating u_lambda = 4π J_lambda / c
    over 91.2–205 nm and scaling by a nominal Draine normalization.
    The absolute normalization cancels if you only use log10(chi/nH) thresholds.
    """
    # UV band limits in cm
    uv = (lam_cm >= 91.2e-7) & (lam_cm <= 205e-7)
    if not np.any(uv):
        raise ValueError('No UV wavelengths (91.2–205 nm) found in J_lambda file')
    lam = lam_cm[uv]
    J = Jlam[..., uv]
    ulam = 4.0*np.pi*J / C_LIGHT
    chi = np.trapz(ulam, lam, axis=-1)
    # Optional absolute scaling could be applied here; thresholds are relative
    return chi


def _read_numberdens_h2_binp(model: Path, nrad: int, ncol: int, nsec: int) -> np.ndarray | None:
    """Read numberdens_h2.binp if present, returning shape (ncol,nrad,nsec) in cm^-3.
    gas_density.compute_gas_mass_volume_density writes header [1,8,Ncells] then data
    with axes swapped to (nsec, ncol, nrad).
    """
    p = model / 'numberdens_h2.binp'
    if not p.exists():
        return None
    buf = np.fromfile(p, dtype='float64')
    if buf.size < 3:
        return None
    data = buf[3:]
    try:
        arr = data.reshape((nsec, ncol, nrad))  # nsec, ncol, nrad
    except Exception:
        return None
    arr = np.swapaxes(arr, 0, 1)  # ncol, nsec, nrad
    arr = np.swapaxes(arr, 1, 2)  # ncol, nrad, nsec
    return arr


def _read_dust_density_binp(model: Path, nrad: int, ncol: int, nsec: int, nbin: int) -> np.ndarray | None:
    """Read dust_density.binp and return total dust mass density [g/cm^3] with shape (ncol,nrad,nsec).
    dust_density.compute_dust_mass_volume_density writes header [1,8,Ncells, nbin] then data with
    axes swapped to (nbin, nsec, ncol, nrad).
    """
    p = model / 'dust_density.binp'
    if not p.exists():
        return None
    buf = np.fromfile(p, dtype='float64')
    if buf.size < 4:
        return None
    data = buf[4:]
    try:
        arr = data.reshape((nbin, nsec, ncol, nrad))
    except Exception:
        return None
    # Sum dust density over bins -> (nsec, ncol, nrad)
    total = np.sum(arr, axis=0)
    # Reorder to (ncol, nrad, nsec)
    total = np.swapaxes(total, 0, 1)  # ncol, nsec, nrad
    total = np.swapaxes(total, 1, 2)  # ncol, nrad, nsec
    return total


def _read_temperature(model: Path, par) -> np.ndarray:
    """Return temperature [K] with shape (ncol,nrad,nsec).
    Prefer gas_temperature.inp if present; else dust_temperature.bdat if present.
    - gas_temperature.inp ascii: format 1, then nx ny nz, then values.
    - dust_temperature.bdat binary: header of 4 numbers then values shaped (nbin, nsec, ncol, nrad).
      We take the largest-grain bin as proxy (consistent with other parts of this codebase).
    """
    gt = model / 'gas_temperature.inp'
    if gt.exists():
        with gt.open() as f:
            _iformat = int(f.readline().strip())
            nx, ny, nz = map(int, f.readline().split())
            dat = np.fromfile(f, sep='\n', count=nx*ny*nz)
        T = dat.reshape((nx, ny, nz))  # assume (ncol,nrad,nsec)
        return T

    dt = model / 'dust_temperature.bdat'
    if dt.exists():
        buf = np.fromfile(dt, dtype='float64')
        if buf.size < 4:
            raise RuntimeError('dust_temperature.bdat too small')
        data = buf[4:]
        try:
            Temp = data.reshape((par.nbin, par.gas.nsec, par.gas.ncol, par.gas.nrad))
        except Exception as e:
            raise RuntimeError(f'Could not reshape dust_temperature.bdat: {e}')
        # take largest bin (index nbin-1), then reorder to (ncol,nrad,nsec)
        T = Temp[par.nbin-1, :, :, :]  # (nsec,ncol,nrad)
        T = np.swapaxes(T, 0, 1)  # (ncol,nsec,nrad)
        T = np.swapaxes(T, 1, 2)  # (ncol,nrad,nsec)
        return T

    raise FileNotFoundError('Neither gas_temperature.inp nor dust_temperature.bdat found.')


def _prepare_fields(model: Path, chi0: float, kappa_uv: float):
    _cwd = os.getcwd()
    try:
        os.chdir(model)
        from fargo2radmc3d.core import par as _par
    finally:
        os.chdir(_cwd)
    par = _par
    T = _read_temperature(model, par)
    nh2 = _read_numberdens_h2_binp(model, par.gas.nrad, par.gas.ncol, par.gas.nsec)
    if nh2 is not None:
        nH = 2.0 * nh2
    else:
        rhod = _read_dust_density_binp(model, par.gas.nrad, par.gas.ncol, par.gas.nsec, par.nbin)
        if rhod is None:
            raise FileNotFoundError('Need numberdens_h2.binp or dust_density.binp to derive n_H for diagnostics')
        rhog = rhod / float(par.ratio)
        nH = rhog / (MU_H * M_H)
    lam_cm, Jlam = _load_mean_intensity_lambda(model)
    if (lam_cm is not None) and (Jlam is not None):
        try:
            chi = _compute_chi_from_Jlambda(lam_cm, Jlam)
        except Exception:
            rhog = nH * (MU_H * M_H)
            chi = _estimate_chi_vertical(rhog, par=par, chi0=chi0, kappa_uv=kappa_uv)
    else:
        rhog = nH * (MU_H * M_H)
        chi = _estimate_chi_vertical(rhog, par=par, chi0=chi0, kappa_uv=kappa_uv)
    return par, T, nH, chi




def _estimate_chi_vertical(rho_gas: np.ndarray, par, chi0: float = 1.0, kappa_uv: float = 1e3) -> np.ndarray:
    """UV proxy via vertical column integration to the nearest disk surface.
    chi ≈ chi0 * exp(-tau_uv), tau_uv = kappa_uv * min(Σ_top, Σ_bottom),
    where Σ are gas columns [g cm^-2] computed along z at fixed (r,phi).
    Inputs/outputs shape: (ncol, nrad, nsec).
    """
    # Δz at each (j,i): dz = r_mid * |cos(theta_{j+1}) - cos(theta_j)| * culength_cm
    rmid = par.gas.rmed.reshape((1, par.gas.nrad))  # (1,nrad)
    tedge = par.gas.tedge  # (ncol+1)
    dz2d = np.abs((np.cos(tedge[1:]) - np.cos(tedge[:-1])).reshape((par.gas.ncol, 1)) * rmid) * (par.gas.culength * 1e2)
    # Broadcast dz to 3D
    dz3d = np.repeat(dz2d[:, :, np.newaxis], par.gas.nsec, axis=2)
    # Columns to top (decreasing j) and bottom (increasing j)
    col_top = np.cumsum((rho_gas[::-1, :, :] * dz3d[::-1, :, :]), axis=0)[::-1, :, :]
    col_bot = np.cumsum((rho_gas * dz3d), axis=0)
    col = np.minimum(col_top, col_bot)  # g cm^-2
    tau = kappa_uv * col
    return chi0 * np.exp(-tau)


def _write_numberdens_ascii(mol: str, nco: np.ndarray, outdir: Path) -> Path:
    out = outdir / f'numberdens_{mol.lower()}.inp'
    ncol, nrad, nsec = nco.shape
    with out.open('w') as f:
        f.write('1\n')
        f.write(f'{ncol} {nrad} {nsec}\n')
        nco.ravel(order='C').tofile(f, sep='\n')
        f.write('\n')
    return out


def _write_numberdens_binp(mol: str, nco: np.ndarray, outdir: Path) -> Path:
    out = outdir / f'numberdens_{mol.lower()}.binp'
    ncol, nrad, nsec = nco.shape
    with out.open('wb') as f:
        hdr = np.array([1, 8, nrad*nsec*ncol], dtype=np.int64)
        hdr.tofile(f)
        # RADMC-3D expects ordering (nsec, ncol, nrad) for ascii writer elsewhere in this repo.
        arr = np.swapaxes(nco, 1, 2)  # (ncol,nsec,nrad)
        arr = np.swapaxes(arr, 0, 1)  # (nsec,ncol,nrad)
        arr.tofile(f)
    return out


def run(model_dir: str | Path,
        mol: str = 'co',
        X0: float = 5e-5,
        eps: float = EPS_DEFAULT,
        Tfrz: float = T_FRZ_DEFAULT,
        use_dust_temperature: bool = True,
        use_uv_proxy: bool = True,
        kappa_uv: float = 1e3,
        chi0_proxy: float = 1.0) -> dict:
    model = Path(model_dir)

    # Ensure par.py reads params.dat from the model directory
    _cwd = os.getcwd()
    try:
        os.chdir(model)
        from fargo2radmc3d.core import par as _par
    finally:
        os.chdir(_cwd)
    par = _par  # alias
    # Ensure shared freezeout module points to the same par (params.dat) as this run
    FO.par = par

    # 1) Temperature (K)
    T = _read_temperature(model, par)

    # 2) n_H per H nucleus: prefer H2 number density if available; else build from dust density and ratio
    nH = None
    nh2 = _read_numberdens_h2_binp(model, par.gas.nrad, par.gas.ncol, par.gas.nsec)
    if nh2 is not None:
        # if all H is in H2, n_H ≈ 2 * n(H2)
        nH = 2.0 * nh2
    else:
        # Derive gas density from dust density and global dust-to-gas ratio, then nH = rho/(mu_H m_H)
        rhod = _read_dust_density_binp(model, par.gas.nrad, par.gas.ncol, par.gas.nsec, par.nbin)
        if rhod is None:
            raise FileNotFoundError('Could not find numberdens_h2.binp or dust_density.binp to derive n_H.')
        if getattr(par, 'ratio', None) is None:
            raise RuntimeError('dust-to-gas mass ratio (ratio) is missing in params.dat')
        rhog = rhod / float(par.ratio)
        nH = rhog / (MU_H * M_H)

    # 3) UV field chi: attempt exact from mean_intensity_lambda.* if present; else proxy
    lam_cm, Jlam = _load_mean_intensity_lambda(model)
    if (lam_cm is not None) and (Jlam is not None):
        try:
            chi = _compute_chi_from_Jlambda(lam_cm, Jlam)
        except Exception:
            rhog = nH * (MU_H * M_H)
            chi = _estimate_chi_vertical(rhog, par=par, chi0=chi0_proxy, kappa_uv=kappa_uv)
    else:
        rhog = nH * (MU_H * M_H)
        chi = _estimate_chi_vertical(rhog, par=par, chi0=chi0_proxy, kappa_uv=kappa_uv)

    # 4) Apply Pinte+18 switches
    X = np.full_like(T, float(X0), dtype=float)

    # Photodissociation (kill CO)
    mask_pdiss = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDISS)
    X[mask_pdiss] = 0.0

    # Photodesorption escape mask (optional; default disabled)
    if str(getattr(par, 'photodesorption', 'No')).lower() == 'yes':
        mask_pdes = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDES)
    else:
        mask_pdes = np.zeros_like(T, dtype=bool)

    # 5) Number density of molecule before optional freeze-out
    nco = X * nH  # cm^-3, shape (ncol,nrad,nsec)

    # Apply freeze-out via shared routine (dust temperature), unless disabled in params.dat
    if str(getattr(par, 'freezeout', 'Yes')).lower() == 'yes':
        # Convert to (nsec,ncol,nrad) for freezeout helper
        nco_cube = np.swapaxes(np.swapaxes(nco, 1, 2), 0, 1)
        nco_before = nco_cube.copy()
        # Apply FO based on dust temperature
        nco_cube, _ = FO.apply_gas_freezeout(rhogascube=nco_cube, rhogascube_cyl=None,
                                             temp_source='dust', threshold_K=float(Tfrz), eps=float(eps), dust_bin=par.nbin-1)
        # Restore photodesorption-escape cells (skip FO where mask_pdes is true)
        mask_pdes_cube = np.swapaxes(np.swapaxes(mask_pdes, 1, 2), 0, 1)
        nco_cube[mask_pdes_cube] = nco_before[mask_pdes_cube]
        # Convert back to (ncol,nrad,nsec)
        nco = np.swapaxes(np.swapaxes(nco_cube, 0, 1), 1, 2)

    # 6) Write outputs (binary only to avoid RADMC-3D ambiguity when both *.binp and *.inp exist)
    p_bin = _write_numberdens_binp(mol, nco, model)
    # Clean up any stale ASCII file from previous runs
    try:
        ascii_path = model / f'numberdens_{mol.lower()}.inp'
        if ascii_path.exists():
            ascii_path.unlink()
    except Exception:
        pass

    return {'wrote_bin': str(p_bin), 'shape': (par.gas.ncol, par.gas.nrad, par.gas.nsec)}


#################
# PLOTTING HELPERS
######################


def plot_pdiss_fraction(model_dir: str | Path = '.', chi0: float = 1.0, kappa_uv: float = 1e3) -> None:
    model = Path(model_dir)
    par, _T, nH, chi = _prepare_fields(model, chi0=chi0, kappa_uv=kappa_uv)
    mask_pdiss = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDISS).astype(float)
    f_pdiss = np.sum(mask_pdiss, axis=2) / par.gas.nsec
    R, Z = rz_edges_au(par)
    outdir = str(model / 'chemistry_diagnostics')
    plot_rz_scalar_field(R=R, Z=Z, data2d=f_pdiss, cmap='nipy_spectral', norm=None, cbar_label='Photodissociation fraction', outdir=outdir, filename='photodissociation_fraction_Rz.png')


def plot_pinte_co_diagnostics(model_dir: str | Path = '.', chi0: float = 1.0, kappa_uv: float = 1e3) -> None:
    model = Path(model_dir)
    par, T, nH, chi = _prepare_fields(model, chi0=chi0, kappa_uv=kappa_uv)
    R, Z = rz_edges_au(par)
    outdir = str(model / 'chemistry_diagnostics')
    # chi
    plot_rz_scalar_field(R=R, Z=Z, data2d=np.mean(chi, axis=2), cmap='nipy_spectral', norm=None, cbar_label='chi (Draine units)', outdir=outdir, filename='chi_Rz.png')
    # log10(chi/nH)
    log_chi_over_nH = np.log10(chi / (nH + 1e-99))
    plot_rz_scalar_field(R=R, Z=Z, data2d=np.mean(log_chi_over_nH, axis=2), cmap='nipy_spectral', norm=None, cbar_label='log10(chi / nH)', outdir=outdir, filename='log10_chi_over_nH_Rz.png')
    # Temperature
    plot_rz_scalar_field(R=R, Z=Z, data2d=np.mean(T, axis=2), cmap='inferno', norm=None, cbar_label='Temperature [K]', outdir=str(model / 'temperature_diagnostics'), filename='temperature_Rz.png')
    # n_H
    plot_rz_scalar_field(R=R, Z=Z, data2d=np.mean(nH, axis=2), cmap='viridis', norm=None, cbar_label='n_H [cm$^{-3}$]', outdir=str(model / 'disk_diagnostics'), filename='nH_Rz.png')
