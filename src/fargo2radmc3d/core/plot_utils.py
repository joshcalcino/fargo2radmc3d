import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def species_label(gasspecies: str) -> str:
    if gasspecies == 'co':
        return r'$^{12}$CO'
    elif gasspecies == '13co':
        return r'$^{13}$CO'
    elif gasspecies == 'c17o':
        return r'C$^{17}$O'
    elif gasspecies == 'c18o':
        return r'C$^{18}$O'
    else:
        return str(gasspecies).upper()


def rz_edges_au(par):
    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge, par.gas.tedge)
    R = radius_matrix * np.sin(theta_matrix) * par.gas.culength / 1.5e11
    Z = radius_matrix * np.cos(theta_matrix) * par.gas.culength / 1.5e11
    return R, Z


def xy_edges_au(par):
    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge, par.gas.pedge)
    X = radius_matrix * np.cos(theta_matrix) * par.gas.culength / 1.5e11
    Y = radius_matrix * np.sin(theta_matrix) * par.gas.culength / 1.5e11
    return X, Y


def _setup_axes(figsize=(8., 8.)):
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='DejaVu Sans')
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
    ax = plt.gca()
    ax.tick_params(top='on', right='on', length=5, width=1.0, direction='out')
    ax.tick_params(axis='x', which='minor', top=True)
    ax.tick_params(axis='y', which='minor', right=True)
    return fig, ax


def plot_rz_scalar_field(R, Z, data2d, cmap: str = 'nipy_spectral', norm = None, cbar_label: str = '', outdir: str = '.', filename: str = 'rz_plot.png'):
    if norm is None:
        vmin = float(np.nanmin(data2d))
        vmax = float(np.nanmax(data2d))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = _setup_axes()
    ax.set_xlabel('Radius [au]')
    ax.set_ylabel('Altitude [au]')
    ax.set_ylim(float(np.nanmin(Z)), float(np.nanmax(Z)))
    ax.set_xlim(float(np.nanmin(R)), float(np.nanmax(R)))
    CF = ax.pcolormesh(R, Z, data2d, cmap=cmap, norm=norm, rasterized=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='2.5%', pad=0.12)
    cb = plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.xaxis.set_label_position('top')
    cax.set_xlabel(cbar_label)
    cax.xaxis.labelpad = 8
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, filename), dpi=160)
    plt.close(fig)


def plot_sky_image(data2d, extent, cmap: str = 'viridis', norm = None, cbar_label: str = '', outdir: str = '.', filename: str = 'sky_image.png', xlabel: str = 'RA offset [arcsec]', ylabel: str = 'Dec offset [arcsec]', draw_origin_marker: bool = False):
    if norm is None:
        vmin = float(np.nanmin(data2d))
        vmax = float(np.nanmax(data2d))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = _setup_axes()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlim(extent[0], extent[1])
    CF = ax.imshow(data2d, cmap=cmap, origin='lower', interpolation='bilinear', extent=extent, norm=norm, aspect='auto')
    if draw_origin_marker:
        ax.plot(0.0, 0.0, '+', color='white', markersize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='2.5%', pad=0.12)
    cb = plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.xaxis.set_label_position('top')
    cax.set_xlabel(cbar_label)
    cax.xaxis.labelpad = 8
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, filename), dpi=160)
    plt.close(fig)


def plot_xy_scalar_field(X, Y, data2d, cmap: str = 'nipy_spectral', norm = None, cbar_label: str = '', outdir: str = '.', filename: str = 'xy_plot.png'):
    if norm is None:
        vmin = float(np.nanmin(data2d))
        vmax = float(np.nanmax(data2d))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = _setup_axes()
    ax.set_xlabel('x [au]')
    ax.set_ylabel('y [au]')
    ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
    ax.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    CF = ax.pcolormesh(X, Y, data2d, cmap=cmap, norm=norm, rasterized=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='2.5%', pad=0.12)
    cb = plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.xaxis.set_label_position('top')
    cax.set_xlabel(cbar_label)
    cax.xaxis.labelpad = 8
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, filename), dpi=160)
    plt.close(fig)
