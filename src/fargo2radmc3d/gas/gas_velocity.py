# import global variables
from ..core import par

import numpy as np

from ..core.mesh import *
from ..core.field import *

import matplotlib
import matplotlib.pyplot as plt
import os
from ..core.plot_utils import xy_edges_au, plot_xy_scalar_field, plot_sky_image, species_label

# =========================
# Microturbulent line broadening
# =========================
def write_gas_microturb():

    MTURB = open('microturbulence.binp','wb')
    # requested header
    # hdr[0] = format number
    # hdr[1] = data precision (8 means double)
    # hdr[2] = nb of grid cells
    hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol], dtype=int)
    hdr.tofile(MTURB)

    # Default case: uniform microturbulence set by 'turbvel' parameter in params.dat
    microturb = np.ones((par.gas.ncol,par.gas.nrad,par.gas.nsec))*par.turbvel*1.0e2  # ncol, nrad, nsec in cm/s

    if par.turbvel == 'cavity':  # Baruteau et al. 2021 model
        for k in range(par.gas.nsec):
            for j in range(par.gas.ncol):
                for i in range(par.gas.nrad):
                    if par.gas.rmed[i] < 1.0:
                        myalpha = 3e-2   # inside cavity
                    else:
                        myalpha = 3e-4   # outside cavity
                    # v_turb ~ sqrt(alpha) x isothermal sound speed
                    microturb[j,i,k] = np.sqrt(myalpha * par.kB * gas_temp[j,i,k] / 2.3 / par.mp)  
        if par.verbose:
            print('min and max of microturbulent velocity in cm/s = ',microturb.min(),microturb.max())

    # If writing data in an ascii file the ordering should be: nsec, ncol, nrad.
    # We therefore need to swap axes of array microturb
    # before dumping it in a binary file! just like mastermind game!
    microturb = np.swapaxes(microturb, 0, 1)  # nrad ncol nsec
    microturb = np.swapaxes(microturb, 0, 2)  # nsec ncol nrad
    microturb.tofile(MTURB)
    MTURB.close()


# =========================
# Compute gas velocity field on RADMC 3D grid
# =========================
def compute_gas_velocity():

    # 3D simulation carried out with Fargo 3D (spherical coordinates already)
    if par.fargo3d == 'Yes' and par.hydro2D == 'No':

        vtheta3D  = Field(field='gasvz'+str(par.on)+'.dat', directory=par.dir).data  # code units
        vtheta3D *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s
        
        vrad3D    = Field(field='gasvy'+str(par.on)+'.dat', directory=par.dir).data  # code units
        vrad3D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        vphi3D    = Field(field='gasvx'+str(par.on)+'.dat', directory=par.dir).data  # code units
        init_vphi_path = os.path.join(par.dir if par.dir.endswith('/') == False else par.dir[:-1], 'gasvx0.dat')
        if os.path.isfile(init_vphi_path):
            vphi3D0 = Field(field='gasvx0.dat', directory=par.dir).data  # code units
        else:
            vphi3D0 = np.zeros_like(vphi3D)

        f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        omegaframe  = omega[par.on]
        omegaframe0 = omega[0]
        
        for theta in range(par.gas.ncol):
            for phi in range(par.gas.nsec):
                vphi3D[theta,:,phi] += par.gas.rmed*omegaframe
                vphi3D0[theta,:,phi] += par.gas.rmed*omegaframe0
        vphi3D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s
        vphi3D0  *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

    else:  # 2D simulation carried out with dusty fargo adsg or Fargo 3D

        # arrays allocation
        vrad3D_cyl   = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))   # zeros!
        vphi3D_cyl   = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))   # zeros!
        vphi3D0_cyl  = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))   # zeros!
        vtheta3D     = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!
        vrad3D       = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!
        vphi3D       = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!
        vphi3D0      = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!

        if par.fargo3d == 'No':
            vrad2D   = Field(field='gasvrad'+str(par.on)+'.dat', directory=par.dir).data  # code units
            vphi2D   = Field(field='gasvtheta'+str(par.on)+'.dat', directory=par.dir).data  # code units
            vphi2D0  = Field(field='gasvtheta0.dat', directory=par.dir).data  # code units
            f1, xpla, ypla, f4, f5, f6, f7, date, omega, f10, f11 = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        else:
            vrad2D    = Field(field='gasvy'+str(par.on)+'.dat', directory=par.dir).data  # code units
            vphi2D    = Field(field='gasvx'+str(par.on)+'.dat', directory=par.dir).data  # code units
            vphi2D0   = Field(field='gasvx0.dat', directory=par.dir).data  # code units
            f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
            
        vrad2D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        omegaframe  = omega[par.on]
        omegaframe0 = omega[0]
        print('omegaframe = ', omegaframe)
        
        for phi in range(par.gas.nsec):
            vphi2D[:,phi]  += par.gas.rmed*omegaframe
            vphi2D0[:,phi] += par.gas.rmed*omegaframe0
        vphi2D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s
        vphi2D0  *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        # Make gas velocity axisymmetric (testing purposes)
        if ('axisymgas' in open('params.dat').read()) and (par.axisymgas == 'Yes'):
            axivrad2D = np.sum(vrad2D,axis=1)/par.gas.nsec
            axivphi2D = np.sum(vphi2D,axis=1)/par.gas.nsec  
            axivphi2D0 = np.sum(vphi2D0,axis=1)/par.gas.nsec  
            for i in range(par.gas.nrad):
                vrad2D[i,:] = axivrad2D[i]
                vphi2D[i,:] = axivphi2D[i]
                vphi2D0[i,:] = axivphi2D0[i]
            
        # Vertical expansion for vrad and vphi (vtheta being assumed zero)
        for z in range(par.gas.nver):
            vrad3D_cyl[z,:,:] = vrad2D
            vphi3D_cyl[z,:,:] = vphi2D
            vphi3D0_cyl[z,:,:] = vphi2D0

        # Now, sweep through the spherical grid
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                R = par.gas.rmed[i]*np.sin(par.gas.tmed[j])  # cylindrical radius
                z = par.gas.rmed[i]*np.cos(par.gas.tmed[j])  # vertical altitude
                icyl = np.argmin(np.abs(par.gas.rmed-R))
                if R < par.gas.rmed[icyl] and icyl > 0:
                    icyl-=1
                jcyl = np.argmin(np.abs(par.gas.zmed-z))
                if z < par.gas.zmed[jcyl] and jcyl > 0:
                    jcyl-=1
                if (icyl < par.gas.nrad-1 and jcyl < par.gas.nver-1):
                    vrad3D[j,i,:] = ( vrad3D_cyl[jcyl,icyl,:]*(par.gas.rmed[icyl+1]-R)*(par.gas.zmed[jcyl+1]-z) + vrad3D_cyl[jcyl+1,icyl,:]*(par.gas.rmed[icyl+1]-R)*(z-par.gas.zmed[jcyl]) + vrad3D_cyl[jcyl,icyl+1,:]*(R-par.gas.rmed[icyl])*(par.gas.zmed[jcyl+1]-z) + vrad3D_cyl[jcyl+1,icyl+1,:]*(R-par.gas.rmed[icyl])*(z-par.gas.zmed[jcyl]) ) / ( (par.gas.rmed[icyl+1]-par.gas.rmed[icyl]) * (par.gas.zmed[jcyl+1]-par.gas.zmed[jcyl]) )
                    vphi3D[j,i,:] = ( vphi3D_cyl[jcyl,icyl,:]*(par.gas.rmed[icyl+1]-R)*(par.gas.zmed[jcyl+1]-z) + vphi3D_cyl[jcyl+1,icyl,:]*(par.gas.rmed[icyl+1]-R)*(z-par.gas.zmed[jcyl]) + vphi3D_cyl[jcyl,icyl+1,:]*(R-par.gas.rmed[icyl])*(par.gas.zmed[jcyl+1]-z) + vphi3D_cyl[jcyl+1,icyl+1,:]*(R-par.gas.rmed[icyl])*(z-par.gas.zmed[jcyl]) ) / ( (par.gas.rmed[icyl+1]-par.gas.rmed[icyl]) * (par.gas.zmed[jcyl+1]-par.gas.zmed[jcyl]) )
                    vphi3D0[j,i,:]= ( vphi3D0_cyl[jcyl,icyl,:]*(par.gas.rmed[icyl+1]-R)*(par.gas.zmed[jcyl+1]-z) + vphi3D0_cyl[jcyl+1,icyl,:]*(par.gas.rmed[icyl+1]-R)*(z-par.gas.zmed[jcyl]) + vphi3D0_cyl[jcyl,icyl+1,:]*(R-par.gas.rmed[icyl])*(par.gas.zmed[jcyl+1]-z) + vphi3D0_cyl[jcyl+1,icyl+1,:]*(R-par.gas.rmed[icyl])*(z-par.gas.zmed[jcyl]) ) / ( (par.gas.rmed[icyl+1]-par.gas.rmed[icyl]) * (par.gas.zmed[jcyl+1]-par.gas.zmed[jcyl]) )
                else:
                # simple nearest-grid point interpolation...
                    vrad3D[j,i,:] = vrad3D_cyl[jcyl,icyl,:]   
                    vphi3D[j,i,:] = vphi3D_cyl[jcyl,icyl,:]
                    vphi3D0[j,i,:]= vphi3D0_cyl[jcyl,icyl,:]
                
                if (('TestVz' in open('params.dat').read()) and (par.TestVz == 'Yes')):  # just to test: add a vertical velocity in planets wakes
                    for k in range(par.gas.nsec):
                        if (vphi2D[i,k] - vphi2D0[i,k])/vphi2D0[i,k] > 0.04:
                            if j < par.gas.ncol//2-2:
                                vtheta3D[j,i,k] = -0.2*vphi2D0[i,k]
                            else:
                                vtheta3D[j,i,k] = 0.2*vphi2D0[i,k]
                        elif (vphi2D[i,k] - vphi2D0[i,k])/vphi2D0[i,k] < -0.04:
                            if j < par.gas.ncol//2-2:
                                vtheta3D[j,i,k] = 0.2*vphi2D0[i,k]
                            else:
                                vtheta3D[j,i,k] = -0.2*vphi2D0[i,k]

                    
    print('--------- writing gas_velocity.inp file ----------')

    # Define gas velocity array that contains all three components
    gasvel = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec,3))
    gasvel[:,:,:,0] = vrad3D
    gasvel[:,:,:,1] = vtheta3D
    #vphi3D = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # CUIDADIN!! Testing...
    gasvel[:,:,:,2] = vphi3D

    VELOUT = open('gas_velocity.binp','wb')
    # requested header
    # hdr[0] = format number
    # hdr[1] = data precision (8 means double)
    # hdr[2] = nb of grid cells
    hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol], dtype=int)
    hdr.tofile(VELOUT)
    
    # If writing data in an ascii file the ordering should be: nsec, ncol, nrad
    # We therefore need to swap axes of array gasvel
    # before dumping it in a binary file! just like mastermind game!
    gasvel = np.swapaxes(gasvel, 0, 1)  # nrad ncol nsec
    gasvel = np.swapaxes(gasvel, 0, 2)  # nsec ncol nrad
    gasvel.tofile(VELOUT)
    VELOUT.close()
    

    # finally output plots of the gas velocity
    if par.plot_gas_quantities == 'Yes':
        outdir = 'disk_diagnostics'
        os.makedirs(outdir, exist_ok=True)

        if par.half_a_disc == 'No':
            midplane_col_index = par.gas.ncol//2-1
        else:
            midplane_col_index = par.gas.ncol-1

        vrmid = vrad3D[midplane_col_index,:,:]/1e5
        vtmid  = vphi3D[midplane_col_index,:,:]/1e5
        vtmid0 = vphi3D0[midplane_col_index,:,:]/1e5
        vvupper  = vtheta3D[par.gas.ncol-1,:,:]/1e5

        X, Y = xy_edges_au(par)
        strgas = species_label(par.gasspecies)

        print('--------- a) plotting midplane radial velocity (x,y) ----------')
        mynorm = matplotlib.colors.Normalize(vmin=vrmid.min(), vmax=vrmid.max())
        plot_xy_scalar_field(X=X, Y=Y, data2d=np.transpose(vrmid), cmap='nipy_spectral', norm=mynorm, cbar_label=strgas+' midplane radial velocity '+r'[km s$^{-1}$]', outdir=outdir, filename='vrad_midplane.png')

        print('--------- b) plotting midplane azimuthal velocity (x,y) ----------')
        mynorm = matplotlib.colors.Normalize(vmin=vtmid.min(), vmax=vtmid.max())
        plot_xy_scalar_field(X=X, Y=Y, data2d=np.transpose(vtmid), cmap='nipy_spectral', norm=mynorm, cbar_label=strgas+' midplane azimuthal velocity '+r'[km s$^{-1}$]', outdir=outdir, filename='vphi_midplane.png')

        print('--------- c) plotting upper vertical velocity (x,y) ----------')
        mynorm = matplotlib.colors.Normalize(vmin=vvupper.min(), vmax=vvupper.max())
        plot_xy_scalar_field(X=X, Y=Y, data2d=np.transpose(vvupper), cmap='nipy_spectral', norm=mynorm, cbar_label=strgas+' above vertical velocity '+r'[km s$^{-1}$]', outdir=outdir, filename='vvert_upper.png')


        print('--------- d) plotting line-of-sight velocity in sky plane ----------')

        vlos_disc_plane = np.zeros((par.gas.nrad,par.gas.nsec))

        # sky-plane x- and y- coordinates in arcseconds
        Xs = np.zeros((par.nbpixels,par.nbpixels))
        Ys = np.zeros((par.nbpixels,par.nbpixels))
        v_los = np.zeros((par.nbpixels,par.nbpixels))
        v_los_residual = np.zeros((par.nbpixels,par.nbpixels))
        
        # midplane radial velocity:
        vrmid = vrad3D[midplane_col_index,:,:]/1e5 # nrad, nsec  # in km/s
        vrmid = np.roll(vrmid,shift=int(par.gas.nsec//2),axis=1)

        # midplane azimuthal velocity:
        vtmid = vphi3D[midplane_col_index,:,:]/1e5 # nrad, nsec  # in km/s
        vtmid = np.roll(vtmid,shift=int(par.gas.nsec//2),axis=1)
        vtmid0 = vphi3D0[midplane_col_index,:,:]/1e5 # nrad, nsec  # in km/s
        vtmid0 = np.roll(vtmid0,shift=int(par.gas.nsec//2),axis=1)

        # midplane vertical velocity:
        vzmid = vtheta3D[midplane_col_index,:,:]/1e5 # nrad, nsec  # in km/s
        vzmid = np.roll(vzmid,shift=int(par.gas.nsec//2),axis=1)


        # minimum and maximum values of x- and y-coordinates on sky-plane:
        if par.minmaxaxis == '#':
            maxXsYs = par.gas.culength*par.gas.rmed[par.gas.nrad-1]/(par.au*1e-2)/par.distance
        else:
            maxXsYs = par.minmaxaxis

        incl = par.inclination*np.pi/180.0
        # Loop over sky-plane grid
        for i in range(par.nbpixels):
            for j in range(par.nbpixels):
                Xs[i,j] = -maxXsYs + 2.0*maxXsYs*i/(par.nbpixels-1.0)  # in arcseconds
                Ys[i,j] = -maxXsYs + 2.0*maxXsYs*j/(par.nbpixels-1.0)  # in arcseconds
                xs = Xs[i,j]*par.distance*par.au/(1e2*par.gas.culength)  # in code units
                ys = Ys[i,j]*par.distance*par.au/(1e2*par.gas.culength)  # in code units
                # rotation by -posangle
                xsb = xs*np.cos(-par.posangle*np.pi/180.0) - ys*np.sin(-par.posangle*np.pi/180.0)
                ysb = xs*np.sin(-par.posangle*np.pi/180.0) + ys*np.cos(-par.posangle*np.pi/180.0)
                # deproject with inclination
                xsc = xsb
                ysc = ysb/np.cos(incl)
                # rotation by -phiangle
                xsd = xsc*np.cos(-par.phiangle*np.pi/180.0) - ysc*np.sin(-par.phiangle*np.pi/180.0)
                ysd = xsc*np.sin(-par.phiangle*np.pi/180.0) + ysc*np.cos(-par.phiangle*np.pi/180.0)
                # polar coordinates in disc plane
                rd = np.sqrt(xsd**2 + ysd**2)
                td = math.atan2(ysd,xsd) + np.pi # between 0 and 2pi

                # find indices in disc simulation's grid 
                if rd >= par.gas.redge.min() and rd <= par.gas.redge.max():

                    id  = np.argmin(np.abs(par.gas.rmed-rd))
                    if rd < par.gas.rmed[id] and id > 0:
                        id -= 1

                    jd  = np.argmin(np.abs(par.gas.pedge-td))
                    if td < par.gas.pedge[jd] and jd > 0:
                        jd -= 1

                    # Bilinear interpolation                    
                    if ( rd >= par.gas.rmed[0] and id < par.gas.nrad-1 ):

                        jdp1 = jd+1
                        dr   = par.gas.rmed[id+1] - par.gas.rmed[id]
                        dphi = 2.0*np.pi/par.gas.nsec
                        
                        # weighting coefficient when doing bilinear interpolation
                        xij     = (par.gas.rmed[id+1]-rd) * (par.gas.pedge[jdp1]-td) / (dr*dphi)
                        xijp1   = (par.gas.rmed[id+1]-rd) * (td-par.gas.pedge[jd])   / (dr*dphi)
                        xip1j   = (rd-par.gas.rmed[id])   * (par.gas.pedge[jdp1]-td) / (dr*dphi)
                        xip1jp1 = (rd-par.gas.rmed[id])   * (td-par.gas.pedge[jd])   / (dr*dphi)

                        # Check all is well...
                        if xij < 0 or xij > 1:
                            print('beware that xij < 0 or xij > 1 in gas_velocity.py:',id,jd,xij,par.gas.rmed[id],rd,par.gas.rmed[id+1],par.gas.pedge[jd],td,par.gas.pedge[jdp1])
                        if xijp1 < 0 or xijp1 > 1:
                            print('beware that xijp1 < 0 or xijp1 > 1 in gas_velocity.py:',id,jd,xij,par.gas.rmed[id],rd,par.gas.rmed[id+1],par.gas.pedge[jd],td,par.gas.pedge[jdp1])
                        if xip1j < 0 or xip1j > 1:
                            print('beware that xip1j < 0 or xip1j > 1 in gas_velocity.py:',id,jd,xij,par.gas.rmed[id],rd,par.gas.rmed[id+1],par.gas.pedge[jd],td,par.gas.pedge[jdp1])
                        if xip1jp1 < 0 or xip1jp1 > 1:
                            print('beware that xip1jp1 < 0 or xip1jp1 > 1 in gas_velocity.py:',id,jd,xij,par.gas.rmed[id],rd,par.gas.rmed[id+1],par.gas.pedge[jd],td,par.gas.pedge[jdp1])

                        # final interpolated value
                        if jd == par.gas.nsec-1:
                            vrd = vrmid[id,jd]*xij + vrmid[id,0]*xijp1 + vrmid[id+1,jd]*xip1j + vrmid[id+1,0]*xip1jp1
                            vtd = vtmid[id,jd]*xij + vtmid[id,0]*xijp1 + vtmid[id+1,jd]*xip1j + vtmid[id+1,0]*xip1jp1
                            vtd0= vtmid0[id,jd]*xij + vtmid0[id,0]*xijp1 + vtmid0[id+1,jd]*xip1j + vtmid0[id+1,0]*xip1jp1
                        else:
                            vrd = vrmid[id,jd]*xij + vrmid[id,jdp1]*xijp1 + vrmid[id+1,jd]*xip1j + vrmid[id+1,jdp1]*xip1jp1
                            vtd = vtmid[id,jd]*xij + vtmid[id,jdp1]*xijp1 + vtmid[id+1,jd]*xip1j + vtmid[id+1,jdp1]*xip1jp1
                            vtd0= vtmid0[id,jd]*xij + vtmid0[id,jdp1]*xijp1 + vtmid0[id+1,jd]*xip1j + vtmid0[id+1,jdp1]*xip1jp1
                        #vrd = vrmid[id,jd]
                        #vtd = vtmid[id,jd]

                    else:
                    # simple nearest-grid point interpolation...
                        vrd = vrmid[id,jd]
                        vtd = vtmid[id,jd]
                        vtd0 = vtmid0[id,jd]

                    #vrd = vrmid[id,jd]
                    #vtd = vtmid[id,jd]
                    vzd = vzmid[id,jd]

                else:
                    vtd = 0.0
                    vtd0= 0.0
                    vrd = 0.0

                    vzd = 0.0

                # line-of-sight velocity in skyplane
                #v_los[i,j] = vtd*np.sin(incl)*np.cos(math.atan2(ysc,xsc)) + vrd*np.sin(incl)*np.sin(math.atan2(ysc,xsc))
                v_los[i,j] = vtd*np.sin(incl)*np.cos(math.atan2(ysc,xsc)) + vrd*np.sin(incl)*np.sin(math.atan2(ysc,xsc)) - vzd*np.cos(incl)
                # line-of-sight residual velocity in skyplane (initial azimuthal velocity is subtracted; initial radial velocity is not)
                # v_los_residual[i,j] = (vtd-vtd0)*np.sin(incl)*np.cos(math.atan2(ysc,xsc)) + vrd*np.sin(incl)*np.sin(math.atan2(ysc,xsc))
                v_los_residual[i,j] = (vtd-vtd0)*np.sin(incl)*np.cos(math.atan2(ysc,xsc)) + vrd*np.sin(incl)*np.sin(math.atan2(ysc,xsc)) - vzd*np.cos(incl)

        print('min and max of v_los = ',v_los.min(),v_los.max())
        print('min and max of v_los_residual = ',v_los_residual.min(),v_los_residual.max())

        vlosabs = np.maximum(np.abs(v_los.min()), np.abs(v_los.max()))
        mynorm = matplotlib.colors.Normalize(vmin=-vlosabs, vmax=vlosabs)
        plot_sky_image(data2d=v_los, extent=[-maxXsYs, maxXsYs, -maxXsYs, maxXsYs], cmap='RdBu_r', norm=mynorm, cbar_label='Disc midplane line of sight velocity [km/s]', outdir=outdir, filename='vlos_midplane.png', draw_origin_marker=True)

        # ---------------------------------
        # figure with residual los velocity
        # ---------------------------------
        vlosabs = np.maximum(np.abs(v_los_residual.min()), np.abs(v_los_residual.max()))
        mynorm = matplotlib.colors.Normalize(vmin=-vlosabs, vmax=vlosabs)
        plot_sky_image(data2d=v_los_residual, extent=[-maxXsYs, maxXsYs, -maxXsYs, maxXsYs], cmap='RdBu_r', norm=mynorm, cbar_label='Disc midplane line of sight residual  velocity [km/s]', outdir=outdir, filename='vlos_residual_midplane.png', draw_origin_marker=True)

    if par.hydro2D == 'Yes':
        del vrad3D_cyl, vphi3D_cyl, vphi3D0_cyl
    del vrad3D, vphi3D, vphi3D0, vtheta3D
