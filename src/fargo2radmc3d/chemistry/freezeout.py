from ..core import par
import numpy as np
import os


def apply_gas_freezeout(rhogascube, rhogascube_cyl, temp_source='hydro', threshold_K=19.0, eps=1e-5, dust_bin=-1):
    threshold = float(threshold_K)
    eps = float(eps)
    if temp_source == 'hydro':
        path = 'gas_tempcyl.binp'
        if not os.path.isfile(path):
            print('[freezeout] gas_tempcyl.binp not found; skipping hydro-based freezeout')
            return rhogascube, rhogascube_cyl
        buf = np.fromfile(path, dtype='float64')[3:]
        if par.hydro2D == 'No':
            try:
                gas_temp = buf.reshape(par.gas.ncol, par.gas.nrad, par.gas.nsec)
            except Exception as e:
                print(f'[freezeout] reshape hydro temp failed: {e}; skipping')
                return rhogascube, rhogascube_cyl
            axitemp = np.sum(gas_temp, axis=2) / par.gas.nsec
            for j in range(par.gas.ncol):
                for i in range(par.gas.nrad):
                    if axitemp[j, i] < threshold and rhogascube is not None:
                        rhogascube[j, i, :] *= eps
        else:
            try:
                gas_temp_cyl = buf.reshape(par.gas.nver, par.gas.nrad, par.gas.nsec)
            except Exception as e:
                print(f'[freezeout] reshape hydro temp failed: {e}; skipping')
                return rhogascube, rhogascube_cyl
            axitemp = np.sum(gas_temp_cyl, axis=2) / par.gas.nsec
            for j in range(par.gas.nver):
                for i in range(par.gas.nrad):
                    if axitemp[j, i] < threshold and rhogascube_cyl is not None:
                        rhogascube_cyl[j, i, :] *= eps
        try:
            os.remove(path)
        except Exception:
            pass
        return rhogascube, rhogascube_cyl
    elif temp_source == 'dust':
        path = 'dust_temperature.bdat'
        if not os.path.isfile(path):
            print('[freezeout] dust_temperature.bdat not found; skipping dust-based freezeout')
            return rhogascube, rhogascube_cyl
        buf = np.fromfile(path, dtype='float64')
        if buf.size < 4:
            print('[freezeout] dust_temperature.bdat too small; skipping')
            return rhogascube, rhogascube_cyl
        data = buf[4:]
        try:
            Temp = data.reshape(par.nbin, par.gas.nsec, par.gas.ncol, par.gas.nrad)
        except Exception as e:
            print(f'[freezeout] reshape dust temp failed: {e}; skipping')
            return rhogascube, rhogascube_cyl
        ibin = par.nbin - 1 if int(dust_bin) == -1 else int(dust_bin)
        if ibin < 0 or ibin >= par.nbin:
            ibin = par.nbin - 1
        T = Temp[ibin, :, :, :]
        if rhogascube is None:
            return rhogascube, rhogascube_cyl
        mask = (T < threshold)
        try:
            rhogascube[mask] *= eps
        except Exception as e:
            print(f'[freezeout] applying mask failed: {e}; skipping')
        return rhogascube, rhogascube_cyl
    else:
        print(f'[freezeout] unknown temp_source={temp_source}; skipping')
        return rhogascube, rhogascube_cyl
