import numpy as np
from pathlib import Path
from typing import Tuple, Union

from .header import AMReXParticleHeader


def read_amrex_binary_particle_file(
    fn: Union[str, Path], header: AMReXParticleHeader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns the particle data stored in a particular
    plot file. It returns two numpy arrays, the
    first containing the particle integer data, and the second the
    particle real data.
    """
    ptype = "particles"
    base_fn = Path(fn) / ptype

    idtype = header.idtype_str
    fdtype = header.rdtype_str

    idata = np.empty((header.num_particles, header.num_int), dtype=header.int_type)
    rdata = np.empty((header.num_particles, header.num_real), dtype=header.real_type)

    ip = 0
    for lvl, level_grids in enumerate(header.grids):
        for which, count, where in level_grids:
            if count == 0:
                continue
            fn = base_fn / f"Level_{lvl}" / f"DATA_{which:05d}"

            with open(fn, "rb") as f:
                f.seek(where)
                if header.is_checkpoint:
                    ints = np.fromfile(f, dtype=idtype, count=count)
                    idata[ip : ip + count] = ints

                floats = np.fromfile(f, dtype=fdtype, count=count)
                rdata[ip : ip + count] = floats
            ip += count

    return idata, rdata
