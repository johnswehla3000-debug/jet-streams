#!/usr/bin/env python3
"""
update_data.py - refresh the Jet Stream web app's wind data from NOAA NOMADS.

Downloads the most recent 7 days (29 six-hourly analyses, f000) of GFS 0.5 deg
UGRD/VGRD at 250 mb via the NOMADS grib filter (the same source the video
renderer used), then rewrites ./data/wind_XX.png + ./data/manifest.json.

Requirements:  pip install numpy pillow eccodes   (eccodes wheels bundle the C lib)
Optional:      pip install cartopy shapely  (only for --coast, regenerating coast.json)

Usage:
  python update_data.py                   # latest available week from NOMADS
  python update_data.py --end 2026092606  # week ending at a given cycle (YYYYMMDDHH)
  python update_data.py --grib-dir ../jetstream/data   # convert already-downloaded GRIBs
  python update_data.py --coast           # also rebuild data/coast.json (Natural Earth 50m)

NOMADS keeps only ~10 days of GFS online, so older weeks are not available there.

Encoding: each analysis is one lossless RGB PNG on the native 0.5 deg grid
(720 x 361, row 0 = 90N, col 0 = 0E, eastward). u and v are each linearly
quantized to 12 bits between that step's own min and max (step ~0.03 m/s):
  R = u12 >> 4,  G = v12 >> 4,  B = ((u12 & 15) << 4) | (v12 & 15)
  u = umin + u12 * (umax - umin) / 4095  (likewise v); ranges are in manifest.json.
"""
import argparse, datetime as dt, glob, json, os, sys, time, urllib.request, urllib.error

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
NOMADS = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?'
          'dir=%2Fgfs.{ymd}%2F{hh}%2Fatmos&file=gfs.t{hh}z.pgrb2full.0p50.f000'
          '&var_UGRD=on&var_VGRD=on&lev_250_mb=on')
NSTEPS = 29          # 7 days, 6-hourly, inclusive


def url_for(t):
    return NOMADS.format(ymd=t.strftime('%Y%m%d'), hh=t.strftime('%H'))


def fetch(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                b = r.read()
            if not b.startswith(b'GRIB'):
                raise IOError('response is not GRIB (cycle probably not published yet)')
            with open(dest, 'wb') as f:
                f.write(b)
            return True
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                break
            time.sleep(3 * (k + 1))
    print('   fail:', last)
    return False


def latest_cycle(gdir):
    """Newest 6-hourly cycle whose f000 file is available on NOMADS."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    t = now.replace(minute=0, second=0, microsecond=0, hour=now.hour - now.hour % 6)
    os.makedirs(gdir, exist_ok=True)
    tmp = os.path.join(gdir, '_probe.grb2')
    for _ in range(8):
        print('probing', t.strftime('%Y-%m-%d %HZ'))
        if fetch(url_for(t), tmp, tries=1):
            os.remove(tmp)
            return t
        t -= dt.timedelta(hours=6)
    sys.exit('No recent GFS cycle found on NOMADS.')


def download(end, gdir):
    os.makedirs(gdir, exist_ok=True)
    times = [end - dt.timedelta(hours=6 * k) for k in range(NSTEPS)][::-1]
    for t in times:
        fn = os.path.join(gdir, f"gfs_{t.strftime('%Y%m%d_%H')}z_f000.grb2")
        if os.path.exists(fn) and os.path.getsize(fn) > 1000:
            continue
        print('downloading', t.strftime('%Y-%m-%d %HZ'))
        if not fetch(url_for(t), fn):
            sys.exit(f'download failed for {t}')
        time.sleep(0.6)      # be polite to NOMADS
    return [os.path.join(gdir, f"gfs_{t.strftime('%Y%m%d_%H')}z_f000.grb2") for t in times]


def read_grib(fn):
    """Return (u, v, valid-time) on a 361x720 grid, row0=90N, col0=0E (same as render.py's winds.npz)."""
    import eccodes
    d = {}
    with open(fn, 'rb') as f:
        while True:
            g = eccodes.codes_grib_new_from_file(f)
            if g is None:
                break
            try:
                sn = eccodes.codes_get(g, 'shortName')
                if eccodes.codes_get(g, 'level') != 250:
                    continue
                ni, nj = eccodes.codes_get(g, 'Ni'), eccodes.codes_get(g, 'Nj')
                vals = eccodes.codes_get_values(g).reshape(nj, ni).astype(np.float32)
                if eccodes.codes_get(g, 'latitudeOfFirstGridPointInDegrees') < 0:
                    vals = vals[::-1]
                d[sn] = vals
                date, tim = eccodes.codes_get(g, 'dataDate'), eccodes.codes_get(g, 'dataTime')
                ft = eccodes.codes_get(g, 'forecastTime')
            finally:
                eccodes.codes_release(g)
    t = dt.datetime.strptime(f'{date}{tim:04d}', '%Y%m%d%H%M') + dt.timedelta(hours=ft)
    return d['u'], d['v'], t


def q12(a):
    lo, hi = float(a.min()), float(a.max())
    q = np.round((a - lo) / (hi - lo) * 4095).astype(np.uint16)
    return q, lo, hi


def encode(files):
    os.makedirs(DATA, exist_ok=True)
    for old in glob.glob(os.path.join(DATA, 'wind_*.png')):
        os.remove(old)
    frames, times = [], []
    for k, fn in enumerate(files):
        u, v, t = read_grib(fn)
        assert u.shape == (361, 720), u.shape
        qu, ulo, uhi = q12(u)
        qv, vlo, vhi = q12(v)
        rgb = np.stack([(qu >> 4), (qv >> 4), ((qu & 15) << 4) | (qv & 15)], -1).astype(np.uint8)
        name = f'wind_{k:02d}.png'
        Image.fromarray(rgb, 'RGB').save(os.path.join(DATA, name), optimize=True)
        frames.append(dict(file=name, u=[round(ulo, 5), round(uhi, 5)], v=[round(vlo, 5), round(vhi, 5)]))
        times.append(t.strftime('%Y-%m-%dT%H:%MZ'))
        print(f'  {name}  {times[-1]}  max {float(np.hypot(u, v).max()):.1f} m/s')
    man = dict(source='NOAA GFS 0.5 deg analysis (f000), UGRD/VGRD 250 mb, via NOMADS',
               grid=dict(nlon=720, nlat=361, lon0=0.0, lat0=90.0, dlon=0.5, dlat=-0.5),
               encoding='12-bit linear per component: R=u>>4, G=v>>4, B=(u&15)<<4|(v&15)',
               step_hours=6, times=times, frames=frames,
               generated=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'))
    with open(os.path.join(DATA, 'manifest.json'), 'w') as f:
        json.dump(man, f, separators=(',', ':'))
    print('wrote', len(frames), 'frames ->', DATA)


def build_coast(res='50m', tol=0.04):
    """Natural Earth coastline -> compact JSON: list of polylines, integer lon/lat * 100, delta-encoded."""
    import cartopy.feature as cf
    lines = []
    for g in cf.NaturalEarthFeature('physical', 'coastline', res).geometries():
        for ln in getattr(g, 'geoms', [g]):
            ln = ln.simplify(tol, preserve_topology=False)
            c = np.round(np.asarray(ln.coords) * 100).astype(int)
            if len(c) < 2:
                continue
            flat = [int(c[0, 0]), int(c[0, 1])]
            for a, b in zip(c[:-1], c[1:]):
                flat += [int(b[0] - a[0]), int(b[1] - a[1])]
            lines.append(flat)
    with open(os.path.join(DATA, 'coast.json'), 'w') as f:
        json.dump(dict(src=f'Natural Earth {res} coastline (public domain)', scale=100, delta=True, lines=lines), f,
                  separators=(',', ':'))
    print('coast lines:', len(lines), 'points:', sum(len(l) for l in lines) // 2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--end', help='last cycle YYYYMMDDHH (UTC, HH in 00/06/12/18); default = newest available')
    ap.add_argument('--grib-dir', help='skip download; convert the 29 newest *.grb2 files in this directory')
    ap.add_argument('--cache', default=os.path.join(HERE, 'grib_cache'), help='download directory (default ./grib_cache)')
    ap.add_argument('--coast', action='store_true', help='also rebuild data/coast.json (needs cartopy)')
    a = ap.parse_args()
    if a.grib_dir:
        files = sorted(glob.glob(os.path.join(a.grib_dir, '*.grb2')))[-NSTEPS:]
    else:
        end = dt.datetime.strptime(a.end, '%Y%m%d%H') if a.end else latest_cycle(a.cache)
        files = download(end, a.cache)
    encode(files)
    if a.coast:
        build_coast()
    sys.stdout.flush()
    os._exit(0)   # avoid a harmless eccodes/GEOS teardown crash at interpreter exit
