# Earth’s Jet Streams: interactive

An interactive, offline-capable web version of the “Earth’s Jet Streams: A Week of Winds at 250 mb” video.
Glowing particle trails follow the real NOAA GFS 0.5° analysis winds at 250 mb (about 10 to 11 km up) for
one week (29 six-hourly analyses, **19 Sep 2026 06:00 to 26 Sep 2026 06:00 UTC**), coloured by speed from 0 to 80+ m/s.

* Orthographic globe: **drag** to rotate, **scroll / pinch** to zoom. It auto-rotates gently and pauses while you interact.
* **Globe / Map** toggle morphs to an equirectangular world map (drag to pan, scroll/pinch to zoom).
* Timeline: play/pause, a scrubber over the 7 days, and a speed button (¼× to 4×; 1× = the week in one minute).
  Winds are interpolated linearly in time between the 6-hour analyses. Times are shown in UTC.
* **Click / tap** anywhere to read out latitude/longitude, wind speed (m/s, mph, km/h) and direction
  (meteorological “from”, in degrees and compass points) at the current time. The readout keeps updating while it plays.
* Keyboard: `Space` play/pause, `←/→` step one hour, `G` globe/map, `F` FPS meter, `Esc` close readout.

## Open it

Browsers block `fetch()` from `file://` pages, so serve the folder with any static web server:

```bash
cd jetstream-app
python3 -m http.server 8000        # then open http://localhost:8000
# or: npx serve .   /  php -S localhost:8000  /  any web server
```

It has no build step and loads nothing from the internet at runtime (fonts are vendored), so it works offline.

### Hosting

It is a plain static folder, so any static host works:

* **Netlify Drop**: drag the `jetstream-app` folder (or the zip, unpacked) onto https://app.netlify.com/drop.
* **GitHub Pages**: commit the folder contents to a repo, then Settings → Pages → deploy from branch (root).
* Cloudflare Pages, Vercel (static), S3/R2 buckets, or an nginx/Apache directory also work.

Requirements: a browser with **WebGL2** (current Chrome, Edge, Firefox, Safari 15+, iOS 15+, Android Chrome).

## URL options (handy for embedding or kiosks)

| param | effect |
|---|---|
| `?view=map` | start in map view |
| `?t=36` | start at hour 36 of the week |
| `?play=0` | start paused |
| `?spin=0` | no auto-rotate |
| `?lon=-100&lat=35` | initial globe centre |
| `?fps=1` | show the FPS / particle meter |
| `?q=0.5`, `?noadapt` | fixed quality factor / turn off adaptive particle count |
| `?dpr=1` | cap render resolution (default cap is 2× device pixels) |

## Files

```
index.html  css/app.css  js/app.js     the app (vanilla JS + WebGL2, no libraries)
fonts/                                  Source Serif Pro + IBM Plex Mono subsets (SIL OFL)
data/manifest.json                      times + per-step quantisation ranges
data/wind_00..28.png                    u/v wind at 250 mb, one lossless PNG per analysis (~0.5 MB each)
data/coast.json                         Natural Earth 50m coastline (public domain), simplified and delta-encoded
update_data.py                          refresh the data with the latest week from NOAA NOMADS
```

**Data encoding.** Each analysis keeps the native GFS 0.5° grid (720 × 361, row 0 = 90°N, column 0 = 0°E).
u and v are each linearly quantised to 12 bits between that step’s own min and max
(R = u>>4, G = v>>4, B = low nibbles). The worst-case round-trip error against the GRIB values is **0.019 m/s**.
Nothing is smoothed, resampled or invented. Readouts are bilinear in space and linear in time from these values.

## Updating to a new week

```bash
pip install numpy pillow eccodes          # eccodes wheels bundle the C library
python3 update_data.py                    # newest available cycle, previous 7 days
python3 update_data.py --end 2026100300   # week ending at a specific cycle (YYYYMMDDHH, UTC)
python3 update_data.py --grib-dir /path/to/grib2/files   # convert GRIBs you already have
python3 update_data.py --coast            # also rebuild coast.json (needs cartopy + shapely)
```

The script downloads just UGRD/VGRD at 250 mb (f000 analyses) through the NOMADS grib filter
(`filter_gfs_0p50.pl`, about 380 KB per cycle) into `./grib_cache/`, then rewrites `data/`. Reload the page to see the new week.
The title, timestamps and date range update automatically from `manifest.json`.
NOMADS only keeps the last ~10 days online. For older weeks, use the NOAA GFS archives on AWS/NCEI and `--grib-dir`.

## How it works

* The CPU advects 10k to 90k particles (depending on screen area and measured frame rate) through the
  time-interpolated wind field in lon/lat. The field is re-interpolated once per frame.
* Each particle’s last 10 positions live in an RGBA32F ring-buffer texture, so only one row block is uploaded per frame.
  Trails are drawn as additive GL lines whose vertices are projected in the vertex shader
  (orthographic ↔ equirectangular morph), so rotating the globe never smears the trails.
* The flow goes into a half-float buffer, then a quarter-resolution bloom and soft tone map (the same curve as the video renderer).
  After that come the globe disc, rim glow, and coastlines, which dim under bright flow.
* Adaptive quality lowers the particle count if the frame rate falls under ~48 fps.

## Credits

Data: NOAA/NCEP Global Forecast System (GFS) 0.5° analysis via NOMADS (public domain).
Coastlines: Natural Earth (public domain). Fonts: Source Serif Pro (Adobe) and IBM Plex Mono (IBM), SIL Open Font License 1.1.

## Live site and daily refresh

Live at **https://johnswehla3000-debug.github.io/jet-streams/**.
`.github/workflows/update.yml` runs daily at 11:20 UTC (after the 06Z GFS analysis is published), on manual dispatch,
and on every push to `main`. It installs `requirements.txt`, runs `update_data.py` to fetch the newest 7 days from NOMADS,
and deploys the site via `actions/upload-pages-artifact` + `actions/deploy-pages`.
The generated wind PNGs and `manifest.json` are **not** committed (they are gitignored) so the repository does not grow
by ~15 MB a day; run `python3 update_data.py` locally to generate them before serving the folder yourself.
