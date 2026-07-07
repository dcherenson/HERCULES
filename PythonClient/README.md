# Python API for AirSim

This package contains simple Python client for [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim). 
It can also be installed as a Python module. This integrates most API functions over RPC.

Note that this is renamed `hercules_cosysairsim` from the original `airsim` module. 

## Dependencies
This package depends on `numpy` and `msgpack` and would automatically install `numpy` and `rpc-msgpack` (this may need administrator/sudo prompt):
```
pip install numpy
pip install rpc-msgpack
```

## HERCULES Python environment (`herculesvenv`)

This fork is **not published to PyPI**, so `pip install hercules-cosys-airsim` will not
work. Instead, recreate the reference virtual environment used to run these scripts.
An exact pinned lockfile is provided at
[`requirements-herculesvenv.txt`](requirements-herculesvenv.txt) (Python 3.10.12).

```bash
# from the repo root (or wherever you keep envs)
python3 -m venv herculesvenv
source herculesvenv/bin/activate          # Windows: herculesvenv\Scripts\activate
pip install --upgrade pip
pip install -r PythonClient/requirements-herculesvenv.txt
```

The scripts locate the `hercules_cosysairsim` package through each folder's
`setup_path.py` (which adds `PythonClient/` to `sys.path`), so **no install of the
client package itself is required** — just activate the venv and run, e.g.:

```bash
cd PythonClient/car && python hello_car.py
```

## (Optional) Installing the client as a Python module from source

If you prefer `import hercules_cosysairsim` to work from anywhere without `setup_path`,
install the package from the `PythonClient` folder:

```bash
pip install ./PythonClient
```

> **Requires `setuptools >= 61`.** The reference venv ships setuptools 59.6.0, which is
> too old for the `[project]` table in `pyproject.toml` and will silently build a broken
> `UNKNOWN-0.0.0` wheel. Run `pip install --upgrade setuptools` first.

## More Info

More information on AirSim can be found at:
https://cosys-lab.github.io/

