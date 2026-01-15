# Package Installation

This project can be installed in **editable (development)** mode or as a **standard pip package**
(wheel or source distribution).

## Requirements
- Python **3.9+** recommended
- `pip` ≥ 21
- NumPy, Pandas, Matplotlib, SciPy (installed automatically via dependencies)

---

### Option 1: Editable install (recommended for development)

Use this if you are modifying the code or running the tutorials locally.

```
git clone https://github.com/terryjbates/LaurieOnTracking.git
cd LaurieOnTracking
pip install -e .
```

#### Test the Install
```
python -c "import metrica; import metrica.Metrica_IO as mio; print('OK')"
```

### Option 2: Standard pip Install

Recommended for users.

#### Build the Package
```
pip install build
python -m build
```

This creates a `dist` directory containing:
```
* *.whl (wheel)
* .tar.gz (source distribution)
```

#### Install From the Wheel
```
pip install dist\metrica_tracking-<version>-py3-none-any.whl
```

### Install From Source
```
pip install dist\metrica_tracking-<version>.tar.gz
```

#### Verify Installation
```
python -c "import metrica; print(metrica.__file__)"

```
