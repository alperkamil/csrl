# CSRL v2 — Installation Guide

This document explains how to set up **CSRL v2** and its external dependencies on a typical Linux system (tested on Ubuntu 22+).

> **Quick summary:** You need Python ≥ 3.10 (ideally in a virtualenv), the **Owl Toolkit**, and **Spot** with its Python bindings. Then `pip install -e .` inside this repo.


---

### Requirements
- [**Python**](https://www.python.org/) ≥ 3.10
- [**Owl**](https://owl.model.in.tum.de/) ≥ 21.0 — `ltl2ldba` and `ltl2dpa` binaries must be in your `PATH`
- [**Spot**](https://spot.lrde.epita.fr/) ≥ 2.11 — with Python bindings installed in your environment
---


## Python
See [Python Setup and Usage](https://docs.python.org/3/using/index.html) for installation details.
Ubuntu 22+ ships with Python ≥ 3.10.

Verify:
```bash
python3 --version
# Expect: Python 3.10.x or newer
```

Create and activate a virtual environment (recommended):

```bash
export VENV_DIR=~/venvs/csrl
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
```

---
## Owl Toolkit
Download from the [official releases](https://github.com/owl-toolkit/owl/releases/tag/release-21.0) and place binaries/libs on your `PATH`/`LD_LIBRARY_PATH`. Example:

```bash
# Choose a local install prefix that is already (or will be) on your PATH
export LOCAL=/usr/local

# Download and unpack (adjust the URL/version to match the release you want)
wget https://github.com/owl-toolkit/owl/releases/download/release-21.0/owl-linux-amd64-21.0.zip
unzip owl-linux-amd64-21.0.zip

# Copy into your prefix (use sudo if installing to /usr/local)
sudo cp -r owl-*/bin/* "$LOCAL/bin/"
sudo cp -r owl-*/lib/* "$LOCAL/lib/"

# Ensure your environment can see the libraries
export PATH="$LOCAL/bin:$PATH"
export LD_LIBRARY_PATH="$LOCAL/lib:${LD_LIBRARY_PATH:-}"
```

Verify:
```bash
owl --version
# Expect: owl (version: 21.0)
```

---
##  Spot
Follow [Spot’s docs](https://spot.lre.epita.fr/install.html) if you prefer packages; otherwise, build from source as below (with Python bindings into your venv):

```bash
# Where to install Spot (same LOCAL as above)
export LOCAL=/usr/local

# Compute your pythonX.Y string inside the active venv
export PYTHON_VERSION=$(python -c 'import sys;print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')

# Download a Spot tarball (adjust version as needed)
wget http://www.lrde.epita.fr/dload/spot/spot-2.12.1.tar.gz
tar -xzf spot-2.12.1.tar.gz
cd spot-2.12.1

# Optional: ensure build deps are present (Ubuntu)
# sudo apt-get install -y build-essential python3-dev pkg-config autoconf automake libtool bison flex
./configure --prefix="$LOCAL" --with-pythondir="$VENV_DIR/lib/$PYTHON_VERSION/site-packages"
make -j"$(nproc)"
sudo make install
cd ..
```

Confirm the Python bindings:
```bash
python -c "import spot"
# Expect no output
```

---
## Install CSRL
```bash
git clone https://github.com/alperkamil/csrl
cd csrl
pip install -e .
```


---
## Example: Ubuntu 22.04 script
The following is a complete script for Ubuntu 22.04 (requires sudo) that you can paste into a fresh shell:

```bash
# --- System essentials ---
sudo apt update
sudo apt install -y zip wget git build-essential python3-dev python3-venv pkg-config autoconf automake libtool bison flex

# --- Python virtual environment ---
export VENV_DIR=~/venvs/csrl
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel

# --- Locations for third-party installs ---
export LOCAL=/usr/local
export PATH="$LOCAL/bin:$PATH"
export LD_LIBRARY_PATH="$LOCAL/lib:${LD_LIBRARY_PATH:-}"

# --- Owl Toolkit ---
wget https://github.com/owl-toolkit/owl/releases/download/release-21.0/owl-linux-amd64-21.0.zip
unzip -q owl-linux-amd64-21.0.zip
sudo cp -r owl-*/bin/* "$LOCAL/bin/"
sudo cp -r owl-*/lib/* "$LOCAL/lib/"
owl --version

# --- Spot ---
export PYTHON_VERSION=$(python -c 'import sys;print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')
wget http://www.lrde.epita.fr/dload/spot/spot-2.12.1.tar.gz
tar -xzf spot-2.12.1.tar.gz
cd spot-2.12.1
./configure --prefix="$LOCAL" --with-pythondir="$VENV_DIR/lib/$PYTHON_VERSION/site-packages"
make -j"$(nproc)"
sudo make install
cd ..

python -c "import spot"

# --- CSRL ---
git clone https://github.com/alperkamil/csrl
cd csrl
pip install -e .

```

