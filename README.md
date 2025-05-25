# CSRL v2: Control Synthesis from Formal Specifications using Reinforcement Learning
This repository includes implementations of several learning-based synthesis algorithms. 


## Dependencies
 - [Python](https://www.python.org/) (>=3.10)
 - [Owl](https://owl.model.in.tum.de/) (>=21.0) : `ltl2ldba` and `ltl2dpa` must be in `PATH`
 - [Spot](https://spot.lrde.epita.fr/) (>=2.11)


## Example Installation on Ubuntu 22.04 with Root Access

## CSRL
```sh
git clone https://github.com/alperkamil/csrl.git
export PROJECT_HOME=$PWD/csrl
cd $PROJECT_HOME
```

## Tools
```sh
sudo apt install zip git build-essential python3-dev python3-venv
```

### Virtual Environment
```sh
python3 -m venv .venv
source .venv/bin/activate
```

### Python Packages
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### Owl and Spot
```sh
wget https://github.com/owl-toolkit/owl/releases/download/release-21.0/owl-linux-amd64-21.0.zip
unzip owl-linux-amd64-21.0.zip
sudo cp owl-linux-musl-amd64-21.0/bin/* /usr/local/bin/
sudo cp owl-linux-musl-amd64-21.0/lib/* /usr/local/lib/

wget http://www.lrde.epita.fr/dload/spot/spot-2.12.1.tar.gz
tar -xzf spot-2.12.1.tar.gz
cd spot-2.12.1
./configure --prefix=/usr/local --with-pythondir=$PROJECT_HOME/.venv/lib/python3.10/site-packages
make -j8
sudo make install
cd ..
```
