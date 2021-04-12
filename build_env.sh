#! /bin/bash

# Install GPU driver
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

sudo apt update
sudo apt install -y cuda
sudo apt install -y build-essential \
    libffi-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    python-pip \
    make \
    wget \
    curl \
    llvm \
    xclip \
    unzip \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    libssl-dev \
    python \
    python3 \
    python3-dev \
    python3-venv \
    python3-distutils


# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
wait
source $HOME/.poetry/env
poetry config virtualenvs.in-project true

# Install pyenv
curl https://pyenv.run | bash
wait
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install python with pyenv.
pyenv install 3.8.7
wait
pyenv global 3.8.7
