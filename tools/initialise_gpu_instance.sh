# Ubuntu 16.04 GPU-instance with Nvidia driver properly installed
# Maintainer: Jihong
#
# ==================================== #

# Add docker's repo
sudo apt-get update
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo apt-add-repository 'deb https://apt.dockerproject.org/repo ubuntu-xenial main'
sudo apt-get update
# Install
sudo apt-get install -y docker-engine
# add user to docker group
sudo usermod -aG docker $USER

# install nvidia-docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# install python-pip
sudo apt-get install python-pip
