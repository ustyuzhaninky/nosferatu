# base image
# FROM nvidia/driver:418.40.04-ubuntu20.04:latest
FROM tensorflow/tensorflow:latest-gpu
FROM python:3.7

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# nvidia drivers
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN /bin/sh

# packaging dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    dh-make \
    fakeroot \
    build-essential \
    devscripts \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

#Configuring XServer
RUN apt-get update
RUN apt-get install -y xserver-xorg mesa-utils
# RUN nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# RUN /usr/bin/X :0 && export DISPLAY=:0

# Install python and pip
RUN pip install virtualenv
RUN virtualenv otc-env --system-site-packages
RUN chmod +x otc-env/bin/activate
RUN otc-env/bin/activate
RUN git clone https://github.com/ustyuzhaninky/nosferatu.git
COPY requirements.txt /nosferatu/
RUN otc-env/bin/pip install --no-cache-dir -r /nosferatu/requirements.txt

# Installing Obstacle Tower
# RUN wget https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip
# RUN unzip obstacletower_v4.0_linux.zip
RUN /otc-env/bin/activate
# RUN pip install nosferatu/.
RUN pip install git+https://github.com/ustyuzhaninky/nosferatu.git
RUN mkdir "results"

# tell the port number the container should expose
EXPOSE 6006

CMD ["python", "-um nosferatu.train --base_dir=results --gin_files=~/nosferatu/nosferatu_otc.gin"]
# CMD ["tensorboard --logdir=results"]

# run it with `docker run -it --rm --runtime=nvidia josedd/nosferatu:latest-gpu`
