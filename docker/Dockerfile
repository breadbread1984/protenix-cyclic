FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

WORKDIR /app

COPY . .

RUN apt update

RUN apt install -y wget

RUN apt install -y libxrender1 libxext6 libx11-6

RUN apt install -y kalign hmmer

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm -rf Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda create -n protenix python=3.10 -y && conda clean --all -y

SHELL ["conda", "run", "-n", "protenix", "/bin/bash", "-c"]
ENV PATH="/opt/conda/envs/protenix/bin:$PATH"

RUN python3 -m pip install -r requirements.txt --break-system-packages

RUN python3 setup.py build

RUN python3 setup.py install # install protenix

RUN python3 /app/scripts/gen_ccd_cache.py -c /root/common/ -n 8 # download ccd

