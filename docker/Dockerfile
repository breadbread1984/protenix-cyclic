FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

WORKDIR /app

COPY . .

RUN apt install -y python3 python3-pip

RUN python3 -m pip install -r requirements.txt

RUN python3 setup.py build

RUN python3 setup.py install # install protenix

RUN python3 /app/scripts/gen_ccd_cache.py -c /root/common/ -n 8 # download ccd

