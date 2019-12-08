FROM python:3.6.8-jessie

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .pylintrc .pylintrc
COPY pytest.ini pytest.ini
COPY README.md README.md
COPY kglm/ kglm/
COPY experiments/ experiments/
