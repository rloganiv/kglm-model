FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
RUN echo "deb-src http://archive.ubuntu.com/ubuntu/ xenial main" | tee -a /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev &&\
    rm -rf /var/lib/apy/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace

COPY experiments/ experiments/
COPY kglm/ kglm/
COPY .pylintrc .pylintrc
COPY pytest.ini pytest.ini
COPY README.md README.md
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN chmod -R a+w /workspace
