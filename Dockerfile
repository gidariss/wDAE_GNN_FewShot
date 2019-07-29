FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN yes | unminimize

RUN apt-get update && apt-get install -y wget bzip2
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes

RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl

RUN pip install tensorboardX scikit-image tqdm pyyaml easydict future h5py torchnet pip
RUN apt-get install unzip

COPY ./ ./wDAE_GNN_FewShot
RUN pip install -e ./wDAE_GNN_FewShot

WORKDIR ./wDAE_GNN_FewShot
