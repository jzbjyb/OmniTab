FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN echo 1
# essential tools
RUN apt-get update
RUN apt-get -y install openssh-client vim tmux sudo apt-transport-https apt-utils curl \
    git wget lsb-release ca-certificates gnupg gcc g++ pv iftop libopenmpi-dev

# Conda environment
ENV MINICONDA_VERSION py37_4.9.2
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# bugs for amulet
RUN pip install pip==9.0.0
RUN pip install ruamel.yaml==0.16 --disable-pip-version-check
RUN pip install --upgrade pip

# TaBERT env
COPY scripts /tmp/scripts/
RUN export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0"
RUN conda env create --file /tmp/scripts/env.yml

# MAPO env
RUN conda env update --name tabert --file /tmp/scripts/env_mapo.yml
SHELL ["conda", "run", "-n", "tabert", "/bin/bash", "-c"]

# deepspeed
RUN pip install deepspeed
RUN apt-get -y install pdsh

# ELECTRA
RUN pip install -r /tmp/scripts/requirements_electra.txt

# wandb
RUN pip install wandb==0.10.33

# timeout
RUN pip install timeout-decorator==0.5.0

# faiss
RUN conda install -c pytorch faiss-gpu==1.6.3 cudatoolkit=10.0

# nltk
RUN python -m nltk.downloader stopwords -d /usr/share/nltk_data

# spacy
RUN python -m spacy download en_core_web_sm

# sling
RUN pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl

# misc
RUN pip install absl-py==0.15.0

RUN pip install datasets==1.15.1

RUN pip install scikit-learn==0.24.2

# TAPAS
#RUN apt-get -y install protobuf-compiler
#COPY requirements_tapas.txt /tmp/scripts/.
#RUN pip install -r /tmp/scripts/requirements_tapas.txt
#RUN conda install cudatoolkit=10.1
#RUN conda install -c anaconda cudnn=7.6.5
