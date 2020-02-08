FROM debian:stable

RUN apt-get update

RUN apt-get install r-base r-base-dev -y

RUN Rscript -e "install.packages('tidyverse')"

RUN Rscript -e "install.packages('docopt')"

RUN Rscript -e "install.packages('stringr')"

RUN Rscript -e "install.packages('testthat')"

FROM debian:buster-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

CMD [ "/bin/bash" ]

RUN conda install scikit-learn -y

RUN conda install matplotlib

RUN conda install numpy

RUN conda install pandas

RUN conda install altair -y

RUN conda install seaborn


