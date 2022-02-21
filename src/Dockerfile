FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
ADD . /
WORKDIR /
RUN conda env create -f environment.yml
SHELL ["/bin/bash", "-c"]
RUN echo "source activate paddle" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
CMD ["/bin/bash"] 