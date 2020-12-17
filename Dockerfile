FROM 281113102427.dkr.ecr.cn-north-1.amazonaws.com.cn/hanalytics/biomindenvironment:2.10.1 AS biomindenvironment
FROM 281113102427.dkr.ecr.cn-north-1.amazonaws.com.cn/hanalytics/tritonserver:2.10.1

USER root:root

ENV LANG C.UTF-8

RUN rm /etc/apt/sources.list && \
    ts_mirror=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ && \
    echo "deb $ts_mirror bionic main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb $ts_mirror bionic-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb $ts_mirror bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb $ts_mirror bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxml2 \
        git \
        python3.6-dev \
        unzip \
        make \
        cmake \
        tree \
        nmap \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN CUDA_TOOLKIT=cuda_10.1.243_418.87.00_linux.run && \
        wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/$CUDA_TOOLKIT && \
        sh $CUDA_TOOLKIT --silent --toolkit && \
        rm $CUDA_TOOLKIT

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    wheel \
    setuptools \
    mock \
    "future>=0.17.1"  \
    "protobuf==3.12.1" \
    flake8 \
    flake8-mypy \
    numpy==1.18.0 \
    pathos==0.2.1 \
    Rx==1.6.1 \
    zmq==0.0.0 \
    pyrsistent==0.14.2 \
    SimpleITK==1.1.0 \
    itk==5.0.1 \
    pydicom==1.2.1 \
    PyYAML==3.12 \
    toolz==0.9.0 \
    palettable==3.1.1 \
    requests-futures==0.9.7 \
    Jinja2==2.10 \
    six==1.11.0 \
    scikit-image==0.16.2 \
    promise==2.2.1 \
    lxml==4.3.0 \
    Pillow==5.4.1 \
    python-keycloak==0.16.0 \
    pyAesCrypt==0.4.3 \
    pyminizip==0.2.3 \
    cryptography==2.4.1 \
    psycopg2==2.7.5 \
    google-cloud-storage==1.23.0 \
    itkwidgets==0.25.3 \
    widgetsnbextension==3.5.1 \
    psutil==5.7.0 \
    line-profiler==3.0.2 \
    "tensorflow-gpu==2.1.0" \
    jupyterlab==1.0.0 \
    opencv-python-headless==3.4.2.17 \
    pandas==0.23.4 \
    pyCLI==2.0.3  \
    diff-match-patch==20200713 \
    watchdog==0.10.3 \
    flask-restful==0.3.8 \
    flask-cors==3.0.9 \
    websockets==8.1

COPY --from=biomindenvironment /opt/tensorRT/python/tensorrtserver-1.7.0-py2.py3-none-linux_x86_64.whl /opt/tensorRT/python/
COPY --from=biomindenvironment /opt/tensorRT/bin/perf_client /opt/tensorRT/bin/perf_client

RUN pip3 install --upgrade /opt/tensorRT/python/tensorrtserver-*.whl

ARG USERNAME=tensorrt-server

RUN usermod -u 1000 $USERNAME \
    && mkdir -p \
        /home/$USERNAME/.vscode-server/extensions \
        /home/$USERNAME/.vscode-server-insiders/extensions \
        /home/$USERNAME/.gnupg \
    && chown -R 1000:1000 \
        /home/$USERNAME

USER 1000:1000
