FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER Harijaona Ravelondrina <hravelondrina@smartpredict.io>

# Export the CUDA evironment variables manually
ENV CUDA_ROOT=/usr/local/cuda-9.0
ENV CUDA_HOME=/usr/local/cuda-9.0
ENV CUDA_PATH=/usr/local/cuda-9.0
ENV CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so.7
ENV PATH=$PATH:/usr/local/cuda-9.0/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64
ENV TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"


# Set CUDA in etc config file
RUN echo export CUDA_ROOT=/usr/local/cuda-9.0 >>/etc/profile && \
	echo export CUDA_ROOT=/usr/local/cuda-9.0 >>/etc/bash.bashrc && \
	echo export CUDA_HOME=/usr/local/cuda-9.0 >>/etc/profile && \
	echo export CUDA_HOME=/usr/local/cuda-9.0 >>/etc/bash.bashrc && \
	echo export CUDA_PATH=/usr/local/cuda-9.0 >>/etc/profile && \
	echo export CUDA_PATH=/usr/local/cuda-9.0 >>/etc/bash.bashrc && \
#	echo export CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so.7 >>/etc/profile && \
#	echo export CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so.7 >>/etc/bash.bashrc && \
	echo export PATH=$PATH:/usr/local/cuda-9.0/bin >>/etc/profile && \
	echo export PATH=$PATH:/usr/local/cuda-9.0/bin >>/etc/bash.bashrc && \
#	echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64 >>/etc/profile && \
#	echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64 >>/etc/bash.bashrc && \
	echo export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__" >>/etc/profile && \
	echo export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__" >>/etc/bash.bashrc

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
		apt-utils git curl vim unzip openssh-client wget \
		build-essential cmake libopenblas-dev libcurl3-dev \
		libfreetype6-dev libhdf5-dev libhdf5-serial-dev libpng12-dev libzmq3-dev pkg-config \
		python-dev rsync software-properties-common unzip zip \
		zlib1g-dev libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev \
		libc6-dev libbz2-dev bc g++ gfortran libffi-dev libfreetype6-dev libjpeg-dev \
		liblcms2-dev libopenblas-dev liblapack-dev libpng12-dev libssl-dev \
		libtiff5-dev libwebp-dev libzmq3-dev nano qt5-default libvtk6-dev libpng-dev libjasper-dev \
		libopenexr-dev libgdal-dev libdc1394-22-dev libavcodec-dev libavformat-dev \
		libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
		libopencore-amrnb-dev libv4l-dev libxine2-dev libtbb-dev libeigen3-dev python-dev \
		python-tk python-numpy python3-dev python3-numpy ant default-jdk doxygen && \
	apt-get clean && \
	apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

#
# Python 3.5
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get update && apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases && \
    rm -rf /var/lib/apt/lists/*

# Add SNI support to Python
RUN pip3 --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Pillow and it's dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow && \
    rm -rf /var/lib/apt/lists/*
	
# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    	numpy scipy sklearn scikit-image pandas matplotlib Cython requests \
		nose h5py seaborn plotly sympy tqdm

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install --upgrade ipython && \
	pip3 --no-cache-dir install \
		Cython ipykernel path.py pygments six sphinx wheel zmq && \
	python3.5 -m ipykernel.kernelspec

#
# Jupyter Notebook
#
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN pip3 --no-cache-dir install jupyter jupyterlab && \
    mkdir /root/.jupyter 

# Set password for jupyter
RUN echo export PASSWORD="ZKg8nYn2C]xTB*jT2J*u)7M"*XRgn7{ >>/etc/profile && \
	echo export PASSWORD="ZKg8nYn2C]xTB*jT2J*u)7M"*XRgn7{ >>/etc/bash.bashrc

# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/
EXPOSE 8888

ENV CUDNN_VERSION 7.3.1.20
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
		libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
		libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 \
		cuda-command-line-tools-9-0 \
		cuda-cublas-dev-9-0 \
		cuda-cudart-dev-9-0 \
		cuda-cufft-dev-9-0 \
		cuda-curand-dev-9-0 \
		cuda-cusolver-dev-9-0 \
		cuda-cusparse-dev-9-0 \
		libnccl2 \
		libnccl-dev && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

#
# Tensorflow 1.11.0 - GPU
#
RUN pip3 install --no-cache-dir --upgrade tensorflow-gpu 

# Expose port for TensorBoard
EXPOSE 6006

#
# OpenCV 3.4.1
#
# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall && \
    rm -rf /var/lib/apt/lists/*
	
# Get source from github
RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          .. && \
    make -j"$(nproc)" && \
    make install

#
# Caffe
#
# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
    libhdf5-serial-dev protobuf-compiler liblmdb-dev libgoogle-glog-dev \
    libboost-all-dev && \
    pip3 install lmdb && \
    rm -rf /var/lib/apt/lists/*

# Get source. Use master branch because the latest stable release (rc3) misses critical fixes.
RUN git clone -b master --depth 1 https://github.com/BVLC/caffe.git /usr/local/src/caffe
# Python dependencies
RUN pip3 --no-cache-dir install -r /usr/local/src/caffe/python/requirements.txt
# Compile
RUN cd /usr/local/src/caffe && mkdir build && cd build && \
    cmake -D CPU_ONLY=ON -D python_version=3 -D BLAS=open -D USE_OPENCV=ON .. && \
    make -j"$(nproc)" all && \
    make install
# Enivronment variables
ENV PYTHONPATH=/usr/local/src/caffe/python:$PYTHONPATH \
	PATH=/usr/local/src/caffe/build/tools:$PATH
# Fix: old version of python-dateutil breaks caffe. Update it.
RUN pip3 install --no-cache-dir python-dateutil --upgrade

#
# Keras 2.2.4
#
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras

#
# PyTorch 0.4.1
#
RUN pip3 install torch && \
    pip3 install torchvision

#
# PyCocoTools
#
# Using a fork of the original that has a fix for Python 3.
# I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# but it doesn't seem to be active anymore.
RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

# Export the LUA evironment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
	LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
	PATH=/root/torch/install/bin:$PATH \
	LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install the latest versions of nn, cutorch, cunn, cuDNN bindings and iTorch
RUN luarocks install nn && \
	luarocks install cutorch && \
	luarocks install cunn && \
    luarocks install loadcaffe && \
	\
	cd /root && git clone https://github.com/soumith/cudnn.torch.git && cd cudnn.torch && \
	git checkout R4 && \
	luarocks make && \
	\
	cd /root && git clone https://github.com/facebook/iTorch.git && \
	cd iTorch && \
	luarocks make

WORKDIR "/root"
CMD ["/bin/bash"]