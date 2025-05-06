#!/bin/bash
export ENV_VENUS_PROXY=http://zzachzhang:rmdRjCXJAhvOXxhE@vproxy.woa.com:31289
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcos.cn,.myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

export XDG_CACHE=/group/40043/leizizhang/pretrained
export TORCH_HOME=/group/40043/leizizhang/pretrained
export HF_HOME=/group/40043/leizizhang/pretrained

# export XDG_CACHE=/group/40005/leizizhang/pretrained
# export TORCH_HOME=/group/40005/leizizhang/pretrained
# export HF_HOME=/group/40005/leizizhang/pretrained

export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200

pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
pip config set global.trusted-host mirrors.tencent.com

export CACHE_PATH="~/.cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200

which python

CUDA_VISIBLE_DEVICES=1 python app.py

python /group/40034/leizizhang/projects/multi_occupy.py
