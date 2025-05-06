export ENV_VENUS_PROXY=http://zzachzhang:rmdRjCXJAhvOXxhE@vproxy.woa.com:31289
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcos.cn,.myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY
source /group/40034/yuxuanbian/Video_Inpainting/set_proxy

. /data/miniconda3/etc/profile.d/conda.sh
which conda

conda create -n cog python=3.10 -y
conda activate cog

pip install -r requirements.txt
