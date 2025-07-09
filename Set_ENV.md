# Install unsloth
git clone https://github.com/unslothai/unsloth.git

for unsloth build system variance, pls use zhenyuan_linux_27 branch for development
pls refer to repo: https://github.com/leizhenyuan/unsloth

Build: python -m build

Install: pip install ./dist/xxx.tar.gz"[intel-gpu-torch270]" --no-deps
# Install unsloth-zoo

git clone https://github.com/unslothai/unsloth-zoo

Build: python -m build

Install: pip install /dist/xxx.tar.gz
# Pytorch changes for torch
PyTorch need related changes forced using math sdpa, pls refer to https://github.com/leizhenyuan/pytorch/pull/1

Currently, we only support math sdpa, pls add code ‘from torch.nn.attention import SDPBackend, sdpa_kernel’ within unsloth/models/llama.py in line 28, and 
‘            with sdpa_kernel([SDPBackend.MATH]):’  in line 526

# Install BNB for qlora

git clone https://github.com/xiaolil1/bitsandbytes

cmake -DCOMPUTE_BACKEND=xpu -S .

make

pip install -e .
