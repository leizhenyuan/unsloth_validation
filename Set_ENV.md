git clone https://github.com/unslothai/unsloth.git
git checkout zhenyuan_linux_27

Build: python -m build
Install: pip install ./dist/xxx.tar.gz"[intel-gpu-torch270]"

git clone https://github.com/unslothai/unsloth-zoo

Build: python -m build
Install: pip install /dist/xxx.tar.gz

PyTorch need related changes forced using math sdpa, pls refer to https://github.com/leizhenyuan/pytorch/pull/1

Currently, we only support math sdpa, pls add code ‘from torch.nn.attention import SDPBackend, sdpa_kernel’ within unsloth/models/llama.py in line 28, and 
‘            with sdpa_kernel([SDPBackend.MATH]):’  in line 469

