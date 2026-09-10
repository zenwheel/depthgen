#!/bin/sh
# One-time setup: virtualenv, dependencies, pretrained weights (LaMa inpainter, LPIPS).
#   ./setup.sh              # core install
#   ./setup.sh --with-iw3   # also install nunif/iw3 into the venv (optional reference backend)
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

PY="${PYTHON:-python3}"
WITH_IW3=0
for a in "$@"; do
  case "$a" in
    --with-iw3) WITH_IW3=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [ ! -x .venv/bin/python ]; then
  echo "==> creating virtualenv with $PY"
  "$PY" -m venv .venv
fi
echo "==> installing python dependencies"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

mkdir -p weights
# big-lama traced to TorchScript (from the simple-lama-inpainting project; LaMa itself is Apache-2.0)
if [ ! -f weights/big-lama.pt ]; then
  echo "==> downloading LaMa inpainting weights (~200 MB)"
  curl -L --fail --progress-bar -o weights/big-lama.pt.part \
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
  mv weights/big-lama.pt.part weights/big-lama.pt
fi
# LPIPS needs torchvision's AlexNet; cache it under weights/ instead of ~/.cache
echo "==> caching LPIPS backbone"
TORCH_HOME="$(pwd)/weights/torch" .venv/bin/python -c "
import warnings; warnings.filterwarnings('ignore')
import lpips; lpips.LPIPS(net='alex', verbose=False)
print('lpips ok')"

if [ "$WITH_IW3" = 1 ]; then
  if [ ! -d vendor/nunif ]; then
    echo "==> fetching nunif/iw3 sources (MIT licence)"
    mkdir -p vendor
    git clone -q --depth 1 https://github.com/nagadomi/nunif.git vendor/nunif
  fi
  echo "==> installing nunif requirements"
  .venv/bin/pip install -q -r vendor/nunif/requirements.txt
  echo "==> downloading iw3 models"
  (cd vendor/nunif && ../../.venv/bin/python -m iw3.download_models)
fi

ls -la weights

echo "==> smoke test"
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from spatialize.device import pick_device
from spatialize.infill import available_backends
print('device', pick_device(), 'backends', ', '.join(available_backends()))
"
echo "OK. Use: $(pwd)/bin/spatialize PHOTO.jpg [-o OUT_DIR] [--parallax 2.0] [--infill auto]"
