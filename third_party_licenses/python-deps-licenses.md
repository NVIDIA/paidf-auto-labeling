<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python dependency license report — `paidf-auto-labeling`

This report was generated from inside the published container image using
[`licensecheck`](https://pypi.org/project/licensecheck/), so it reflects the
exact Python packages bundled by the published image (not just the declared
deps in `pyproject.toml`).

## How to reproduce

```bash
IMAGE=nvcr.io/nv-metropolis-dev/metropolis-sdg/paidf-auto-labeling:1.0.0-79167352.mr84

docker run --rm --user 0:0 --entrypoint /bin/bash \
  -v "$(pwd):/host_repo" "$IMAGE" -lc '
    source /opt/venv/bin/activate
    uv pip install --quiet licensecheck
    pip freeze > /tmp/freeze.txt
    # Drop git+ URL deps (licensecheck cannot resolve them without git binary)
    # and strip PyTorch local version suffixes (+cu128 / +cpu) so PyPI resolves.
    grep -vE "@ git\+" /tmp/freeze.txt \
      | sed -E "s/\+cu[0-9]+//; s/\+cpu//" > /tmp/installed.txt
    licensecheck \
      --license Apache-2.0 \
      --format markdown \
      --requirements-paths /tmp/installed.txt \
      --file /host_repo/third_party_licenses/python-deps-licenses.md \
      --zero
  '
```

## Notes on the report

A few entries deserve clarification:

- **`av` (PyAV) — not in the table.** Installed from a git URL
  (`git+https://github.com/PyAV-Org/PyAV@bae50fa...`) so `licensecheck` cannot
  resolve it from PyPI. PyAV is **BSD-3-Clause**. We rebuild PyAV against
  our LGPL-only FFmpeg in the Dockerfile — see `THIRD_PARTY_NOTICES.md`.
- **`rfdetr` — not in the table.** Installed from a GitHub source-tarball URL,
  also skipped by `licensecheck`. RF-DETR is **Apache-2.0**.
- **`pi-heif` shown as LGPLv3.** The installed wheel METADATA actually
  declares **BSD-3-Clause** (`pip show pi-heif` confirms); the LGPLv3
  string comes from the PyPI classifier referencing the bundled `libheif`
  C library. `pi-heif` is an indirect dep pulled in by `roboflow` (a
  transitive dep of RF-DETR); the pipeline does not import it directly,
  and `libheif` is used as a dynamically linked LGPL library (permitted
  under LGPLv3 §4).
- **`OTHER_PROPRIETARY LICENSE` / `NVIDIA PROPRIETARY SOFTWARE` entries** are
  NVIDIA CUDA wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
  `nvidia-nccl-cu12`, …). These are governed by the
  [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) and are
  the standard CUDA runtime wheels installed as transitive deps of
  `torch==2.8.0+cu128`.
- `licensecheck` was run with `--license Apache-2.0`; the `Compatible`
  column reflects compatibility with **distributing the project under
  Apache-2.0**, not the upstream package's own license.

## Info

- program: licensecheck
- version: 2025.1.0
- license: MIT LICENSE

## Project License

APACHE LICENSE

## Packages

Find a list of packages below

|Compatible|Package|
|:--|:--|
|✔|accelerate|
|✔|aiofiles|
|✔|annotated-doc|
|✔|annotated-types|
|✔|antlr4-python3-runtime|
|✔|anyio|
|✔|appdirs|
|✔|asttokens|
|✔|attrs|
|✔|beartype|
|✔|boolean-py|
|✔|boto3|
|✔|botocore|
|✔|bracex|
|✔|cattrs|
|✔|certifi|
|✔|cffi|
|✔|charset-normalizer|
|✔|click|
|✔|cmake|
|✔|contourpy|
|✔|cycler|
|✔|cython|
|✔|decorator|
|✔|defusedxml|
|✔|diffusers|
|✔|distro|
|✔|einops|
|✔|executing|
|✔|fairscale|
|✔|ffmpeg-python|
|✔|fhconfparser|
|✔|filelock|
|✔|filetype|
|✔|filterpy|
|✔|flash-attn|
|✔|fonttools|
|✔|fsspec|
|✔|future|
|✔|h11|
|✔|hf-xet|
|✔|httpcore|
|✔|httpx|
|✔|huggingface-hub|
|✔|idna|
|✔|importlib-metadata|
|✔|ipython|
|✔|ipython-pygments-lexers|
|✔|jedi|
|✔|jinja2|
|✔|jiter|
|✔|jmespath|
|✔|jsonschema|
|✔|jsonschema-specifications|
|✔|kiwisolver|
|✔|lap|
|✔|lark|
|✔|license-expression|
|✔|licensecheck|
|✔|loguru|
|✔|markdown|
|✔|markdown-it-py|
|✔|markupsafe|
|✔|matplotlib|
|✖|matplotlib-inline|
|✔|mdurl|
|✔|mediapy|
|✔|mpmath|
|✔|multi-storage-client|
|✔|networkx|
|✔|ninja|
|✔|numpy|
|✖|nvidia-cublas-cu12|
|✖|nvidia-cuda-cupti-cu12|
|✖|nvidia-cuda-nvrtc-cu12|
|✖|nvidia-cuda-runtime-cu12|
|✖|nvidia-cudnn-cu12|
|✖|nvidia-cufft-cu12|
|✖|nvidia-cufile-cu12|
|✖|nvidia-curand-cu12|
|✖|nvidia-cusolver-cu12|
|✖|nvidia-cusparse-cu12|
|✖|nvidia-cusparselt-cu12|
|✖|nvidia-nccl-cu12|
|✖|nvidia-nvjitlink-cu12|
|✖|nvidia-nvtx-cu12|
|✔|omegaconf|
|✔|openai|
|✔|opencv-python|
|✔|opencv-python-headless|
|✔|opentelemetry-api|
|✔|packaging|
|✔|pandas|
|✔|parso|
|✔|peft|
|✔|pexpect|
|✔|pi-heif|
|✔|pillow|
|✔|pillow-avif-plugin|
|✔|platformdirs|
|✔|polygraphy|
|✔|prettytable|
|✔|prompt-toolkit|
|✔|psutil|
|✔|ptyprocess|
|✔|pure-eval|
|✔|pycocotools|
|✔|pycparser|
|✔|pydantic|
|✔|pydantic-core|
|✔|pygments|
|✔|pyparsing|
|✔|python-dateutil|
|✔|python-dotenv|
|✔|pytz|
|✔|pyyaml|
|✔|referencing|
|✔|regex|
|✔|requests|
|✔|requests-cache|
|✔|requests-toolbelt|
|✔|requirements-parser|
|✔|rf100vl|
|✔|rfdetr|
|✔|rich|
|✔|roboflow|
|✔|rotary-embedding-torch|
|✔|rpds-py|
|✔|s3transfer|
|✔|safetensors|
|✔|scikit-build|
|✔|scipy|
|✔|setuptools|
|✔|shellingham|
|✔|six|
|✔|sniffio|
|✔|stack-data|
|✔|supervision|
|✔|sympy|
|✔|timm|
|✔|tokenizers|
|✔|tomli|
|✔|torch|
|✔|torchaudio|
|✔|torchvision|
|✔|tqdm|
|✔|traitlets|
|✔|transformers|
|✔|triton|
|✔|typer|
|✔|typer-slim|
|✔|typing-extensions|
|✔|typing-inspection|
|✔|tzdata|
|✔|url-normalize|
|✔|urllib3|
|✔|uv|
|✔|wcmatch|
|✔|wcwidth|
|✔|wheel|
|✔|xattr|
|✔|zipp|

### accelerate-1.12.0

- HomePage: https://github.com/huggingface/accelerate
- Author: The HuggingFace team
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 1483618

### aiofiles-23.2.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 46818

### annotated-doc-0.0.4

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 8905

### annotated-types-0.7.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 36458

### antlr4-python3-runtime-4.9.3

- HomePage: http://www.antlr.org
- Author: Eric Vergnaud, Terence Parr, Sam Harwell
- License: BSD
- Compatible: True
- Size: 474074

### anyio-4.12.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 453547

### appdirs-1.4.4

- HomePage: http://github.com/ActiveState/appdirs
- Author: Trent Mick
- License: MIT LICENSE
- Compatible: True
- Size: 34918

### asttokens-3.0.1

- HomePage: https://github.com/gristlabs/asttokens
- Author: Dmitry Sagalovskiy, Grist Labs
- License: APACHE 2.0
- Compatible: True
- Size: 77047

### attrs-23.2.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 202960

### beartype-0.22.9

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 4604232

### boolean-py-5.0

- HomePage: https://github.com/bastikr/boolean.py
- Author: Sebastian Kraemer
- License: BSD-2-CLAUSE
- Compatible: True
- Size: 112060

### boto3-1.42.51

- HomePage: https://github.com/boto/boto3
- Author: Amazon Web Services
- License: APACHE-2.0
- Compatible: True
- Size: 988283

### botocore-1.42.51

- HomePage: https://github.com/boto/botocore
- Author: Amazon Web Services
- License: APACHE-2.0
- Compatible: True
- Size: 18794581

### bracex-2.6

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 29289

### cattrs-24.1.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 233480

### certifi-2026.1.4

- HomePage: https://github.com/certifi/python-certifi
- Author: Kenneth Reitz
- License: MOZILLA PUBLIC LICENSE 2.0 _MPL 2.0_
- Compatible: True
- Size: 278248

### cffi-2.0.0

- HomePage: UNKNOWN
- Author: Armin Rigo, Maciej Fijalkowski
- License: MIT
- Compatible: True
- Size: 728565

### charset-normalizer-3.4.4

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 485878

### click-8.3.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 387747

### cmake-3.31.10

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE;; BSD LICENSE
- Compatible: True
- Size: 67388827

### contourpy-1.3.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 1089258

### cycler-0.12.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 22086

### cython-3.2.4

- HomePage: https://cython.org/
- Author: Robert Bradshaw, Stefan Behnel, David Woods, Greg Ewing, et al.
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 10444579

### decorator-5.2.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 22459

### defusedxml-0.7.1

- HomePage: https://github.com/tiran/defusedxml
- Author: Christian Heimes
- License: PYTHON SOFTWARE FOUNDATION LICENSE
- Compatible: True
- Size: 67382

### diffusers-0.38.0

- HomePage: https://github.com/huggingface/diffusers
- Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/diffusers/graphs/contributors)
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 22939095

### distro-1.9.0

- HomePage: https://github.com/python-distro/distro
- Author: Nir Cohen
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 69038

### einops-0.7.0

- HomePage: UNKNOWN
- Author: Alex Rogozhnikov
- License: MIT LICENSE
- Compatible: True
- Size: 141472

### executing-2.2.1

- HomePage: https://github.com/alexmojaki/executing
- Author: Alex Hall
- License: MIT LICENSE
- Compatible: True
- Size: 94416

### fairscale-0.4.13

- HomePage: UNKNOWN
- Author: Foundational AI Research @ Meta AI
- License: BSD LICENSE
- Compatible: True
- Size: 1063382

### ffmpeg-python-0.2.0

- HomePage: https://github.com/kkroening/ffmpeg-python
- Author: Karl Kroening
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 77314

### fhconfparser-2024.1

- HomePage: https://github.com/FHPythonUtils/FHConfParser
- Author: FredHappyface
- License: MIT LICENSE
- Compatible: True
- Size: 25375

### filelock-3.24.2

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 70694

### filetype-1.2.0

- HomePage: https://github.com/h2non/filetype.py
- Author: Tomas Aparicio
- License: MIT LICENSE
- Compatible: True
- Size: 70637

### filterpy-1.4.5

- HomePage: https://github.com/rlabbe/filterpy
- Author: Roger Labbe
- License: MIT LICENSE
- Compatible: True
- Size: 346128

### flash-attn-2.8.3

- HomePage: https://github.com/Dao-AILab/flash-attention
- Author: Tri Dao
- License: BSD LICENSE
- Compatible: True
- Size: 1002806678

### fonttools-4.61.1

- HomePage: http://github.com/fonttools/fonttools
- Author: Just van Rossum
- License: MIT
- Compatible: True
- Size: 20838619

### fsspec-2026.2.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 732664

### future-1.0.0

- HomePage: https://python-future.org
- Author: Ed Schofield
- License: MIT LICENSE
- Compatible: True
- Size: 1529512

### h11-0.16.0

- HomePage: https://github.com/python-hyper/h11
- Author: Nathaniel J. Smith
- License: MIT LICENSE
- Compatible: True
- Size: 102693

### hf-xet-1.2.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 8327007

### httpcore-1.0.9

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 285071

### httpx-0.28.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 293374

### huggingface-hub-1.3.0

- HomePage: https://github.com/huggingface/huggingface_hub
- Author: Hugging Face, Inc.
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 2132637

### idna-3.7

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 315496

### importlib-metadata-8.7.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE-2.0
- Compatible: True
- Size: 73997

### ipython-9.10.0

- HomePage: UNKNOWN
- Author: The IPython Development Team
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 1991546

### ipython-pygments-lexers-1.1.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 22566

### jedi-0.19.2

- HomePage: https://github.com/davidhalter/jedi
- Author: David Halter
- License: MIT LICENSE
- Compatible: True
- Size: 4319771

### jinja2-3.1.6

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 494554

### jiter-0.13.0

- HomePage: https://github.com/pydantic/jiter/
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 842616

### jmespath-1.1.0

- HomePage: https://github.com/jmespath/jmespath.py
- Author: James Saryerwinnie
- License: MIT LICENSE
- Compatible: True
- Size: 69153

### jsonschema-4.26.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 472557

### jsonschema-specifications-2025.9.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 44039

### kiwisolver-1.4.9

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 5749705

### lap-0.5.12

- HomePage: https://github.com/gatagat/lap
- Author: gatagat, rathaROG, and co.
- License: BSD LICENSE
- Compatible: True
- Size: 2425141

### lark-1.3.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 354349

### license-expression-30.4.4

- HomePage: https://github.com/aboutcode-org/license-expression
- Author: nexB. Inc. and others
- License: APACHE-2.0
- Compatible: True
- Size: 1115065

### licensecheck-2025.1.0

- HomePage: https://github.com/FHPythonUtils/LicenseCheck
- Author: FredHappyface
- License: MIT
- Compatible: True
- Size: 69583

### loguru-0.7.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 223875

### markdown-3.10.2

- HomePage: UNKNOWN
- Author: Manfred Stienstra, Yuri Takhteyev
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 330159

### markdown-it-py-4.0.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 223812

### markupsafe-3.0.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 66242

### matplotlib-3.10.8

- HomePage: UNKNOWN
- Author: John D. Hunter, Michael Droettboom
- License: PYTHON SOFTWARE FOUNDATION LICENSE
- Compatible: True
- Size: 22147576

### matplotlib-inline-0.2.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: 
- Compatible: False
- Size: 20502

### mdurl-0.1.2

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 22522

### mediapy-1.2.5

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 87619

### mpmath-1.3.0

- HomePage: http://mpmath.org/
- Author: Fredrik Johansson
- License: BSD LICENSE
- Compatible: True
- Size: 1942522

### multi-storage-client-0.36.0

- HomePage: UNKNOWN
- Author: NVIDIA Multi-Storage Client Team
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 8695586

### networkx-3.6.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 6978486

### ninja-1.13.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE;; BSD LICENSE
- Compatible: True
- Size: 399593

### numpy-2.2.6

- HomePage: UNKNOWN
- Author: Travis E. Oliphant et al.
- License: BSD LICENSE
- Compatible: True
- Size: 57711977

### nvidia-cublas-cu12-12.8.4.1

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 869569496

### nvidia-cuda-cupti-cu12-12.8.90

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 42514059

### nvidia-cuda-nvrtc-cu12-12.8.93

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 222207464

### nvidia-cuda-runtime-cu12-12.8.90

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 4980951

### nvidia-cudnn-cu12-9.10.2.21

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 1053622200

### nvidia-cufft-cu12-11.3.3.83

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 281137914

### nvidia-cufile-cu12-1.13.1.3

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 3346642

### nvidia-curand-cu12-10.3.9.90

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 138914947

### nvidia-cusolver-cu12-11.7.3.90

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 405117553

### nvidia-cusparse-cu12-12.5.8.93

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 388350763

### nvidia-cusparselt-cu12-0.7.1

- HomePage: https://developer.nvidia.com/cusparselt
- Author: NVIDIA Corporation
- License: NVIDIA PROPRIETARY SOFTWARE
- Compatible: False
- Size: 452023439

### nvidia-nccl-cu12-2.27.3

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 429660825

### nvidia-nvjitlink-cu12-12.8.93

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 94179583

### nvidia-nvtx-cu12-12.8.90

- HomePage: https://developer.nvidia.com/cuda-zone
- Author: Nvidia CUDA Installer Team
- License: OTHER_PROPRIETARY LICENSE
- Compatible: False
- Size: 364234

### omegaconf-2.3.0

- HomePage: https://github.com/omry/omegaconf
- Author: Omry Yadan
- License: BSD LICENSE
- Compatible: True
- Size: 347670

### openai-1.61.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 1807258

### opencv-python-4.12.0.88

- HomePage: https://github.com/opencv/opencv-python
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 84625723

### opencv-python-headless-4.10.0.84

- HomePage: https://github.com/opencv/opencv-python
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: -1

### opentelemetry-api-1.39.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE-2.0
- Compatible: True
- Size: 204746

### packaging-26.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE-2.0;; BSD-2-CLAUSE
- Compatible: True
- Size: 274976

### pandas-2.3.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 43210712

### parso-0.8.6

- HomePage: https://github.com/davidhalter/parso
- Author: David Halter
- License: MIT LICENSE
- Compatible: True
- Size: 351629

### peft-0.18.1

- HomePage: https://github.com/huggingface/peft
- Author: The HuggingFace team
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 1973786

### pexpect-4.9.0

- HomePage: https://pexpect.readthedocs.io/
- Author: Noah Spurrier; Thomas Kluyver; Jeff Quast
- License: ISC LICENSE _ISCL_
- Compatible: True
- Size: 189522

### pi-heif-1.2.0

- HomePage: https://github.com/bigcat88/pillow_heif
- Author: Alexander Piskun
- License: GNU LESSER GENERAL PUBLIC LICENSE V3 _LGPLV3_
- Compatible: True
- Size: 4230223

### pillow-12.2.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT-CMU
- Compatible: True
- Size: 19506106

### pillow-avif-plugin-1.5.5

- HomePage: https://github.com/fdintino/pillow-avif-plugin/
- Author: Frankie Dintino
- License: MIT LICENSE
- Compatible: True
- Size: 13178259

### platformdirs-4.9.6

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 104042

### polygraphy-0.49.26

- HomePage: https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy
- Author: NVIDIA
- License: APACHE 2.0
- Compatible: True
- Size: 1218789

### prettytable-3.17.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 154981

### prompt-toolkit-3.0.52

- HomePage: UNKNOWN
- Author: Jonathan Slenders
- License: BSD LICENSE
- Compatible: True
- Size: 1380539

### psutil-7.2.2

- HomePage: https://github.com/giampaolo/psutil
- Author: Giampaolo Rodola
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 519392

### ptyprocess-0.7.0

- HomePage: https://github.com/pexpect/ptyprocess
- Author: Thomas Kluyver
- License: ISC LICENSE _ISCL_
- Compatible: True
- Size: 39289

### pure-eval-0.2.3

- HomePage: http://github.com/alexmojaki/pure_eval
- Author: Alex Hall
- License: MIT LICENSE
- Compatible: True
- Size: 32052

### pycocotools-2.0.11

- HomePage: https://github.com/ppwwyyxx/cocoapi
- Author: UNKNOWN
- License: FREEBSD
- Compatible: True
- Size: 1412255

### pycparser-3.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 202719

### pydantic-2.12.5

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 1842120

### pydantic-core-2.41.5

- HomePage: https://github.com/pydantic/pydantic-core
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 5096092

### pygments-2.20.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-2-CLAUSE
- Compatible: True
- Size: 4491252

### pyparsing-3.3.2

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 462142

### python-dateutil-2.9.0.post0

- HomePage: https://github.com/dateutil/dateutil
- Author: Gustavo Niemeyer
- License: APACHE SOFTWARE LICENSE;; BSD LICENSE
- Compatible: True
- Size: 439501

### python-dotenv-1.2.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD-3-CLAUSE
- Compatible: True
- Size: 57575

### pytz-2025.2

- HomePage: http://pythonhosted.org/pytz
- Author: Stuart Bishop
- License: MIT LICENSE
- Compatible: True
- Size: 957583

### pyyaml-6.0.3

- HomePage: https://pyyaml.org/
- Author: Kirill Simonov
- License: MIT LICENSE
- Compatible: True
- Size: 2875611

### referencing-0.37.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 113788

### regex-2026.1.15

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE-2.0;; CNRI-PYTHON
- Compatible: True
- Size: 3039699

### requests-2.32.5

- HomePage: https://requests.readthedocs.io
- Author: Kenneth Reitz
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 202504

### requests-cache-1.3.2

- HomePage: UNKNOWN
- Author: Jordan Cook, Roman Haritonov
- License: BSD-2-CLAUSE
- Compatible: True
- Size: 206100

### requests-toolbelt-1.0.0

- HomePage: https://toolbelt.readthedocs.io/
- Author: Ian Cordasco, Cory Benfield
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 142194

### requirements-parser-0.13.0

- HomePage: UNKNOWN
- Author: Paul Horton
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 35870

### rf100vl-1.1.0

- HomePage: UNKNOWN
- Author: Roboflow, Inc.
- License: MIT LICENSE
- Compatible: True
- Size: -1

### rfdetr-1.4.2

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 607144

### rich-14.3.2

- HomePage: https://github.com/Textualize/rich
- Author: Will McGugan
- License: MIT LICENSE
- Compatible: True
- Size: 1235135

### roboflow-1.3.8

- HomePage: https://github.com/roboflow-ai/roboflow-python
- Author: Roboflow
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: -1

### rotary-embedding-torch-0.5.3

- HomePage: https://github.com/lucidrains/rotary-embedding-torch
- Author: Phil Wang
- License: MIT LICENSE
- Compatible: True
- Size: 10956

### rpds-py-0.30.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 1069412

### s3transfer-0.16.0

- HomePage: https://github.com/boto/s3transfer
- Author: Amazon Web Services
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 320916

### safetensors-0.8.0rc0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 1302764

### scikit-build-0.19.0

- HomePage: UNKNOWN
- Author: The scikit-build team
- License: MIT LICENSE
- Compatible: True
- Size: 245224

### scipy-1.17.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 113548120

### setuptools-82.0.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 3402359

### shellingham-1.5.4

- HomePage: https://github.com/sarugaku/shellingham
- Author: Tzu-ping Chung
- License: ISC LICENSE _ISCL_
- Compatible: True
- Size: 17427

### six-1.17.0

- HomePage: https://github.com/benjaminp/six
- Author: Benjamin Peterson
- License: MIT LICENSE
- Compatible: True
- Size: 37542

### sniffio-1.3.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE;; MIT LICENSE
- Compatible: True
- Size: 21891

### stack-data-0.6.3

- HomePage: http://github.com/alexmojaki/stack_data
- Author: Alex Hall
- License: MIT LICENSE
- Compatible: True
- Size: 74354

### supervision-0.27.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 913457

### sympy-1.14.0

- HomePage: https://sympy.org
- Author: SymPy development team
- License: BSD LICENSE
- Compatible: True
- Size: 26696777

### timm-1.0.24

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 8976649

### tokenizers-0.22.2

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: APACHE SOFTWARE LICENSE
- Compatible: True
- Size: 10319440

### tomli-2.4.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 730908

### torch-2.8.0+cu128

- HomePage: https://pytorch.org/
- Author: PyTorch Team
- License: BSD LICENSE
- Compatible: True
- Size: 1694000217

### torchaudio-2.8.0+cu128

- HomePage: https://github.com/pytorch/audio
- Author: Soumith Chintala, David Pollack, Sean Naren, Peter Goldsborough, Moto Hira, Caroline Chen, Jeff Hwang, Zhaoheng Ni, Xiaohui Zhang
- License: BSD LICENSE
- Compatible: True
- Size: 16272608

### torchvision-0.23.0+cu128

- HomePage: https://github.com/pytorch/vision
- Author: PyTorch Core Team
- License: BSD
- Compatible: True
- Size: 23789182

### tqdm-4.67.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT;; MPL-2.0
- Compatible: True
- Size: 222749

### traitlets-5.14.3

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: BSD LICENSE
- Compatible: True
- Size: 314245

### transformers-5.1.0

- HomePage: https://github.com/huggingface/transformers
- Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
- License: APACHE 2.0 LICENSE
- Compatible: True
- Size: 43090673

### triton-3.4.0

- HomePage: https://github.com/triton-lang/triton/
- Author: Philippe Tillet
- License: MIT LICENSE
- Compatible: True
- Size: 564919226

### typer-0.24.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 265930

### typer-slim-0.24.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 5389

### typing-extensions-4.15.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: PSF-2.0
- Compatible: True
- Size: 177708

### typing-inspection-0.4.2

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 52832

### tzdata-2025.3

- HomePage: https://github.com/python/tzdata
- Author: Python Software Foundation
- License: APACHE-2.0
- Compatible: True
- Size: 529733

### url-normalize-3.0.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 31545

### urllib3-2.7.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 428936

### uv-0.11.16

- HomePage: https://pypi.org/project/uv/
- Author: UNKNOWN
- License: APACHE-2.0;; MIT
- Compatible: True
- Size: 60732538

### wcmatch-10.1

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT LICENSE
- Compatible: True
- Size: 149809

### wcwidth-0.6.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 345991

### wheel-0.47.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 92039

### xattr-1.3.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 125558

### zipp-3.23.0

- HomePage: UNKNOWN
- Author: UNKNOWN
- License: MIT
- Compatible: True
- Size: 22363

