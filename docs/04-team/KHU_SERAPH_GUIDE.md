# 🚀 KHU Seraph 서버 GPU 사용을 위한 완벽 가이드

### 1. 시작하며: 이 문서의 목표

이 문서는 우리 팀이 **경희대학교 Seraph HPC 서버**에 처음 접속하여, **VSCode**로 코드를 효율적으로 관리하고, **GPU를 할당받아 PyTorch 딥러닝 모델을 학습**시키는 전 과정을 안내합니다.

복잡한 서버 환경에 대한 두려움을 없애고, 누구나 쉽게 따라 할 수 있도록 모든 단계와 핵심 개념을 상세히 설명했습니다. 이 가이드만 있다면, 우리 모두가 동일한 환경에서 막힘없이 연구를 진행할 수 있을 것입니다.

---

### 2. Part 1: 내 컴퓨터(Mac/Windows) 사전 준비

원활한 원격 개발 환경을 구축하기 위해, 내 컴퓨터에 두 가지 필수 프로그램을 설치해야 합니다.

*   **Visual Studio Code (VSCode)**: 우리의 주력 코드 편집기입니다. 강력한 확장 기능으로 원격 서버에서도 내 컴퓨터처럼 코드를 작성할 수 있게 해줍니다. [공식 홈페이지에서 다운로드](https://code.visualstudio.com/)
*   **터미널 프로그램**:
    *   **Mac 사용자**: 기본 터미널도 좋지만, 다양한 편의 기능을 제공하는 [iTerm2](https://iterm2.com/) 사용을 강력히 추천합니다.
    *   **Windows 사용자**: VSCode 내장 터미널을 사용하거나, [Windows Terminal](https://apps.microsoft.com/detail/9N0DX20HK701?hl=ko-kr&gl=KR)을 설치하면 좋습니다.

---

### 3. Part 2: VSCode와 Seraph 서버, 첫 만남 설정하기

> **핵심 개념: SFTP (Secure File Transfer Protocol)**
> 이 방식은 내 컴퓨터의 프로젝트 폴더와 서버의 특정 폴더를 '연결'하는 것입니다. VSCode에서 파일을 저장(`Ctrl+S` 또는 `Cmd+S`)하면, 그 즉시 변경 내용이 서버로 자동 전송(업로드)됩니다.

#### **단계 1: VSCode 확장 프로그램 설치**

1.  VSCode를 실행합니다.
2.  왼쪽 사이드바에서 **네모 블록 모양의 Extensions 아이콘** (단축키: `Ctrl+Shift+X` 또는 `Cmd+Shift+X`)을 클릭합니다.
3.  검색창에 `SFTP`를 입력하고, **Natizyskunk**가 만든 버전을 찾아 `Install` 버튼을 눌러 설치합니다.

#### **단계 2: 프로젝트 폴더 생성 및 SFTP 설정**

1.  내 컴퓨터의 원하는 위치(예: 바탕화면, 문서)에 이번 프로젝트를 위한 새 폴더를 만듭니다. (예: `Quant-Dream-Project`)
2.  VSCode에서 `File > Open Folder...`를 통해 방금 만든 폴더를 엽니다.
3.  VSCode에서 **Command Palette**를 엽니다. (단축키: `Ctrl+Shift+P` 또는 `Cmd+Shift+P`)
4.  `SFTP: Config`를 입력하고 엔터를 누르면, 현재 폴더 내에 `.vscode/sftp.json` 파일이 자동으로 생성됩니다.
5.  생성된 `sftp.json` 파일의 **모든 내용을 지우고, 아래의 설정 코드를 그대로 붙여넣으세요.**

```json
{
    "name": "KHU Seraph",
    "host": "moana.khu.ac.kr",
    "protocol": "sftp",
    "port": 30080,
    "username": "여기에_본인의_Seraph_ID를_입력하세요",
    "password": "여기에_본인의_비밀번호를_입력하세요",
    "remotePath": "/data/여기에_본인의_Seraph_ID를_입력하세요/workspace",
    "uploadOnSave": true,
    "watcher": {
        "files": "**/*",
        "autoUpload": true,
        "autoDelete": true
    },
    "ignore": [
        ".vscode",
        ".git",
        ".DS_Store"
    ]
}
```

> **❗️ 중요 설정 설명**
> *   `username` / `password`: 본인의 Seraph 서버 아이디와 비밀번호를 정확히 입력합니다.
> *   `remotePath`: 서버에 생성될 우리의 '작업 공간' 경로입니다. 아이디 부분을 본인 것으로 수정해주세요.
> *   `uploadOnSave: true`: **이 설정의 핵심입니다.** 파일을 저장할 때마다 자동으로 서버에 업로드해주는 기능입니다.

---

### 4. Part 3: 나만의 격리된 연구 환경, Conda 가상환경 만들기

> **핵심 개념: 왜 가상환경을 써야 할까?**
> 서버의 기본 환경(`base`)은 모든 사용자가 공유하는 '공용 주방'과 같습니다. 여러 사람이 다양한 재료(라이브러리)를 마구 섞어두면 충돌이 일어나기 쉽습니다. **가상환경**은 나만의 '개인 요리 공간'을 만드는 것과 같습니다. 다른 사람의 영향을 받지 않고, 내 프로젝트에 필요한 재료들만 깔끔하게 설치하여 안정적인 연구 환경을 보장합니다.

#### **단계 1: 터미널로 Seraph 서버 접속**

터미널을 열고, 아래 명령어를 입력해 서버에 접속합니다.

```bash
# 형식: ssh [아이디]@[서버주소] -p [포트번호]
ssh s202xxxxx@moana.khu.ac.kr -p 30080
```

#### **단계 2: PyTorch 전용 가상환경 생성**

1.  접속이 완료되면, `qd_torch_env` 라는 이름의 새 가상환경을 **Python 3.11 버전**으로 생성합니다.

    ```bash
    conda create -n qd_torch_env python=3.11
    ```
2.  설치 과정에서 `Proceed ([y]/n)?` 질문이 나오면 `y`를 입력하고 엔터를 누릅니다.

#### **단계 3: 가상환경 활성화 및 PyTorch 설치**

1.  방금 만든 '개인 요리 공간'으로 들어갑니다.

    ```bash
    conda activate qd_torch_env
    ```
    명령어 줄 맨 앞에 `(qd_torch_env)` 라는 표시가 생기면 성공적으로 활성화된 것입니다.
2.  활성화된 환경 안에, **CUDA 12.1 버전과 호환되는 PyTorch**를 설치합니다. 이 버전이 Seraph 서버의 하드웨어 드라이버와 가장 안정적으로 호환됩니다.

    ```bash
    # PyTorch 공식 홈페이지의 추천 설치 명령어입니다.
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
3.  마찬가지로 `y`를 눌러 설치를 진행합니다. 설치에는 몇 분 정도 시간이 소요될 수 있습니다.

---

### 5. Part 4: GPU 할당받고 코드 실행하기 (실전)

> **핵심 개념: 로그인 노드 vs 컴퓨팅 노드**
> *   **로그인 노드 (`moana-master`)**: 우리가 `ssh`로 처음 접속하는 '도서관 로비' 같은 곳입니다. 여기서는 파일 관리, 가상환경 설정 등 가벼운 작업만 수행합니다. **여기엔 GPU가 없습니다.**
> *   **컴퓨팅 노드 (`moana-y1`, `moana-y6` 등)**: `srun` 명령으로 자원을 할당받아 들어가는 'GPU 열람실'입니다. 실제 모델 학습과 같은 무거운 계산은 반드시 이곳에서 해야 합니다. **GPU는 여기에만 있습니다.**

#### **단계 1: VSCode에서 테스트 코드 작성**

VSCode의 프로젝트 폴더에 `gpu_test.py` 파일을 새로 만들고 아래 코드를 붙여넣습니다.

```python
# gpu_test.py
import torch
import os

print("========== 딥러닝 서버 환경 정보 ==========")
# 현재 접속된 노드(서버)의 이름을 출력합니다.
# 'moana-y'로 시작해야 GPU 노드입니다.
print(f"✅ 현재 노드: {os.uname().nodename}")
print("========================================")

# PyTorch 및 CUDA 사용 가능 여부를 확인합니다.
print(f"✅ PyTorch 버전: {torch.__version__}")
print(f"✅ CUDA 사용 가능 여부: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"✅ 현재 할당된 GPU ID: {gpu_id}")
    print(f"✅ 현재 할당된 GPU 이름: {gpu_name}")
    
    print("\n🎉 GPU를 사용하여 텐서 연산을 시작합니다!")
    # GPU 메모리에 두 개의 텐서를 생성합니다.
    tensor_a = torch.randn(3, 3).to('cuda')
    tensor_b = torch.randn(3, 3).to('cuda')
    
    print("텐서 A (on GPU):\n", tensor_a)
    print("텐서 B (on GPU):\n", tensor_b)
    print("\nGPU 연산 결과 (A + B):\n", tensor_a + tensor_b)
    
else:
    print("\n❗️ 경고: GPU를 사용할 수 없습니다. srun으로 GPU 노드에 접속했는지 확인하세요.")
```
코드를 작성하고 저장(`Ctrl+S` 또는 `Cmd+S`)하면, SFTP 설정에 따라 자동으로 서버에 업로드됩니다.

#### **단계 2: 터미널에서 GPU 할당 및 코드 실행**

서버에 접속된 터미널에서, 아래의 **황금 순서**를 절대 잊지 말고 차례대로 실행합니다.

```bash
# 1. '도서관 로비'에서 내 개인 작업실(가상환경) 들어가기
conda activate qd_torch_env

# 2. 'GPU 열람실'(컴퓨팅 노드) 자리 예약하고 들어가기
# --gres=gpu:1 -> GPU 1개를 할당해달라는 요청
# -p debug_ce_ugrad -> 학부생이 사용하는 파티션(그룹)
srun --gres=gpu:1 -p debug_ce_ugrad --pty bash

# 3. ❗️가장 중요❗️ 'GPU 열람실'에 들어왔으니, 다시 한번 내 개인 작업실 열기
conda activate qd_torch_env

# 4. 코드가 저장된 내 작업 공간으로 이동
# sftp.json의 remotePath와 동일한 경로입니다.
cd /data/본인의_Seraph_ID/workspace

# 5. 드디어 파이썬 코드 실행!
python gpu_test.py
```

> **성공 화면**: 터미널에 `CUDA 사용 가능 여부: True` 라는 메시지와 함께, 할당받은 GPU의 이름과 텐서 연산 결과가 출력되면 완벽하게 성공한 것입니다!

---

### 6. 문제 해결 (FAQ)

*   **Q: `torch.cuda.is_available()`가 `False`로 나와요!**
    *   **A:** 99%는 **`srun`으로 GPU 노드에 진입한 후, `conda activate qd_torch_env`를 다시 실행하지 않았기 때문**입니다. 'GPU 열람실'에 들어가면 새로운 쉘이 열리므로, 반드시 가상환경을 다시 활성화해야 합니다.
*   **Q: `ModuleNotFoundError: No module named 'torch'` 에러가 나요.**
    *   **A:** 현재 쉘에서 `conda activate qd_torch_env` 명령을 실행했는지 확인하세요. `(qd_torch_env)` 표시가 없다면 가상환경에 들어가지 않은 것입니다.
*   **Q: `srun: command not found` 에러가 나요.**
    *   **A:** `ssh`로 Seraph 서버에 정상적으로 접속했는지 확인하세요. `srun`은 Seraph 서버 내에서만 작동하는 명령어입니다.

이 가이드가 우리 팀의 원활한 연구 활동에 든든한 발판이 되기를 바랍니다. 궁금한 점이 있다면 언제든지 팀 채널에 질문해주세요! 🚀