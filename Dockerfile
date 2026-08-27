# camera_calibrator Dockerfile
# =============================
#
# 이 프로젝트의 README가 스스로 여러 번 경고하는 설치 함정들
# (opencv-python과 opencv-contrib-python 동시 설치 금지, ROS1/ROS2는
# 환경을 source해야 함, PySide6가 여러 시스템 공유 라이브러리를 필요로 함)을
# 매번 새로 겪지 않도록 만든 재현 가능한 환경.
#
# 이 이미지는 주로 두 가지 용도다:
#   1) 헤드리스 CLI 실행 (docker run으로 배치/CI 처리) - 기본 ENTRYPOINT
#   2) 개발 컨테이너의 베이스 (.devcontainer/devcontainer.json이 이 파일을 그대로 사용)
#
# GUI(app.main, PySide6 창)는 컨테이너 안에 디스플레이가 없어 기본적으로
# QT_QPA_PLATFORM=offscreen으로 뜬다(오프스크린 렌더링만 하고 화면엔 안 보임).
# 실제로 창을 띄우고 싶다면 호스트에서 X11을 포워딩해야 한다 (맨 아래 사용법 참고).

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/yaaaaangeo/camera_calibrator"
LABEL description="ChArUco 기반 카메라 캘리브레이션 도구 (CLI 헤드리스 실행 / 개발 컨테이너 겸용)"

# OpenCV(contrib)와 PySide6(Qt) 둘 다 시스템 공유 라이브러리가 필요하다 - import
# 시점에 이 라이브러리들이 없으면 즉시 ImportError로 죽는다. 목록은
# .github/workflows/tests.yml의 "Install Qt runtime dependencies" 스텝과
# 의도적으로 동일하게 맞췄다 - CI에서 검증된 목록을 그대로 재사용해서, "CI에서는
# 되는데 로컬/Docker에서는 안 되는" 괴리를 줄인다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 파일만 먼저 복사 - requirements*.txt가 안 바뀌었으면 소스 코드를 고쳐도
# 이 pip install 레이어가 캐시에서 재사용돼 재빌드가 훨씬 빨라진다.
COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# 컨테이너 안에는 디스플레이가 없으므로 기본값은 offscreen - docker run 시
# `-e QT_QPA_PLATFORM=xcb`로 덮어쓰고 X11을 포워딩하면 실제 창도 띄울 수 있다.
ENV QT_QPA_PLATFORM=offscreen
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 기본은 헤드리스 CLI. 이미지/출력 폴더는 볼륨 마운트로 넘긴다:
#
#   docker build -t camera-calibrator .
#
#   docker run --rm \
#     -v "$(pwd)/photos:/data/photos:ro" \
#     -v "$(pwd)/out:/data/out" \
#     camera-calibrator \
#     --images /data/photos --squares-x 7 --squares-y 5 \
#     --square-size 0.04 --marker-size 0.03 --output-dir /data/out
#
# GUI를 실제로 띄우려면 (Linux 호스트 + X11 기준):
#
#   xhost +local:docker
#   docker run --rm -e QT_QPA_PLATFORM=xcb -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --entrypoint python camera-calibrator -m app.main
#
# 테스트를 컨테이너 안에서 돌리려면:
#
#   docker run --rm -e QT_QPA_PLATFORM=offscreen --entrypoint pytest camera-calibrator -m "not slow"
ENTRYPOINT ["python", "-m", "app.cli"]
CMD ["--help"]
