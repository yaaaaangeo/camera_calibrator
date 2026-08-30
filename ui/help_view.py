"""
camera_calibrator.ui.help_view
==============================

처음 사용하는 사람이 Camera Setup 패널과 각 탭의 목적/판단 기준을 빠르게
이해할 수 있게 돕는 읽기 전용 안내 화면.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from ui.theme import Theme


class HelpView(QWidget):
    """상단 도구 탭 설명서."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet(
            f"QTextBrowser {{ background: {Theme.BG_PRIMARY}; color: {Theme.TEXT_PRIMARY}; "
            f"border: 1px solid {Theme.BORDER}; padding: 14px; }}"
        )
        browser.setHtml(self._html())
        layout.addWidget(browser)

    @staticmethod
    def _html() -> str:
        return """
        <style>
            body { font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; font-size: 13px; line-height: 1.55; }
            h1 { font-size: 22px; margin: 0 0 10px 0; }
            h2 { font-size: 17px; margin: 22px 0 8px 0; }
            p { margin: 6px 0; }
            table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; }
            th, td { border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }
            th { background: #f3f6f8; font-weight: 700; }
            .note { background: #f6f8fa; border: 1px solid #d0d7de; padding: 10px; margin: 10px 0; }
            .good { color: #116329; font-weight: 700; }
            .warn { color: #9a6700; font-weight: 700; }
        </style>

        <h1>Camera Calibration Tool 설명서</h1>
        <p>
            이 도구는 이미지를 모아 카메라 내부 파라미터와 왜곡 계수를 구하고,
            여러 모델 중 어떤 결과가 실제 이미지에서 더 안정적인지 비교하기 위한
            Calibration Suite입니다. 처음 화면에서는 Camera Intrinsic을 고릅니다.
        </p>

        <div class="note">
            <b>Camera Intrinsic 추천 흐름</b><br>
            Camera Setup에서 이미지 불러오기 → ① Dataset에서 Coverage/Diversity/Batch 확인 →
            캘리브레이션 실행 → ② Preview에서 보정 결과 확인 →
            ③ Model Comparison에서 모델 선택 → Camera Setup의 Export 버튼으로 저장.
        </div>

        <h2>Camera Setup 패널</h2>
        <p>
            화면 위쪽의 Camera Setup은 왼쪽(Camera) / 중앙(Pattern) / 오른쪽(Actions)
            3열로 구성됩니다.
        </p>
        <table>
            <tr><th>열</th><th>구성</th></tr>
            <tr>
                <td><b>왼쪽 (Camera)</b></td>
                <td>
                    Camera Name(Library 분류용 이름) →
                    <b>INPUT</b>(실시간 / rosbag / 이미지 3버튼 - 실시간은 ROS1/ROS2 이미지
                    토픽을 실시간 구독해 직접 캡처, rosbag은 ROS1 .bag·ROS2 .db3/.mcap
                    로그에서 이미지 토픽을 뽑아 자동 추출, 이미지는 jpg/png/bmp 파일을 직접
                    선택) → <b>해상도 확인</b>(JPEG/PNG 이미지 한 장을 실제로 디코딩해 크기를
                    확인하고 Width/Height에 자동 반영) → Width/Height.
                </td>
            </tr>
            <tr>
                <td><b>중앙 (Pattern)</b></td>
                <td><b>Calibration method</b>(Standard 또는 Object-Releasing, 아래
                    "Advanced Calibration" 참고) → Target-specific fields such as
                    Columns/Rows, Square Size, Center Spacing, Tag Pitch, Marker Size,
                    Dictionary/Tag Family, Grid type(Circle Grid), AprilGrid variant.</td>
            </tr>
            <tr>
                <td><b>오른쪽 (Actions)</b></td>
                <td>
                    Rational model 사용(k4~k6) 체크박스 → <b>캘리브레이션 실행</b> →
                    <b>Export</b>(계산된 모델을 골라 OpenCV YAML로 저장) →
                    <b>취소</b>(코너 검출/모델 계산이 진행 중일 때 즉시 중단하고, 원하는
                    데이터/설정으로 다시 실행할 수 있는 상태로 되돌립니다).
                </td>
            </tr>
        </table>

        <div class="note">
            <b>Advanced Calibration (Object-Releasing)</b><br>
            Calibration method를 Object-Releasing으로 바꾸면, Standard 4모델과는
            별도로 <code>cv2.calibrateCameraRO</code> 기반 결과를 함께 계산합니다 -
            카메라 파라미터뿐 아니라 캘리브레이션 타겟 형상 자체도 함께 추정해,
            타겟 인쇄/부착 오차까지 보정하고 싶은 고정밀 캘리브레이션에 씁니다.
            <b>Checkerboard/Circle Grid만 지원</b>하며(ChArUco/AprilGrid는 부분
            검출이 흔해 지원하지 않음), 타겟 전체가 매 프레임 동일한 순서로 빠짐없이
            검출된 프레임(full-board)만 사용합니다. 결과는 ③ Model Comparison 탭
            아래 <b>Advanced Calibration</b> 패널에 전용 Hold-out Validation과
            Standard Brown-Conrady와의 공정 비교(같은 데이터셋, 같은 train/test
            분할)와 함께 표시되며, Standard의 Hold-out/AIC/BIC와는 섞이지 않습니다.
        </div>

        <h2>탭별 기능</h2>
        <table>
            <tr><th>탭</th><th>무엇을 보는 곳인가</th><th>판단 기준</th></tr>
            <tr>
                <td><b>① Dataset</b></td>
                <td>캘리브레이션에 사용할 이미지의 검출 상태를 확인합니다.
                    위에서부터 Coverage Map(보드가 이미지 안에서 얼마나 다양한 위치/크기/각도로 찍혔는지),
                    Dataset Diversity(다양성 점수 + Overall Dataset Score), Batch(이미지별 상태/코너 수/
                    재투영 오차/품질 점수)를 한 화면에서 확인합니다.</td>
                <td>검출 성공 프레임 수가 충분한지, 이미지가 흔들리거나 흐리지 않은지, 같은 포즈만 반복되지 않았는지 봅니다.
                    중앙만 찍힌 데이터보다 모서리/가장자리까지 넓게 분포한 데이터가 좋습니다. 실시간 캡처는 최소 50장 이상을 권장합니다.</td>
            </tr>
            <tr>
                <td><b>② Preview</b></td>
                <td>이미지 하나를 골라 원본과 (왜곡 보정 후 + Straightness 오버레이)를 나란히 보여줍니다.
                    오른쪽 패널의 초록~빨강 선은 보드의 행/열이 보정 후 얼마나 곧아졌는지를 나타내고,
                    이미지 아래에는 보정 전/후 Line Straightness 수치가 함께 표시됩니다.</td>
                <td>중앙보다 가장자리에서 선이 더 휘는지(빨강에 가까운지), 보정 전/후 수치가
                    얼마나 개선됐는지 봅니다. 0.5px 이하면 방사 왜곡이 사실상 제거된 것으로 봅니다.</td>
            </tr>
            <tr>
                <td><b>③ Model Comparison</b></td>
                <td>Ideal Pinhole, Brown-Conrady, Rational, Fisheye 모델을 같은 데이터 기준으로 비교하고 추천 모델을 보여줍니다.</td>
                <td>RMSE만 보지 말고 Validation, 가장자리 오차, 안정성, 모델 복잡도 패널티를 함께 봅니다. 복잡한 모델은 오차가 조금 낮아도 안정성이 낮으면 추천에서 밀릴 수 있습니다.</td>
            </tr>
        </table>

        <h2>비교표에서 쓰는 주요 기준</h2>
        <table>
            <tr><th>지표</th><th>의미</th><th>해석</th></tr>
            <tr>
                <td><b>Reprojection Error</b></td>
                <td>실제 검출된 코너 위치와 계산된 카메라 모델이 다시 투영한 코너 위치 사이의 픽셀 거리입니다.</td>
                <td><span class="good">낮을수록 좋습니다.</span> 다만 학습 이미지에서만 낮은 값은 과적합일 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>RMSE</b></td>
                <td>오차 제곱 평균의 제곱근입니다. 큰 오차 프레임의 영향을 비교적 크게 받습니다.</td>
                <td>전체 대표 오차로 보기 좋지만, Max/P95와 같이 봐야 합니다.</td>
            </tr>
            <tr>
                <td><b>Mean / Median</b></td>
                <td>Mean은 평균, Median은 중앙값입니다.</td>
                <td>Mean이 Median보다 많이 크면 일부 나쁜 프레임이나 영역이 평균을 끌어올리고 있을 가능성이 있습니다.</td>
            </tr>
            <tr>
                <td><b>P95 / P99 / Max</b></td>
                <td>오차가 큰 쪽의 꼬리를 보는 지표입니다. P95는 95% 지점, P99는 99% 지점, Max는 최악값입니다.</td>
                <td>실사용 안정성을 볼 때 중요합니다. 평균은 낮아도 P99/Max가 크면 특정 영역에서 보정이 깨질 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>Center / Middle / Edge</b></td>
                <td>이미지 중심, 중간 반경, 가장자리 영역을 나누어 오차를 계산합니다.</td>
                <td>광각/왜곡 렌즈는 Edge 오차가 특히 중요합니다. 중앙만 좋은 모델은 실제 보정에서 부족할 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>AIC / BIC</b></td>
                <td>오차와 모델 복잡도를 함께 보는 정보 기준입니다.</td>
                <td><span class="good">낮을수록 좋습니다.</span> 계수가 많은 모델이 단순히 파라미터 수로 이기는 것을 막는 보정값입니다.</td>
            </tr>
            <tr>
                <td><b>K-fold / Hold-out</b></td>
                <td>데이터 일부를 검증용으로 떼어 모델의 일반화 성능을 확인합니다.</td>
                <td>여기서도 안정적으로 낮은 오차를 내는 모델이 실제 사용에 더 안전합니다.</td>
            </tr>
            <tr>
                <td><b>Bootstrap / Stability</b></td>
                <td>데이터를 여러 번 다시 뽑아 계산했을 때 결과가 얼마나 흔들리는지 봅니다.</td>
                <td>오차가 낮아도 파라미터가 크게 흔들리면 데이터가 부족하거나 모델이 과하게 복잡할 수 있습니다.</td>
            </tr>
        </table>
        """
