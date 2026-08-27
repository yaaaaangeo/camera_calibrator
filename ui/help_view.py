"""
camera_calibrator.ui.help_view
==============================

처음 사용하는 사람이 각 탭의 목적과 비교 기준을 빠르게 이해할 수 있게 돕는
읽기 전용 안내 화면.
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
            여러 모델 중 어떤 결과가 실제 이미지에서 더 안정적인지 비교하거나,
            두 카메라 사이의 상대 R/T를 구하기 위한 Calibration Suite입니다.
            처음 화면에서는 Camera Intrinsic 또는 Camera ↔ Camera 중 하나를 고릅니다.
        </p>

        <div class="note">
            <b>Camera Intrinsic 추천 흐름</b><br>
            이미지 불러오기 → ① Dataset 확인 → ② Coverage 확인 → 캘리브레이션 실행 →
            ③ Outlier로 나쁜 프레임 점검 → ④ Validation/⑤ Error Analysis 확인 →
            ⑦ Model Comparison에서 모델 선택 → ⑨ Export 저장 → 필요하면 ⑩ External Compare.
        </div>

        <div class="note">
            <b>Camera ↔ Camera 추천 흐름</b><br>
            Camera 1/2 Intrinsic 준비 → Stereo pair 이미지 폴더 로드 → 공통 ChArUco ID 매칭 확인 →
            Stereo Calibration 실행 → baseline/R/T/rectification/epipolar validation 확인 → YAML/JSON export.
            기본 stereo calibration은 K1/D1/K2/D2를 고정하고 R/T만 최적화합니다.
        </div>

        <h2>탭별 기능</h2>
        <table>
            <tr><th>탭</th><th>무엇을 보는 곳인가</th><th>판단 기준</th></tr>
            <tr>
                <td><b>① Dataset</b></td>
                <td>캘리브레이션에 사용할 이미지, ROS bag, 실시간 ROS 이미지 토픽을 불러오고 검출 상태를 확인합니다.</td>
                <td>검출 성공 프레임 수가 충분한지, 이미지가 흔들리거나 흐리지 않은지, 같은 포즈만 반복되지 않았는지 봅니다. 실시간 캡처는 최소 50장 이상을 권장합니다.</td>
            </tr>
            <tr>
                <td><b>② Coverage</b></td>
                <td>보드가 이미지 안에서 얼마나 다양한 위치와 크기, 각도로 찍혔는지 봅니다.</td>
                <td>중앙만 찍힌 데이터보다 모서리/가장자리까지 넓게 분포한 데이터가 좋습니다. 왜곡 보정은 특히 이미지 가장자리 정보가 중요합니다.</td>
            </tr>
            <tr>
                <td><b>③ Outlier</b></td>
                <td>전체 결과를 망치는 프레임을 찾아 제거 후보로 보여줍니다.</td>
                <td>재투영 오차가 유난히 큰 프레임, 검출점이 불안정한 프레임, 흐림/부분 가림/패턴 오검출 가능성이 있는 프레임을 확인합니다.</td>
            </tr>
            <tr>
                <td><b>④ Validation</b></td>
                <td>학습에 쓴 이미지가 아니라 검증용 이미지에서 모델이 잘 맞는지 확인합니다.</td>
                <td>Training 오차만 낮고 Validation 오차가 높으면 과적합 가능성이 있습니다. Hold-out 결과가 안정적인 모델을 더 신뢰합니다.</td>
            </tr>
            <tr>
                <td><b>⑤ Error Analysis</b></td>
                <td>오차가 이미지 어디에서 커지는지, 보정 후 직선성이 좋아지는지 시각적으로 확인합니다.</td>
                <td>중앙보다 가장자리 오차가 급격히 커지는지, 특정 방향으로 residual이 몰리는지, undistort 후 직선이 휘지 않는지 봅니다.</td>
            </tr>
            <tr>
                <td><b>⑥ Stability</b></td>
                <td>데이터를 나누거나 반복 계산했을 때 파라미터가 얼마나 흔들리는지 봅니다.</td>
                <td>반복할 때 fx/fy/cx/cy와 왜곡 계수가 크게 흔들리면 데이터가 부족하거나 포즈 다양성이 낮을 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>⑦ Model Comparison</b></td>
                <td>Pinhole, Extended Pinhole, Fisheye 모델을 같은 데이터 기준으로 비교하고 추천 모델을 보여줍니다.</td>
                <td>RMSE만 보지 말고 Validation, 가장자리 오차, 안정성, 모델 복잡도 패널티를 함께 봅니다. 복잡한 모델은 오차가 조금 낮아도 안정성이 낮으면 추천에서 밀릴 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>⑧ Diagnosis</b></td>
                <td>현재 데이터와 결과가 신뢰 가능한지 자동 진단합니다.</td>
                <td>이미지 수 부족, coverage 부족, 모델 실패, 비정상 파라미터, 높은 오차 같은 위험 신호를 확인합니다.</td>
            </tr>
            <tr>
                <td><b>⑨ Export</b></td>
                <td>선택한 모델의 결과를 OpenCV YAML, ROS CameraInfo, JSON, CSV, HTML Report로 저장합니다.</td>
                <td>실제 사용할 시스템이 기대하는 포맷을 고릅니다. 저장 파일은 Output 폴더에 생성되며, 모델명/시간/포맷이 들어간 이름으로 구분됩니다.</td>
            </tr>
            <tr>
                <td><b>⑩ External Compare</b></td>
                <td>외부 툴이나 예전 캘리브레이션 결과와 현재 툴 결과를 같은 기준으로 비교합니다.</td>
                <td>Reference/Candidate 파일 비교, 외부 파라미터 vs 내 모델 비교를 지원합니다. Benchmark 이미지가 없으면 Internal Hold-out으로 정상 비교하고, 별도 Benchmark 이미지를 넣으면 Independent Benchmark 기반 HIGH confidence 비교로 승격됩니다.</td>
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

        <h2>External Compare를 읽는 법</h2>
        <p>
            <b>Reference / Candidate 파일 비교</b>는 두 calibration 파일을 같은 조건으로 재평가합니다.
            Reference는 기준으로 보고 싶은 결과, Candidate는 새로 비교할 결과입니다.
            이름이 Reference라고 해서 항상 더 좋은 것은 아니며, 표의 오차 지표가 실제 우열을 말해줍니다.
        </p>
        <p>
            <b>비교할 외부 파라미터</b>는 다른 툴이나 업체가 준 K/D 값을 직접 넣거나 파일로 불러와
            현재 데이터의 test 프레임에서 다시 평가합니다.
            <b>비교할 내모델</b>은 이 앱에서 계산한 Pinhole/Extended/Fisheye 중 하나를 골라 같은 기준으로 맞붙입니다.
        </p>
        <p>
            <span class="warn">파라미터 차이</span>는 설명용 지표입니다.
            fx, fy, cx, cy, distortion 값이 다르다고 해서 그 자체로 어느 쪽이 더 좋다고 말할 수 없습니다.
            실제 판단은 같은 이미지에서의 재투영 오차, validation 오차, edge 오차, worst-case 지표를 기준으로 합니다.
        </p>
        <p>
            <b>Evaluation Source</b>는 기본적으로 Auto를 쓰면 됩니다.
            Independent Benchmark Dataset이 없으면 Internal Hold-out을 사용하며 Confidence는 LIMITED로 표시됩니다.
            이는 결과가 틀렸다는 뜻이 아니라, 같은 촬영 세션에서 떼어 둔 test frame으로 평가했다는 뜻입니다.
            별도의 Benchmark 이미지를 제공하고 중복/품질 문제가 없으면 Independent Benchmark를 사용하며 Confidence는 HIGH로 표시됩니다.
        </p>
        """
