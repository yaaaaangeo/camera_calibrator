"""
calibration.windshield.reflection_suppression.model
==============================================================

STEP 7 - Small U-Net for reflection decomposition.

    RGB Image (3ch, BGR/OpenCV convention throughout)
        -> Encoder(32,64,128,256) with skip connections
        -> Decoder(128,64,32)
        -> Output(4ch) = Reflection Layer(3ch, sigmoid) + Alpha(1ch, sigmoid)

이번 STEP 7 첫 구현에서는 architecture search를 하지 않는다 - encoder/decoder
채널 수와 activation만 config로 바꿀 수 있다(사용자 스펙 9/10번, "Transformer/
GAN/Diffusion 금지, Lightweight U-Net 하나로 검증").

Lazy PyTorch import(calibration.windshield.neural_residual과 완전히 동일한
원칙, 사용자 스펙 12/13번): 이 모듈은 최상단에서 `import torch`를 실행하지
않는다. `torch`/`nn`은 `None`으로 시작하고, `_require_torch()`가 처음
호출되는 시점에만 import된다. `nn.Module`을 상속하는 클래스 정의 자체도
torch가 있어야만 평가 가능하므로, 클래스 정의를 `_get_unet_class()` 함수
안으로 미뤄서 모듈 import 시점에는 절대 평가되지 않게 한다."""

from __future__ import annotations

from typing import Optional

from calibration.windshield.reflection_suppression.config import (
    DEFAULT_ACTIVATION,
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_ENCODER_CHANNELS,
)

torch = None  # type: ignore[assignment]
nn = None  # type: ignore[assignment]
_TORCH_IMPORT_ERROR: Optional[BaseException] = None
_UNetClass = None


def _require_torch() -> None:
    """PyTorch를 실제로 필요로 하는 첫 호출 시점에만 import한다(진짜 lazy
    import) - 이미 로드됐으면 즉시 반환한다."""
    global torch, nn, _TORCH_IMPORT_ERROR
    if torch is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
    except ImportError as e:
        _TORCH_IMPORT_ERROR = e
        raise ImportError(
            "Reflection Suppression 모델을 사용하려면 PyTorch가 필요합니다. "
            '`pip install torch` (또는 `pip install -e ".[neural]"`, STEP 5 Neural '
            "Residual과 같은 optional extra)로 설치한 뒤 다시 시도하세요. Reflection "
            "Evaluation(STEP 6)은 PyTorch 없이도 정상 동작합니다.\n\n"
            "참고(Windows/Anaconda): numpy(Intel MKL)와 PyTorch가 OpenMP 런타임을 "
            "중복 로드해 충돌하면 'OMP: Error #15'로 프로세스가 종료될 수 있습니다 - "
            "이 경우 KMP_DUPLICATE_LIB_OK=TRUE 환경변수를 직접 설정하세요."
        ) from e
    torch = _torch
    nn = _nn


def _activation_module(name: str):
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.SiLU()


def _conv_block(in_ch: int, out_ch: int, activation: str):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), _activation_module(activation),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), _activation_module(activation),
    )


def _get_unet_class():
    """`nn.Module`을 상속하는 클래스 정의를 torch가 실제로 로드된 뒤에만
    평가한다(모듈 최상단에서 클래스를 정의하면 그 자체가 torch를 요구하게
    되므로) - 한 번만 만들고 캐시한다."""
    global _UNetClass
    _require_torch()
    if _UNetClass is not None:
        return _UNetClass

    class ReflectionSuppressionUNet(nn.Module):
        """Encoder(3 downsample) -> Bottleneck -> Decoder(3 upsample, skip
        connection) -> 1x1 conv -> (Reflection RGB sigmoid, Alpha sigmoid).
        임의 해상도 입력을 지원하기 위해 8의 배수로 reflect-pad했다가 다시
        crop한다(3단 downsample = 2^3 = 8)."""

        def __init__(
            self,
            encoder_channels: list[int] = DEFAULT_ENCODER_CHANNELS,
            decoder_channels: list[int] = DEFAULT_DECODER_CHANNELS,
            activation: str = DEFAULT_ACTIVATION,
        ):
            super().__init__()
            if len(encoder_channels) != 4:
                raise ValueError(f"encoder_channels must have exactly 4 stages, got {encoder_channels}")
            if len(decoder_channels) != 3:
                raise ValueError(f"decoder_channels must have exactly 3 stages, got {decoder_channels}")
            c1, c2, c3, c_bottleneck = encoder_channels
            d3, d2, d1 = decoder_channels

            self.enc1 = _conv_block(3, c1, activation)
            self.enc2 = _conv_block(c1, c2, activation)
            self.enc3 = _conv_block(c2, c3, activation)
            self.bottleneck = _conv_block(c3, c_bottleneck, activation)
            self.pool = nn.MaxPool2d(2)

            self.dec3 = _conv_block(c_bottleneck + c3, d3, activation)
            self.dec2 = _conv_block(d3 + c2, d2, activation)
            self.dec1 = _conv_block(d2 + c1, d1, activation)
            self.out_conv = nn.Conv2d(d1, 4, kernel_size=1)

        @staticmethod
        def _upsample_to(x, target):
            return nn.functional.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

        @staticmethod
        def _pad_to_multiple(x, multiple: int = 8):
            h, w = x.shape[2], x.shape[3]
            pad_h = (multiple - h % multiple) % multiple
            pad_w = (multiple - w % multiple) % multiple
            if pad_h == 0 and pad_w == 0:
                return x, (0, 0)
            return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)

        def forward(self, x):
            orig_h, orig_w = x.shape[2], x.shape[3]
            x, _pad = self._pad_to_multiple(x, 8)

            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))

            d3_in = torch.cat([self._upsample_to(b, e3), e3], dim=1)
            d3 = self.dec3(d3_in)
            d2_in = torch.cat([self._upsample_to(d3, e2), e2], dim=1)
            d2 = self.dec2(d2_in)
            d1_in = torch.cat([self._upsample_to(d2, e1), e1], dim=1)
            d1 = self.dec1(d1_in)

            out = self.out_conv(d1)
            out = out[:, :, :orig_h, :orig_w]
            reflection = torch.sigmoid(out[:, :3])
            alpha = torch.sigmoid(out[:, 3:4])
            return reflection, alpha

    _UNetClass = ReflectionSuppressionUNet
    return _UNetClass


def build_model(
    encoder_channels: Optional[list[int]] = None,
    decoder_channels: Optional[list[int]] = None,
    activation: str = DEFAULT_ACTIVATION,
):
    _require_torch()
    cls = _get_unet_class()
    return cls(
        encoder_channels or list(DEFAULT_ENCODER_CHANNELS),
        decoder_channels or list(DEFAULT_DECODER_CHANNELS),
        activation,
    )
