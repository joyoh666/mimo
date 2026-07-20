import os
from types import SimpleNamespace

import torch

from BeamFormer.beamformer.inference_helper import InferenceHelper
from BeamFormer.beamformer.weight_generator import transform_weights
from BeamFormer.configs.submodules import assumption, dataset, estimator, generator, ARN_model


def build_beamformer_setting(
    model_dir="BeamFormer/saved_models/co_train",
    arn_model_path="BeamFormer/ARN_saved_models/arn/model_epoch_final.pth",
):
    ds = dataset.homeoffice_communication_28g()

    return SimpleNamespace(
        dataset=ds,
        assumption=assumption.beam64(),
        generator=generator.parametric_generator(
            generator_pretrained_model=os.path.join(model_dir, "generator_epoch_final.pth")
        ),
        estimator=estimator.PerceiverIO(
            estimator_pretrained_model=os.path.join(model_dir, "estimator_epoch_final.pth")
        ),
        arn_model=ARN_model.typical_ARN(
            ARN_model_pretrained_model=arn_model_path if os.path.exists(arn_model_path) else None
        ),
    )


class BeamFormerCsirsMeasurer:
    def __init__(self, setting=None):
        self.setting = setting or build_beamformer_setting()
        self.helper = InferenceHelper(self.setting)
        self.device = self.helper.device
        self.reference_weights = self._make_reference_weights()

    @torch.no_grad()
    def _make_reference_weights(self):
        """64개 CSI-RS reference beam weight 생성: [1, 64, 16, 16]."""
        z_dim = (
            self.setting.assumption.sample_num
            * self.setting.dataset.M
            * self.setting.dataset.N
        )
        z = torch.zeros(1, z_dim, device=self.device)
        raw_weights = self.helper.generator(z)
        weights, _ = transform_weights(raw_weights)
        return weights.detach().cpu()

    def get_csirs_precoders(self):
        """RF/PHY에 넘길 CSI-RS precoder. shape: [64, 256]."""
        return self.reference_weights[0].reshape(
            self.setting.assumption.sample_num, -1
        )

    @torch.no_grad()
    def simulate_csirs_measurement_from_csi(self, csi_tensor):
        """
        시뮬레이션용 CSI-RS 측정.
        csi_tensor: [freq, rx, tx] 또는 [1, freq, rx, tx]
        return: sample_rss [64], linear power
        """
        if csi_tensor.ndim == 3:
            csi_tensor = csi_tensor.unsqueeze(0)

        csi_tensor = csi_tensor.to(self.device)
        weights = self.reference_weights.to(self.device)

        sample_rss = self.helper.dp.generate_sample_rss(csi_tensor, weights)
        return sample_rss[0].detach().cpu()

    @torch.no_grad()
    def predict_best_beam_from_csirs(self, measured_rsrp, unit="linear"):
        """
        measured_rsrp: CSI-RS 64개 측정값.
            unit="linear": linear RSS/RSRP power
            unit="dbm": dBm RSRP
        """
        rss = torch.as_tensor(measured_rsrp, dtype=torch.float32)

        if unit == "dbm":
            # mW 단위 linear power. 정규화하므로 W/mW 선택은 결과에 영향 거의 없음.
            rss = 10.0 ** (rss / 10.0)

        scale = torch.clamp(torch.max(rss), min=1e-12)
        rss_norm = rss / scale

        pred = self.helper.infer_from_rss(rss_norm, self.reference_weights)

        if self.helper.arn_model.has_pretrained:
            spectrum = self.helper.apply_arn(rss_norm, pred, scale.item())
            spectrum = torch.as_tensor(spectrum).reshape(-1)
        else:
            spectrum = pred.reshape(-1) * scale

        best_index = int(torch.argmax(spectrum).item())

        theta_steps = self.setting.assumption.angle_steps_theta
        phi_steps = self.setting.assumption.angle_steps_phi

        phi_index = best_index // theta_steps
        theta_index = best_index % theta_steps

        phi_deg = phi_index * (360.0 / phi_steps)
        theta_deg = theta_index * (
            self.setting.dataset.max_theta / (theta_steps - 1)
        )

        query_weights = self.helper.dp.generate_query_weights(batch_size=1)
        best_beam_weight = query_weights[0, best_index].detach().cpu()

        return {
            "best_index": best_index,
            "phi_deg": phi_deg,
            "theta_deg": theta_deg,
            "predicted_power": float(spectrum[best_index].item()),
            "beam_weight": best_beam_weight,  # [16, 16]
            "spectrum": spectrum.reshape(phi_steps, theta_steps),
        }