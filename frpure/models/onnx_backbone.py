from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from frpure.models.backbone import FRBackbone

class OnnxBackbone(FRBackbone):
    input_size = 112
    embed_dim = 512
    def __init__(self, onnx_path, input_size=112, norm="arcface", bgr=False, device="cuda", providers=None):
        super().__init__()
        import onnxruntime as ort
        self.input_size = input_size; self.device = device; self.norm = norm; self.bgr = bgr
        if providers is None:
            providers = (["CUDAExecutionProvider","CPUExecutionProvider"]
                         if str(device).startswith("cuda") else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        d = self.sess.get_outputs()[0].shape[-1]
        self.embed_dim = d if isinstance(d, int) else 512
    @torch.no_grad()
    def embed(self, imgs01):
        x = imgs01.unsqueeze(0) if imgs01.dim() == 3 else imgs01
        x = x.detach().to("cpu").float()
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        if self.bgr:
            x = x[:, [2, 1, 0], :, :]
        if self.norm == "arcface":
            x = (x - 0.5) / 0.5
        emb = self.sess.run([self.out], {self.inp: x.numpy().astype("float32")})[0]
        emb = torch.from_numpy(np.asarray(emb)).to(self.device).float()
        return emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-9)
    def _forward_features(self, x):
        raise RuntimeError("OnnxBackbone is non-differentiable: certification/scoring only.")