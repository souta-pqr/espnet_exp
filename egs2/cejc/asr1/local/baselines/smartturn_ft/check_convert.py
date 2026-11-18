"""読み戻した PyTorch のモデルが公開 ONNX と同じ出力を出すかを確かめる（ランダムな入力と CEJC の音声）。"""
import sys, numpy as np, torch, onnxruntime as ort
sys.path.insert(0, sys.path[0])
from st_model import load_from_onnx
P = "/home/kobori/tt_base/smart_turn/smart-turn-v3.2-gpu.onnx"
m, miss, unexp = load_from_onnx(P)
print("足りない重み:", miss); print("余った重み:", unexp)
m.eval()
s = ort.InferenceSession(P, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
x = rng.standard_normal((8, 80, 800)).astype(np.float32) * 0.5
po = s.run(None, {"input_features": x})[0][:, 0]
with torch.no_grad():
    pt = torch.sigmoid(m(torch.from_numpy(x))).numpy()
print("ONNX:", np.round(po, 5)); print("PT  :", np.round(pt, 5)); print("最大の差:", np.abs(po - pt).max())
