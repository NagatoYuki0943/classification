"""source code in mobilevit_timm_s.py
"""

import torch
from mobilevit_timm_s import (
    mobilevitv2_050,
    mobilevitv2_075,
    mobilevitv2_100,
    mobilevitv2_125,
    mobilevitv2_150,
    mobilevitv2_175,
    mobilevitv2_200,
)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 256, 256).to(device)
    model = mobilevitv2_100(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    # 查看结构
    if False:
        onnx_path = 'mobilevitv2_100.onnx'
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=['images'],
            output_names=['classes'],
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print('finished exporting ' + onnx_path)
