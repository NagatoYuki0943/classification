"""source code in metaformer_timm_s.py"""

import torch
from metaformer_timm_s import (
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    poolformer_m36,
    poolformer_m48,
    poolformerv2_s12,
    poolformerv2_s24,
    poolformerv2_s36,
    poolformerv2_m36,
    poolformerv2_m48,
)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = poolformer_s12(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 查看结构
    if False:
        onnx_path = "poolformer_s12.onnx"
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=["images"],
            output_names=["classes"],
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print("finished exporting " + onnx_path)
