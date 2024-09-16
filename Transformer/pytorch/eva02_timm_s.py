"""source code in beit_timm_s.py"""

import torch
from eva_timm_s import (
    eva02_base_patch14_224,
    eva02_base_patch14_448,
    eva02_base_patch14_448,
    eva02_base_patch14_448,
    eva02_base_patch16_clip_224,
    eva02_enormous_patch14_clip_224,
    eva02_enormous_patch14_clip_224,
    eva02_large_patch14_224,
    eva02_large_patch14_224,
    eva02_large_patch14_448,
    eva02_large_patch14_448,
    eva02_large_patch14_448,
    eva02_large_patch14_448,
    eva02_large_patch14_448,
    eva02_large_patch14_448,
    eva02_large_patch14_clip_224,
    eva02_large_patch14_clip_336,
    eva02_small_patch14_224,
    eva02_small_patch14_336,
    eva02_tiny_patch14_224,
    eva02_tiny_patch14_336,
)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = eva02_base_patch14_224(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 查看结构
    if False:
        onnx_path = "eva02_base_patch14_224.onnx"
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
