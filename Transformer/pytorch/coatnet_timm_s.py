"""source code in maxvit_timm_s.py"""

import torch
from maxvit_timm_s import (
    coatnet_0_224,
    coatnet_0_rw_224,
    coatnet_1_224,
    coatnet_1_rw_224,
    coatnet_2_224,
    coatnet_2_rw_224,
    coatnet_3_224,
    coatnet_3_rw_224,
    coatnet_4_224,
    coatnet_5_224,
    coatnet_bn_0_rw_224,
    coatnet_nano_cc_224,
    coatnet_nano_rw_224,
    coatnet_pico_rw_224,
    coatnet_rmlp_0_rw_224,
    coatnet_rmlp_1_rw_224,
    coatnet_rmlp_1_rw2_224,
    coatnet_rmlp_2_rw_224,
    coatnet_rmlp_2_rw_384,
    coatnet_rmlp_3_rw_224,
    coatnet_rmlp_nano_rw_224,
    coatnext_nano_rw_224,
)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = coatnet_0_rw_224(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 查看结构
    if False:
        onnx_path = "coatnet_0_rw_224.onnx"
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
