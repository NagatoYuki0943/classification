import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(["*coatnet*"], exclude_filters=[], pretrained=True)
for model in models_list:
    print(model)
    # coatnet_0_rw_224.sw_in1k
    # coatnet_1_rw_224.sw_in1k
    # coatnet_2_rw_224.sw_in12k
    # coatnet_2_rw_224.sw_in12k_ft_in1k
    # coatnet_3_rw_224.sw_in12k
    # coatnet_bn_0_rw_224.sw_in1k
    # coatnet_nano_rw_224.sw_in1k
    # coatnet_rmlp_1_rw2_224.sw_in12k
    # coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k
    # coatnet_rmlp_1_rw_224.sw_in1k
    # coatnet_rmlp_2_rw_224.sw_in1k
    # coatnet_rmlp_2_rw_224.sw_in12k
    # coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k
    # coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
    # coatnet_rmlp_nano_rw_224.sw_in1k

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)
model = models.maxxvit.coatnet_nano_rw_224(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 5]


# ---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
# ---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'crop_pct': 0.9}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=248, interpolation=bicubic, max_size=None, antialias=None)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
# )


# 查看结构
if False:
    onnx_path = "coatnet_nano_rw_224.onnx"
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
