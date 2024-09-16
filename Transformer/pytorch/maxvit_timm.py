import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(["*maxvit*"], exclude_filters=[], pretrained=True)
for model in models_list:
    print(model)
    # maxvit_base_tf_224.in1k
    # maxvit_base_tf_224.in21k
    # maxvit_base_tf_384.in1k
    # maxvit_base_tf_384.in21k_ft_in1k
    # maxvit_base_tf_512.in1k
    # maxvit_base_tf_512.in21k_ft_in1k
    # maxvit_large_tf_224.in1k
    # maxvit_large_tf_224.in21k
    # maxvit_large_tf_384.in1k
    # maxvit_large_tf_384.in21k_ft_in1k
    # maxvit_large_tf_512.in1k
    # maxvit_large_tf_512.in21k_ft_in1k
    # maxvit_nano_rw_256.sw_in1k
    # maxvit_rmlp_base_rw_224.sw_in12k
    # maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k
    # maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k
    # maxvit_rmlp_nano_rw_256.sw_in1k
    # maxvit_rmlp_pico_rw_256.sw_in1k
    # maxvit_rmlp_small_rw_224.sw_in1k
    # maxvit_rmlp_tiny_rw_256.sw_in1k
    # maxvit_small_tf_224.in1k
    # maxvit_small_tf_384.in1k
    # maxvit_small_tf_512.in1k
    # maxvit_tiny_rw_224.sw_in1k
    # maxvit_tiny_tf_224.in1k
    # maxvit_tiny_tf_384.in1k
    # maxvit_tiny_tf_512.in1k
    # maxvit_xlarge_tf_224.in21k
    # maxvit_xlarge_tf_384.in21k_ft_in1k
    # maxvit_xlarge_tf_512.in21k_ft_in1k

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)
model = models.maxxvit.maxvit_tiny_rw_224(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 5]


# ---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
# ---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'crop_pct': 0.95}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=235, interpolation=bicubic, max_size=None, antialias=None)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
# )


# 查看结构
if False:
    onnx_path = "maxvit_tiny_rw_224.onnx"
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
