import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(
    filter=["*inception_v3*"], exclude_filters=[], pretrained=True
)
for model in models_list:
    print(model)
    # inception_v3.gluon_in1k
    # inception_v3.tf_adv_in1k
    # inception_v3.tf_in1k
    # inception_v3.tv_in1k

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 299, 299).to(device)
# model = models.inception_v3(pretrained=False, num_classes=5).to(device)
model = timm.create_model("inception_v3.tf_in1k", pretrained=False, num_classes=5).to(
    device
)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 5]


# ---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
# ---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 299, 299), 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'crop_pct': 0.875, 'crop_mode': 'center'}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=341, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(299, 299))
#     ToTensor()
#     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
# )


# 查看结构
if False:
    onnx_path = "inception_v3.onnx"
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
    onnx.save(model_simp, onnx_path)
    print("finished exporting " + onnx_path)
