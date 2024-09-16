import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(
    ["*efficientformer*"], exclude_filters=[], pretrained=True
)
for model in models_list:
    print(model)
    # efficientformer_l1.snap_dist_in1k
    # efficientformer_l3.snap_dist_in1k
    # efficientformer_l7.snap_dist_in1k
    # efficientformerv2_l.snap_dist_in1k
    # efficientformerv2_s0.snap_dist_in1k
    # efficientformerv2_s1.snap_dist_in1k
    # efficientformerv2_s2.snap_dist_in1k

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)
model = models.efficientformer.efficientformer_l1(pretrained=False, num_classes=5).to(
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
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.95}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=235, interpolation=bicubic, max_size=None, antialias=None)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
# )


# 查看结构
if False:
    onnx_path = "efficientformer_l1.onnx"
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
