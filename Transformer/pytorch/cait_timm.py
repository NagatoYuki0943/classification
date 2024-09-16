import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(filter=["*cait*"], exclude_filters=[], pretrained=True)
for model in models_list:
    print(model)
    # cait_m36_384.fb_dist_in1k
    # cait_m48_448.fb_dist_in1k
    # cait_s24_224.fb_dist_in1k
    # cait_s24_384.fb_dist_in1k
    # cait_s36_384.fb_dist_in1k
    # cait_xs24_384.fb_dist_in1k
    # cait_xxs24_224.fb_dist_in1k
    # cait_xxs24_384.fb_dist_in1k
    # cait_xxs36_224.fb_dist_in1k
    # cait_xxs36_384.fb_dist_in1k

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)
model = models.cait.cait_xxs24_224(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 5]


# ---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
# ---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 1.0}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
# )


# 查看结构
if False:
    onnx_path = "cait_xxs24_224.onnx"
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
