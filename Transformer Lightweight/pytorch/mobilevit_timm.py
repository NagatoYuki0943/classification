import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(["*mobilevit*"], exclude_filters=["*mobilevitv2*"], pretrained=True)
for model in models_list:
    print(model)
    # mobilevit_s.cvnets_in1k
    # mobilevit_xs.cvnets_in1k
    # mobilevit_xxs.cvnets_in1k

device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(1, 3, 256, 256).to(device)
model = models.mobilevit.mobilevit_s(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size()) # [1, 5]


#---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
#---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 256, 256), 'interpolation': 'bicubic', 'mean': (0.0, 0.0, 0.0), 'std': (1.0, 1.0, 1.0), 'crop_pct': 0.9}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=284, interpolation=bicubic, max_size=None, antialias=None)
#     CenterCrop(size=(256, 256))
#     ToTensor()
#     Normalize(mean=tensor([0., 0., 0.]), std=tensor([1., 1., 1.]))
# )


# 查看结构
if False:
    onnx_path = 'mobilevit_s.onnx'
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
