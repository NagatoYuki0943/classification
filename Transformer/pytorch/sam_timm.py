import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(["*sam*"], pretrained=True)
for model in models_list:
    print(model)
    # samvit_base_patch16.sa1b
    # samvit_huge_patch16.sa1b
    # samvit_large_patch16.sa1b
    # vit_base_patch16_224.sam_in1k
    # vit_base_patch32_224.sam_in1k

device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(1, 3, 1024, 1024).to(device)
# model = timm.create_model("samvit_base_patch16.sa1b", pretrained=True, num_classes=0, global_pool="").to(device)
model = models.vision_transformer_sam.samvit_base_patch16(pretrained=True, num_classes=0, global_pool="").to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size()) # [1, 256, 64, 64]


#---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
#---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 1024, 1024), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 1.0, 'crop_mode': 'center'}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=1024, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(1024, 1024))
#     ToTensor()
#     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
# )


# 查看结构
if False:
    onnx_path = 'samvit_base_patch16.sa1b.onnx'
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
