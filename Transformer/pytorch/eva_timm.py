import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(filter=["*eva*"], exclude_filters=[], pretrained=True)
for model in models_list:
    print(model)
    # eva02_base_patch14_224.mim_in22k
    # eva02_base_patch14_448.mim_in22k_ft_in1k
    # eva02_base_patch14_448.mim_in22k_ft_in22k
    # eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
    # eva02_base_patch16_clip_224.merged2b
    # eva02_enormous_patch14_clip_224.laion2b
    # eva02_enormous_patch14_clip_224.laion2b_plus
    # eva02_large_patch14_224.mim_in22k
    # eva02_large_patch14_224.mim_m38m
    # eva02_large_patch14_448.mim_in22k_ft_in1k
    # eva02_large_patch14_448.mim_in22k_ft_in22k
    # eva02_large_patch14_448.mim_in22k_ft_in22k_in1k
    # eva02_large_patch14_448.mim_m38m_ft_in1k
    # eva02_large_patch14_448.mim_m38m_ft_in22k
    # eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
    # eva02_large_patch14_clip_224.merged2b
    # eva02_large_patch14_clip_336.merged2b
    # eva02_small_patch14_224.mim_in22k
    # eva02_small_patch14_336.mim_in22k_ft_in1k
    # eva02_tiny_patch14_224.mim_in22k
    # eva02_tiny_patch14_336.mim_in22k_ft_in1k
    # eva_giant_patch14_224.clip_ft_in1k
    # eva_giant_patch14_336.clip_ft_in1k
    # eva_giant_patch14_336.m30m_ft_in22k_in1k
    # eva_giant_patch14_560.m30m_ft_in22k_in1k
    # eva_giant_patch14_clip_224.laion400m
    # eva_giant_patch14_clip_224.merged2b
    # eva_large_patch14_196.in22k_ft_in1k
    # eva_large_patch14_196.in22k_ft_in22k_in1k
    # eva_large_patch14_336.in22k_ft_in1k
    # eva_large_patch14_336.in22k_ft_in22k_in1k

device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(1, 3, 224, 224).to(device)
model = models.eva.eva_giant_patch14_224(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size()) # [1, 5]


#---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
#---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711), 'crop_pct': 0.9, 'crop_mode': 'center'}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=248, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=tensor([0.4815, 0.4578, 0.4082]), std=tensor([0.2686, 0.2613, 0.2758]))
# )


# 查看结构
if False:
    onnx_path = 'eva_giant_patch14_224.onnx'
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
