import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(["*resnext*"], exclude_filters=[], pretrained=True)
for model in models_list:
    print(model)
    # cspresnext50.ra_in1k
    # eca_resnext26ts.ch_in1k
    # gcresnext26ts.ch_in1k
    # gcresnext50ts.ch_in1k
    # legacy_seresnext26_32x4d.in1k
    # legacy_seresnext50_32x4d.in1k
    # legacy_seresnext101_32x4d.in1k
    # resnext26ts.ra2_in1k
    # resnext50_32x4d.a1_in1k
    # resnext50_32x4d.a1h_in1k
    # resnext50_32x4d.a2_in1k
    # resnext50_32x4d.a3_in1k
    # resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k
    # resnext50_32x4d.fb_swsl_ig1b_ft_in1k
    # resnext50_32x4d.gluon_in1k
    # resnext50_32x4d.ra_in1k
    # resnext50_32x4d.tv2_in1k
    # resnext50_32x4d.tv_in1k
    # resnext50d_32x4d.bt_in1k
    # resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k
    # resnext101_32x4d.fb_swsl_ig1b_ft_in1k
    # resnext101_32x4d.gluon_in1k
    # resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k
    # resnext101_32x8d.fb_swsl_ig1b_ft_in1k
    # resnext101_32x8d.fb_wsl_ig1b_ft_in1k
    # resnext101_32x8d.tv2_in1k
    # resnext101_32x8d.tv_in1k
    # resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k
    # resnext101_32x16d.fb_swsl_ig1b_ft_in1k
    # resnext101_32x16d.fb_wsl_ig1b_ft_in1k
    # resnext101_32x32d.fb_wsl_ig1b_ft_in1k
    # resnext101_64x4d.c1_in1k
    # resnext101_64x4d.gluon_in1k
    # resnext101_64x4d.tv_in1k
    # seresnext26d_32x4d.bt_in1k
    # seresnext26t_32x4d.bt_in1k
    # seresnext26ts.ch_in1k
    # seresnext50_32x4d.gluon_in1k
    # seresnext50_32x4d.racm_in1k
    # seresnext101_32x4d.gluon_in1k
    # seresnext101_32x8d.ah_in1k
    # seresnext101_64x4d.gluon_in1k
    # seresnext101d_32x8d.ah_in1k
    # seresnextaa101d_32x8d.ah_in1k
    # seresnextaa101d_32x8d.sw_in12k
    # seresnextaa101d_32x8d.sw_in12k_ft_in1k
    # seresnextaa101d_32x8d.sw_in12k_ft_in1k_288
    # seresnextaa201d_32x8d.sw_in12k
    # seresnextaa201d_32x8d.sw_in12k_ft_in1k_384
    # skresnext50_32x4d.ra_in1k

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 384, 384).to(device)
# July 27, 2023
# Added timm trained seresnextaa201d_32x8d.sw_in12k_ft_in1k_384 weights (and .sw_in12k pretrain) with 87.3% top-1 on ImageNet-1k, best ImageNet ResNet family model I'm aware of.
model = models.resnet.seresnextaa201d_32x8d(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 5]


# ---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
# ---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 384, 384), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 1.0, 'crop_mode': 'center'}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=384, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(384, 384))
#     ToTensor()
#     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
# )

# 查看结构
if False:
    onnx_path = "seresnextaa201d_32x8d.onnx"
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
