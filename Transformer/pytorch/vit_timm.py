import torch
import timm
from timm import models
from timm.data import resolve_data_config, create_transform


models_list = timm.list_models(["vit*"], pretrained=True)
for model in models_list:
    print(model)
    # vit_base_patch8_224.augreg2_in21k_ft_in1k
    # vit_base_patch8_224.augreg_in21k
    # vit_base_patch8_224.augreg_in21k_ft_in1k
    # vit_base_patch8_224.dino
    # vit_base_patch14_dinov2.lvd142m
    # vit_base_patch16_224.augreg2_in21k_ft_in1k
    # vit_base_patch16_224.augreg_in1k
    # vit_base_patch16_224.augreg_in21k
    # vit_base_patch16_224.augreg_in21k_ft_in1k
    # vit_base_patch16_224.dino
    # vit_base_patch16_224.mae
    # vit_base_patch16_224.orig_in21k_ft_in1k
    # vit_base_patch16_224.sam_in1k
    # vit_base_patch16_224_miil.in21k
    # vit_base_patch16_224_miil.in21k_ft_in1k
    # vit_base_patch16_384.augreg_in1k
    # vit_base_patch16_384.augreg_in21k_ft_in1k
    # vit_base_patch16_384.orig_in21k_ft_in1k
    # vit_base_patch16_clip_224.datacompxl
    # vit_base_patch16_clip_224.laion2b
    # vit_base_patch16_clip_224.laion2b_ft_in1k
    # vit_base_patch16_clip_224.laion2b_ft_in12k
    # vit_base_patch16_clip_224.laion2b_ft_in12k_in1k
    # vit_base_patch16_clip_224.openai
    # vit_base_patch16_clip_224.openai_ft_in1k
    # vit_base_patch16_clip_224.openai_ft_in12k
    # vit_base_patch16_clip_224.openai_ft_in12k_in1k
    # vit_base_patch16_clip_384.laion2b_ft_in1k
    # vit_base_patch16_clip_384.laion2b_ft_in12k_in1k
    # vit_base_patch16_clip_384.openai_ft_in1k
    # vit_base_patch16_clip_384.openai_ft_in12k_in1k
    # vit_base_patch16_rpn_224.sw_in1k
    # vit_base_patch32_224.augreg_in1k
    # vit_base_patch32_224.augreg_in21k
    # vit_base_patch32_224.augreg_in21k_ft_in1k
    # vit_base_patch32_224.sam_in1k
    # vit_base_patch32_384.augreg_in1k
    # vit_base_patch32_384.augreg_in21k_ft_in1k
    # vit_base_patch32_clip_224.laion2b
    # vit_base_patch32_clip_224.laion2b_ft_in1k
    # vit_base_patch32_clip_224.laion2b_ft_in12k_in1k
    # vit_base_patch32_clip_224.openai
    # vit_base_patch32_clip_224.openai_ft_in1k
    # vit_base_patch32_clip_384.laion2b_ft_in12k_in1k
    # vit_base_patch32_clip_384.openai_ft_in12k_in1k
    # vit_base_patch32_clip_448.laion2b_ft_in12k_in1k
    # vit_base_r50_s16_224.orig_in21k
    # vit_base_r50_s16_384.orig_in21k_ft_in1k
    # vit_giant_patch14_clip_224.laion2b
    # vit_giant_patch14_dinov2.lvd142m
    # vit_gigantic_patch14_clip_224.laion2b
    # vit_gigantic_patch16_224_ijepa.in22k
    # vit_huge_patch14_224.mae
    # vit_huge_patch14_224.orig_in21k
    # vit_huge_patch14_224_ijepa.in1k
    # vit_huge_patch14_224_ijepa.in22k
    # vit_huge_patch14_clip_224.laion2b
    # vit_huge_patch14_clip_224.laion2b_ft_in1k
    # vit_huge_patch14_clip_224.laion2b_ft_in12k
    # vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k
    # vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k
    # vit_huge_patch16_448_ijepa.in1k
    # vit_large_patch14_clip_224.datacompxl
    # vit_large_patch14_clip_224.laion2b
    # vit_large_patch14_clip_224.laion2b_ft_in1k
    # vit_large_patch14_clip_224.laion2b_ft_in12k
    # vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
    # vit_large_patch14_clip_224.openai
    # vit_large_patch14_clip_224.openai_ft_in1k
    # vit_large_patch14_clip_224.openai_ft_in12k
    # vit_large_patch14_clip_224.openai_ft_in12k_in1k
    # vit_large_patch14_clip_336.laion2b_ft_in1k
    # vit_large_patch14_clip_336.laion2b_ft_in12k_in1k
    # vit_large_patch14_clip_336.openai
    # vit_large_patch14_clip_336.openai_ft_in12k_in1k
    # vit_large_patch14_dinov2.lvd142m
    # vit_large_patch16_224.augreg_in21k
    # vit_large_patch16_224.augreg_in21k_ft_in1k
    # vit_large_patch16_224.mae
    # vit_large_patch16_384.augreg_in21k_ft_in1k
    # vit_large_patch32_224.orig_in21k
    # vit_large_patch32_384.orig_in21k_ft_in1k
    # vit_large_r50_s32_224.augreg_in21k
    # vit_large_r50_s32_224.augreg_in21k_ft_in1k
    # vit_large_r50_s32_384.augreg_in21k_ft_in1k
    # vit_medium_patch16_gap_240.sw_in12k
    # vit_medium_patch16_gap_256.sw_in12k_ft_in1k
    # vit_medium_patch16_gap_384.sw_in12k_ft_in1k
    # vit_relpos_base_patch16_224.sw_in1k
    # vit_relpos_base_patch16_clsgap_224.sw_in1k
    # vit_relpos_base_patch32_plus_rpn_256.sw_in1k
    # vit_relpos_medium_patch16_224.sw_in1k
    # vit_relpos_medium_patch16_cls_224.sw_in1k
    # vit_relpos_medium_patch16_rpn_224.sw_in1k
    # vit_relpos_small_patch16_224.sw_in1k
    # vit_small_patch8_224.dino
    # vit_small_patch14_dinov2.lvd142m
    # vit_small_patch16_224.augreg_in1k
    # vit_small_patch16_224.augreg_in21k
    # vit_small_patch16_224.augreg_in21k_ft_in1k
    # vit_small_patch16_224.dino
    # vit_small_patch16_384.augreg_in1k
    # vit_small_patch16_384.augreg_in21k_ft_in1k
    # vit_small_patch32_224.augreg_in21k
    # vit_small_patch32_224.augreg_in21k_ft_in1k
    # vit_small_patch32_384.augreg_in21k_ft_in1k
    # vit_small_r26_s32_224.augreg_in21k
    # vit_small_r26_s32_224.augreg_in21k_ft_in1k
    # vit_small_r26_s32_384.augreg_in21k_ft_in1k
    # vit_srelpos_medium_patch16_224.sw_in1k
    # vit_srelpos_small_patch16_224.sw_in1k
    # vit_tiny_patch16_224.augreg_in21k
    # vit_tiny_patch16_224.augreg_in21k_ft_in1k
    # vit_tiny_patch16_384.augreg_in21k_ft_in1k
    # vit_tiny_r_s16_p8_224.augreg_in21k
    # vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k
    # vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k

device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(1, 3, 224, 224).to(device)
model = models.vision_transformer.vit_base_patch16_224(pretrained=False, num_classes=5).to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size()) # [1, 5]


#---------------------------------------------------------------------#
#   创建对应的图片预处理，配合PIL.Image.Open('path').convert('RGB')
#---------------------------------------------------------------------#
config = resolve_data_config({}, model=model)
print(config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'crop_pct': 0.9, 'crop_mode': 'center'}

transform = create_transform(**config)
print(transform)
# Compose(
#     Resize(size=248, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
# )


# 查看结构
if False:
    onnx_path = 'vit_base_patch16_224.onnx'
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
