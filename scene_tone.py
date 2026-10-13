"""Lightweight scene grading for ordinary sRGB anime illustrations."""

import io

import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from server import PromptServer

from .skin_tone import ThumbnailCache, data_url, encode_png, oklab_to_rgb, rgb_to_oklab, smoothstep


def prepare_mask(mask, image):
    mask = mask.to(device=image.device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or mask.shape[0] not in (1, image.shape[0]):
        raise ValueError("遮罩批次数必须为 1，或与图片批次数一致。")
    if not torch.isfinite(mask).all():
        raise ValueError("遮罩包含无效数值。")
    if mask.shape[-2:] != image.shape[1:3]:
        mask = F.interpolate(mask.unsqueeze(1), size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
    return mask.clamp(0, 1).expand(image.shape[0], -1, -1)


def grade_scene(image, midtones=0, shadows=0, remove_cold=0, warmth=0, strength=100, skin_mask=None):
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("需要 RGB 或 RGBA IMAGE 输入。")
    if strength == 0 or all(value == 0 for value in (midtones, shadows, remove_cold, warmth)):
        return image
    rgb = image[..., :3].float().clamp(0, 1)
    lab = rgb_to_oklab(rgb)
    L, a, b = lab.unbind(-1)
    chroma = torch.sqrt(a.square()+b.square())
    # Preserve near-white clouds and specular highlights, as well as true blacks.
    highlight_weight = 1 - smoothstep(0.82, 0.96, L)
    black_weight = smoothstep(0.04, 0.22, L)
    mid_weight = 4 * L * (1-L) * black_weight * highlight_weight
    shadow_weight = smoothstep(0.04, 0.24, L) * (1-smoothstep(0.40, 0.72, L))
    lift = midtones/100 * 0.45 * mid_weight + shadows/100 * 0.20 * shadow_weight
    adjusted_L = (L + lift).clamp(0, 1)
    # Desaturate only the blue component of muted shadows; don't globally yellow the sky.
    muted_weight = 1 - smoothstep(0.035, 0.105, chroma)
    cold_zone = black_weight * (1-smoothstep(0.68, 0.88, L)) * muted_weight
    adjusted_b = b + (-b).clamp_min(0) * remove_cold/100 * cold_zone
    # Very small warm midtone bias, tapering over already yellow/golden materials.
    warm_weight = mid_weight * muted_weight * (1-smoothstep(0.025, 0.085, b))
    adjusted_b = adjusted_b + warmth/100 * 0.04 * warm_weight
    adjusted_a = a + warmth/100 * 0.008 * warm_weight
    graded = oklab_to_rgb(torch.stack((adjusted_L, adjusted_a, adjusted_b), -1))
    result = rgb + strength/100 * (graded-rgb)
    # Avoid round-trip drift on exactly protected white/black pixels.
    changed = (lift.abs() + (adjusted_b-b).abs() + (adjusted_a-a).abs()) > 0
    result = torch.where(changed.unsqueeze(-1), result, rgb).to(image.dtype)
    if image.shape[-1] == 4:
        result = torch.cat((result, image[...,3:]), -1)
    if skin_mask is not None:
        # The upstream skin mask is already feathered; do not expand it again.
        mask = prepare_mask(skin_mask, image).unsqueeze(-1)
        result = (image.float() + mask * (result.float()-image.float())).to(image.dtype)
        result = torch.where(mask == 0, image, result)
    return result


_thumbnails = ThumbnailCache()


class AnimaSceneTone:
    DESCRIPTION = "提亮中间调、减轻暗部冷灰、少量加暖。接入肤色节点的 skin_mask 后，只调整同一选区，黑色区域保持输入原样；不接遮罩则调整整图。自动肤色选区按颜色判断，可能包含相近颜色的物品。预览只显示批次第一张；上游肤色或遮罩改变后先重新运行，保存也需要再次运行。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "midtones": ("FLOAT", {"default": 0, "min": -20, "max": 30, "step": 1, "tooltip": "中间调通透：正值提亮中间层次，接近纯白的高光逐渐不受影响。"}),
            "shadows": ("FLOAT", {"default": 0, "min": -20, "max": 30, "step": 1, "tooltip": "暗部提亮：改善较暗区域，保留纯黑。太高会让黑衣发灰。"}),
            "remove_cold": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "去冷灰：降低低饱和暗部的蓝色成分；不是全图降低饱和度。"}),
            "warmth": ("FLOAT", {"default": 0, "min": -30, "max": 30, "step": 1, "tooltip": "中间调暖意：正值偏暖、负值偏冷，减弱对已经偏金黄和接近纯白区域的影响。"}),
            "strength": ("FLOAT", {"default": 100, "min": 0, "max": 100, "step": 1, "tooltip": "整体强度：100 为完整效果，0 精确返回输入图。"}),
        }, "optional": {"skin_mask": ("MASK", {"tooltip": "接肤色节点的 skin_mask，复用其选区。白色调整、黑色完全保留；不接时仍调整全图。"})},
           "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Anima/画面调整"

    def apply(self, image, midtones, shadows, remove_cold, warmth, strength,
              unique_id=None, extra_pnginfo=None, skin_mask=None):
        result = grade_scene(image, midtones, shadows, remove_cold, warmth, strength, skin_mask)
        thumbnail = Image.fromarray((image[0,...,:3].detach().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8))
        thumbnail.thumbnail((640,640), Image.Resampling.LANCZOS)
        png = encode_png(np.asarray(thumbnail))
        mask_png = None
        if skin_mask is not None:
            mask = prepare_mask(skin_mask, image)[0].detach().cpu().numpy()
            mask_image = Image.fromarray((mask*255).round().astype(np.uint8))
            mask_image = mask_image.resize(thumbnail.size, Image.Resampling.BILINEAR)
            mask_png = encode_png(np.asarray(mask_image))
        workflow_id = (extra_pnginfo or {}).get("workflow",{}).get("id","api")
        key = _thumbnails.add((str(workflow_id),str(unique_id)),png,mask_png)
        return {"ui": {"scene_preview_key": [key], "scene_original": [data_url(png)]}, "result": (result,)}


@PromptServer.instance.routes.post("/anima_scene_tone/preview")
async def preview(request):
    data = await request.json()
    entry = _thumbnails.entries.get(data.get("key"))
    if entry is None:
        return web.json_response({"error": "预览已过期，请运行一次工作流。"}, status=404)
    values = {}
    for name,low,high,default in (("midtones",-20,30,0),("shadows",-20,30,0),
                                  ("remove_cold",0,100,0),("warmth",-30,30,0),("strength",0,100,100)):
        value = data.get(name,default)
        if not isinstance(value,(int,float)) or not np.isfinite(value) or not low <= value <= high:
            return web.json_response({"error":"参数超出范围："+name},status=400)
        values[name] = float(value)
    source = torch.from_numpy(np.array(Image.open(io.BytesIO(entry[1])).convert("RGB"))).float().unsqueeze(0)/255
    mask = None
    if entry[2] is not None:
        mask = torch.from_numpy(np.array(Image.open(io.BytesIO(entry[2])))).float().unsqueeze(0)/255
    result = grade_scene(source,skin_mask=mask,**values)
    rgb = (result[0].numpy().clip(0,1)*255).round().astype(np.uint8)
    alpha = (mask[0].numpy()[...,None] if mask is not None else 1.0) * 0.65
    overlay = source[0].numpy()*(1-alpha) + np.array([0.1,1.0,0.3])*alpha
    overlay = (overlay.clip(0,1)*255).round().astype(np.uint8)
    return web.json_response({"result":data_url(encode_png(rgb)), "selection":data_url(encode_png(overlay)), "masked":mask is not None})


NODE_CLASS_MAPPINGS = {"AnimaSceneTone": AnimaSceneTone}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaSceneTone": "Anima 画面通透"}
