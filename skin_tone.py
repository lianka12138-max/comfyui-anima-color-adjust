"""Targeted skin grading for muted pink anime illustrations, in display sRGB."""

import base64
import io
import secrets
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from server import PromptServer


def smoothstep(low, high, x):
    t = ((x - low) / (high - low)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def rgb_to_oklab(rgb):
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    r, g, b = linear.unbind(-1)
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b).clamp_min(0).pow(1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b).clamp_min(0).pow(1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b).clamp_min(0).pow(1/3)
    return torch.stack((0.2104542553*l + 0.793617785*m - 0.0040720468*s,
                        1.9779984951*l - 2.428592205*m + 0.4505937099*s,
                        0.0259040371*l + 0.7827717662*m - 0.808675766*s), -1)


def oklab_to_rgb(lab):
    L, a, b = lab.unbind(-1)
    l = (L + 0.3963377774*a + 0.2158037573*b).pow(3)
    m = (L - 0.1055613458*a - 0.0638541728*b).pow(3)
    s = (L - 0.0894841775*a - 1.291485548*b).pow(3)
    linear = torch.stack((4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
                          -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
                          -0.0041960863*l - 0.7034186147*m + 1.707614701*s), -1)
    rgb = torch.where(linear <= 0.0031308, 12.92*linear,
                      1.055*linear.clamp_min(0).pow(1/2.4) - 0.055)
    return rgb.clamp(0, 1)


def skin_selection(rgb, tolerance):
    r, g, b = rgb.unbind(-1)
    maximum, winner = rgb.max(-1)
    delta = maximum - rgb.min(-1).values
    saturation = delta / maximum.clamp_min(1e-6)
    d = delta.clamp_min(1e-6)
    hue = torch.where(winner == 0, (g-b)/d,
                      torch.where(winner == 1, (b-r)/d+2, (r-g)/d+4)) * 60
    distance = torch.abs((hue - 3 + 180) % 360 - 180)
    width = 7 + tolerance * 0.14
    hue_weight = 1 - smoothstep(width, width + 8, distance)
    sat_weight = smoothstep(0.045, 0.105, saturation) * (1-smoothstep(0.32, 0.58, saturation))
    light_weight = smoothstep(0.12, 0.30, maximum) * (1-smoothstep(0.96, 1.0, maximum))
    return hue_weight * sat_weight * light_weight


def auto_skin_selection(rgb, tolerance):
    # Conservative pink/peach selection inside an automatically detected person.
    r, g, b = rgb.unbind(-1)
    maximum, winner = rgb.max(-1)
    delta = maximum-rgb.min(-1).values
    saturation = delta/maximum.clamp_min(1e-6)
    d = delta.clamp_min(1e-6)
    hue = torch.where(winner == 0, (g-b)/d,
                      torch.where(winner == 1, (b-r)/d+2, (r-g)/d+4))*60
    hue = (hue+180)%360-180
    width = 8 + tolerance*0.10
    hue_weight = (1-smoothstep(width, width+8, hue)) * smoothstep(-width-6, -width+2, hue)
    sat_weight = smoothstep(0.075, 0.16, saturation) * (1-smoothstep(0.24, 0.46, saturation))
    red_weight = smoothstep(0.015, 0.055, r-g)
    light_weight = smoothstep(0.22, 0.45, maximum) * (1-smoothstep(0.96, 1.0, maximum))
    return hue_weight * sat_weight * red_weight * light_weight


def soften_mask(mask, edge_softness):
    sigma = min(mask.shape[-2:]) * edge_softness / 100
    if sigma < 0.25:
        return mask
    radius = min(int(sigma * 3 + 0.5), 64)
    positions = torch.arange(-radius, radius+1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-0.5*(positions/sigma).square())
    kernel /= kernel.sum()
    x = mask.unsqueeze(1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1,1,1,-1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1,1,-1,1))
    return x[:, 0].clamp(0, 1)


def grade_skin(image, de_pink=0, brightness=0, saturation=0, tolerance=50,
               edge_softness=0.35, skin_mask=None, person_mask=None):
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("需要 RGB 或 RGBA IMAGE 输入。")
    rgb = image[..., :3].float().clamp(0, 1)
    if skin_mask is None:
        mask = skin_selection(rgb, tolerance) if person_mask is None else auto_skin_selection(rgb, tolerance)
    else:
        mask = skin_mask.to(device=image.device, dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] not in (1, image.shape[0]):
            raise ValueError("遮罩批次数必须为 1，或与图片批次数一致。")
        if mask.shape[-2:] != image.shape[1:3]:
            mask = F.interpolate(mask.unsqueeze(1), size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
        mask = mask.clamp(0, 1).expand(image.shape[0], -1, -1)
    if person_mask is not None:
        region = person_mask.to(device=image.device, dtype=torch.float32)
        if region.ndim == 2:
            region = region.unsqueeze(0)
        if region.ndim != 3 or region.shape[0] not in (1, image.shape[0]):
            raise ValueError("人物范围批次数必须为 1，或与图片批次数一致。")
        if region.shape[-2:] != image.shape[1:3]:
            region = F.interpolate(region.unsqueeze(1), size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
        mask = mask * region.clamp(0, 1)
    # Feather the combined selection; multiplying by a hard person mask afterwards
    # reintroduces the detector's box edges into both grading nodes.
    mask = soften_mask(mask, edge_softness)
    if de_pink == 0 and brightness == 0 and saturation == 0:
        return image, mask
    lab = rgb_to_oklab(rgb)
    L, a, b = lab.unbind(-1)
    # Rotate positive skin chroma gently towards peach; never subtract red into green.
    angle = torch.atan2(b, a)
    chroma = torch.sqrt(a.square() + b.square())
    target = torch.full_like(angle, 0.70)
    rotation = (target-angle).clamp(0, 0.65) * (de_pink / 100)
    adjusted_chroma = chroma * (1 + saturation/100)
    adjusted_a = adjusted_chroma * torch.cos(angle+rotation)
    adjusted_b = adjusted_chroma * torch.sin(angle+rotation)
    midtone = 4 * L * (1-L)
    adjusted_L = (L + brightness/100 * 0.45 * midtone).clamp(0, 1)
    adjusted = oklab_to_rgb(torch.stack((adjusted_L, adjusted_a, adjusted_b), -1))
    result_rgb = rgb + mask.unsqueeze(-1) * (adjusted-rgb)
    if image.shape[-1] == 4:
        result = torch.cat((result_rgb.to(image.dtype), image[..., 3:]), -1)
    else:
        result = result_rgb.to(image.dtype)
    # Exact preservation of unselected pixels is useful for externally supplied masks.
    result = torch.where(mask.unsqueeze(-1) == 0, image, result)
    return result, mask


def encode_png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class ThumbnailCache:
    """Only compressed CPU thumbnails; no model tensors or full-size images retained."""
    def __init__(self):
        self.entries = OrderedDict()
        self.max_bytes = 8 * 1024 * 1024

    def add(self, owner, rgb, mask, region=None):
        for key in list(self.entries):
            if self.entries[key][0] == owner:
                del self.entries[key]
        key = secrets.token_urlsafe(24)
        self.entries[key] = (owner, rgb, mask, region)
        while len(self.entries) > 8 or sum(len(x[1])+len(x[2] or b"")+len(x[3] or b"") for x in self.entries.values()) > self.max_bytes:
            self.entries.popitem(last=False)
        return key


_thumbnails = ThumbnailCache()


class AnimaSkinTone:
    DESCRIPTION = "针对灰粉、偏紫的二次元肤色。三个调整值为 0 时原图不变。自动选区按颜色判断，可能包含相近颜色的衣服；接入 skin_mask 可精确限制。实时预览仅显示批次第一张，保存结果需要再次运行工作流。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "de_pink": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "去粉紫：将皮肤色相温和推向浅桃色，0 不调整。"}),
            "brightness": ("FLOAT", {"default": 0, "min": -30, "max": 30, "step": 1, "tooltip": "肤色提亮：主要改变皮肤中间调，正值提亮。"}),
            "saturation": ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 1, "tooltip": "肤色浓淡：负值减淡，正值加浓；不是全图饱和度。"}),
            "tolerance": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "选区宽容度：越大越容易包含黄色或紫色附近的区域。接外部遮罩时无效。"}),
            "edge_softness": ("FLOAT", {"default": 0.35, "min": 0, "max": 2, "step": 0.05, "tooltip": "边缘柔化：相对图片短边的百分比。0 不柔化。"}),
        }, "optional": {"skin_mask": ("MASK", {"tooltip": "可选皮肤遮罩；连接后替代自动选区。白色调整，黑色保留。"})},
           "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "apply"
    CATEGORY = "Anima/肤色调整"

    def apply(self, image, de_pink, brightness, saturation, tolerance, edge_softness,
              skin_mask=None, unique_id=None, extra_pnginfo=None, person_mask=None):
        result, selection = grade_skin(image, de_pink, brightness, saturation, tolerance, edge_softness, skin_mask, person_mask)
        preview = Image.fromarray((image[0, ..., :3].detach().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8))
        preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
        raw_mask = None
        if skin_mask is not None:
            m = skin_mask if skin_mask.ndim == 2 else skin_mask[0]
            mask_image = Image.fromarray((m.detach().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8))
            mask_image = mask_image.resize(preview.size, Image.Resampling.BILINEAR)
            raw_mask = encode_png(np.asarray(mask_image))
        raw_region = None
        if person_mask is not None:
            m = person_mask if person_mask.ndim == 2 else person_mask[0]
            region_image = Image.fromarray((m.detach().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8))
            region_image = region_image.resize(preview.size, Image.Resampling.BILINEAR)
            raw_region = encode_png(np.asarray(region_image))
        png = encode_png(np.asarray(preview))
        workflow_id = (extra_pnginfo or {}).get("workflow", {}).get("id", "api")
        key = _thumbnails.add((str(workflow_id), str(unique_id)), png, raw_mask, raw_region)
        return {"ui": {"skin_preview_key": [key], "skin_original": [data_url(png)]}, "result": (result, selection)}


@PromptServer.instance.routes.post("/anima_skin_tone/preview")
async def preview(request):
    data = await request.json()
    entry = _thumbnails.entries.get(data.get("key"))
    if entry is None:
        return web.json_response({"error": "预览已过期，请运行一次工作流。"}, status=404)
    values = {}
    for name, low, high, default in (("de_pink",0,100,0), ("brightness",-30,30,0),
                                     ("saturation",-50,50,0), ("tolerance",0,100,50),
                                     ("edge_softness",0,2,0.35)):
        value = data.get(name, default)
        if not isinstance(value, (int,float)) or not np.isfinite(value) or not low <= value <= high:
            return web.json_response({"error": "参数超出范围："+name}, status=400)
        values[name] = float(value)
    source = torch.from_numpy(np.array(Image.open(io.BytesIO(entry[1])).convert("RGB"))).float().unsqueeze(0)/255
    mask = None
    if entry[2] is not None:
        mask = torch.from_numpy(np.array(Image.open(io.BytesIO(entry[2])))).float().unsqueeze(0)/255
    region = None
    if entry[3] is not None:
        region = torch.from_numpy(np.array(Image.open(io.BytesIO(entry[3])))).float().unsqueeze(0)/255
    result, selection = grade_skin(source, skin_mask=mask, person_mask=region, **values)
    rgb = (result[0].numpy().clip(0,1)*255).round().astype(np.uint8)
    overlay = source[0].numpy().copy()
    alpha = selection[0].numpy()[..., None] * 0.65
    overlay = overlay*(1-alpha) + np.array([0.1,1.0,0.3])*alpha
    overlay = (overlay.clip(0,1)*255).round().astype(np.uint8)
    return web.json_response({"result": data_url(encode_png(rgb)), "selection": data_url(encode_png(overlay))})


def detect_person_region(image, segm_detector):
    height, width = image.shape[1:3]
    scale = min(1.0, 1024 / max(height, width))
    detect_h, detect_w = max(1, round(height * scale)), max(1, round(width * scale))
    margin = max(1, round(max(detect_h, detect_w) * 0.03))
    side = max(detect_h, detect_w) + 2 * margin
    left, top = (side-detect_w)//2, (side-detect_h)//2
    region = torch.zeros(image.shape[:3], device=image.device, dtype=torch.float32)
    for index in range(image.shape[0]):
        source = image[index:index+1, ..., :3].float().movedim(-1, 1)
        if (detect_h, detect_w) != (height, width):
            source = F.interpolate(source, size=(detect_h, detect_w), mode="bilinear", align_corners=False)
        # Replicated context moves frame-edge limbs inside the detection canvas.
        # A square canvas also avoids aspect-ratio mask remapping in the provider.
        padded = F.pad(source, (left, side-detect_w-left, top, side-detect_h-top), mode="replicate")
        segs = segm_detector.detect(padded.movedim(1, -1), 0.35, 0, 1.0, drop_size=10)
        canvas = torch.zeros((side, side), device=image.device, dtype=torch.float32)
        for seg in segs[1]:
            if seg.label != "person":
                continue
            x1, y1, x2, y2 = seg.crop_region
            crop = torch.as_tensor(seg.cropped_mask, device=image.device, dtype=torch.float32)
            canvas[y1:y2, x1:x2] = torch.maximum(canvas[y1:y2, x1:x2], crop)
        cropped = canvas[top:top+detect_h, left:left+detect_w]
        region[index] = F.interpolate(cropped[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return region


class AnimaAutoSkinTone(AnimaSkinTone):
    DESCRIPTION = "自动检测 person 范围，再筛选粉桃肤色。换图后运行即可，不用手工遮罩。无人物检测结果时保持原图。相近颜色的衣物、动物或暖色反光仍可能误选；不是精确人体解析。预览只显示批次第一张。"

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types.pop("optional")
        types["required"]["segm_detector"] = ("SEGM_DETECTOR", {"tooltip": "连接 UltralyticsDetectorProvider 的 SEGM_DETECTOR，使用已有 person_yolov8m-seg.pt。"})
        types["required"]["brightness"][1]["default"] = 8
        types["required"]["de_pink"][1]["default"] = 65
        return types

    def apply(self, image, de_pink, brightness, saturation, tolerance, edge_softness,
              segm_detector, unique_id=None, extra_pnginfo=None):
        region = detect_person_region(image, segm_detector)
        return super().apply(image, de_pink, brightness, saturation, tolerance, edge_softness,
                             unique_id=unique_id, extra_pnginfo=extra_pnginfo, person_mask=region)


NODE_CLASS_MAPPINGS = {"AnimaSkinTone": AnimaSkinTone, "AnimaAutoSkinTone": AnimaAutoSkinTone}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaSkinTone": "Anima 肤色调整", "AnimaAutoSkinTone": "Anima 自动肤色调整"}
