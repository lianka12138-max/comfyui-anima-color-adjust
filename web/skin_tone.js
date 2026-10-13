import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const definitions = {
  AnimaSkinTone: {
    labels: { de_pink: "去粉紫", brightness: "肤色提亮", saturation: "肤色浓淡", tolerance: "选区宽容度", edge_softness: "边缘柔化 (%)" },
    reset: { de_pink: 0, brightness: 0, saturation: 0 },
    endpoint: "/anima_skin_tone/preview", key: "skin_preview_key", original: "skin_original", selectionLabel: "肤色选区",
    selectionHelp: "右侧绿色表示调整范围。误选时降低宽容度，或接入皮肤遮罩。",
  },
  AnimaSceneTone: {
    labels: { midtones: "中间调通透", shadows: "暗部提亮", remove_cold: "去冷灰", warmth: "中间调暖意", strength: "整体强度 (%)" },
    reset: { midtones: 0, shadows: 0, remove_cold: 0, warmth: 0, strength: 100 },
    endpoint: "/anima_scene_tone/preview", key: "scene_preview_key", original: "scene_original", selectionLabel: "调整选区",
    selectionHelp: "右侧绿色表示调整范围。接入肤色节点的 skin_mask 可限制范围；上游改动后需重新运行。",
  },
};

definitions.AnimaAutoSkinTone = {
  ...definitions.AnimaSkinTone,
  selectionHelp: "绿色为自动人物范围内筛出的肤色。换图后运行即可；相似色衣料仍可能被选中。",
};

app.registerExtension({
  name: "Anima.SkinTone.Preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const config = definitions[nodeData.name];
    if (!config) return;
    const labels = config.labels;
    const created = nodeType.prototype.onNodeCreated;
    const executed = nodeType.prototype.onExecuted;
    const removed = nodeType.prototype.onRemoved;

    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const node = this;
      let previewKey = null, original = "", timer = null, controller = null, sequence = 0;
      const panel = document.createElement("div");
      panel.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;height:100%;background:#202526;color:#eee;font:12px sans-serif;overflow:hidden;";
      const toolbar = document.createElement("div");
      toolbar.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;";
      const mode = document.createElement("select");
      mode.setAttribute("aria-label", "预览模式");
      for (const text of ["原图 / 调整后", config.selectionLabel]) {
        const option = document.createElement("option"); option.textContent = text; mode.append(option);
      }
      const reset = document.createElement("button"); reset.textContent = "调色归零";
      const status = document.createElement("div");
      status.textContent = "先运行一次。绿色表示选区；预览只显示批次第一张。";
      status.style.cssText = "line-height:1.5;white-space:normal;";
      const images = document.createElement("div");
      images.style.cssText = "display:flex;gap:4px;flex:1;min-height:0;";
      const before = document.createElement("img");
      const after = document.createElement("img");
      for (const img of [before, after]) {
        img.style.cssText = "width:calc(50% - 2px);min-width:0;height:100%;object-fit:contain;visibility:hidden;";
        img.alt = "运行后显示预览"; images.append(img);
      }
      before.alt = "原图"; after.alt = "调整后";
      toolbar.append(mode, reset); panel.append(toolbar, images, status);
      const dom = node.addDOMWidget("skin_preview", "anima_skin_preview", panel, { serialize: false, hideOnZoom: true });
      dom.computeSize = () => [420, 370];
      let latest = null;
      function render() {
        if (original) { before.src = original; before.style.visibility = "visible"; }
        if (latest) { after.src = mode.selectedIndex === 1 ? latest.selection : latest.result; after.style.visibility = "visible"; }
        after.alt = mode.selectedIndex === 1 ? config.selectionLabel + "（绿色）" : "调整后";
      }
      async function refresh() {
        if (!previewKey) return;
        const own = ++sequence;
        controller?.abort(); controller = new AbortController();
        const payload = { key: previewKey };
        for (const name of Object.keys(labels)) payload[name] = node.widgets.find(w => w.name === name)?.value;
        status.textContent = "更新预览…";
        try {
          const response = await api.fetchApi(config.endpoint, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload), signal: controller.signal,
          });
          const data = await response.json();
          if (own !== sequence) return;
          if (!response.ok) throw new Error(data.error || "预览失败");
          latest = data; render();
          status.textContent = "左：原图　右：" + (mode.selectedIndex === 1 ? "选区（绿色）" : "调整后") + "。" + (data.masked === false ? "未接遮罩，当前调整全图。" : "") + "再次运行才能保存结果。";
        } catch (error) {
          if (error.name !== "AbortError" && own === sequence) status.textContent = error.message;
        }
      }
      function schedule() {
        ++sequence; controller?.abort(); clearTimeout(timer);
        timer = setTimeout(refresh, 180);
      }
      for (const widget of node.widgets) {
        if (!labels[widget.name]) continue;
        widget.label = labels[widget.name];
        const callback = widget.callback;
        widget.callback = function () { const value = callback?.apply(this, arguments); schedule(); return value; };
      }
      mode.onchange = () => {
        render();
        status.textContent = mode.selectedIndex === 1 ? config.selectionHelp : "左：原图　右：调整后。再次运行才能保存结果。";
      };
      reset.onclick = () => {
        for (const [name,value] of Object.entries(config.reset)) node.widgets.find(w => w.name === name).value = value;
        node.setDirtyCanvas(true, true); schedule();
      };
      node._skinAccept = (message) => {
        previewKey = message[config.key]?.[0];
        original = message[config.original]?.[0] || "";
        render(); schedule();
      };
      node._skinCleanup = () => { ++sequence; clearTimeout(timer); controller?.abort(); panel.remove(); };
      node.setSize([Math.max(node.size[0], 460), Math.max(node.size[1], 620)]);
    };
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments); this._skinAccept?.(message);
    };
    nodeType.prototype.onRemoved = function () {
      this._skinCleanup?.(); removed?.apply(this, arguments);
    };
  },
});
