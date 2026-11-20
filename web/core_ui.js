import { app } from "../../scripts/app.js";

// Clean English UI labels and styling
const LABELS = {
    H3VMCoreEngine: {
        title: "H3VM Core | Model Execution Engine",
        model: "Model",
        multi_gpu_enabled: "Multi-GPU",
        dual_vae_enabled: "Dual-GPU VAE",
        gpu_mode: "Mode",
        mode4_predictor: "Acceleration Predictor",
        gpu_participation: "Secondary GPU Load",
        custom_participation: "Custom Load",
        expected_steps_hint: "Steps Hint",
        telemetry: "Runtime Monitor",
        attention_kernel: "Attention Kernel",
        capacity_profile: "Capacity VRAM Profile",
        custom_split_primary: "Custom Primary Split % (e.g. 57)",
        chunk_size: "Activation Chunk Size",
        mode: "Mode",
        vae_policy: "VAE Policy",
    },
    H3VMModeAwareVideoVAEDecode: {
        title: "H3VM Video VAE | Dual-GPU Decoder",
        samples: "Latent",
        vae: "Video VAE",
        mode: "H3VM Mode",
        vae_policy: "VAE Policy",
        vae_name: "Secondary VAE File (same weights)",
        images: "Images",
    },
};

const VALUE_MIGRATIONS = {
    gpu_mode: {
        "双卡极速": "Multi-GPU Fast",
        "双卡扩容": "Multi-GPU Capacity",
        "双卡后台": "Multi-GPU Background",
    },
    mode4_predictor: {
        "SPECTRAL｜频谱极速": "SPECTRAL | Fast",
        "LINEAR｜标准极速": "LINEAR | Standard",
    },
    gpu_participation: {
        "全｜100%": "Full | 100%",
        "高｜75%": "High | 75%",
        "中｜50%": "Medium | 50%",
        "低｜25%": "Low | 25%",
        "自定义": "Custom",
    },
};

function applyLabels(node, nodeName) {
    const dict = LABELS[nodeName];
    if (!dict) return;
    node.title = dict.title;

    // Remove legacy ui_language widget if restored from an old workflow
    if (node.widgets) {
        const langIdx = node.widgets.findIndex((w) => w.name === "ui_language");
        if (langIdx !== -1) {
            node.widgets.splice(langIdx, 1);
        }
    }

    for (const widget of node.widgets || []) {
        if (dict[widget.name]) widget.label = dict[widget.name];

        // Migrate any legacy Chinese values saved in the workflow
        const migrations = VALUE_MIGRATIONS[widget.name];
        if (migrations && migrations[widget.value]) {
            widget.value = migrations[widget.value];
        }

        // Clean combo dropdown option list if still in Chinese
        if (widget.type === "combo" && Array.isArray(widget.options?.values)) {
            widget.options.values = widget.options.values.map((v) => {
                const map = VALUE_MIGRATIONS[widget.name];
                return map && map[v] ? map[v] : v;
            });
        }
    }

    for (const input of node.inputs || []) {
        if (dict[input.name]) input.label = dict[input.name];
    }
    for (const output of node.outputs || []) {
        if (dict[output.name]) output.label = dict[output.name];
    }
    for (const widget of node.widgets || []) {
        if (!["multi_gpu_enabled", "telemetry", "dual_vae_enabled"].includes(widget.name)) continue;
        widget.options = widget.options || {};
        widget.options.label_on = "On";
        widget.options.label_off = "Off";
        widget.options.on = "On";
        widget.options.off = "Off";
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "H3VM.CoreEnglishUI",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData?.name;
        if (!LABELS[nodeName]) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;
        const install = function () {
            applyLabels(this, nodeName);
        };
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            install.call(this);
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            install.call(this);
            return result;
        };
    },
    nodeCreated(node) {
        if (LABELS[node.comfyClass]) {
            applyLabels(node, node.comfyClass);
        }
    },
});
