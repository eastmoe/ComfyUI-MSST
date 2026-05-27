import { app } from "../../scripts/app.js";

let modelInfoPromise = null;
let modelInfoData = null;

function loadModelInfo() {
    if (!modelInfoPromise) {
        modelInfoPromise = fetch("/comfy-msst/models-info")
            .then((response) => response.ok ? response.json() : { models: {} })
            .then((data) => {
                modelInfoData = data || { models: {} };
                return modelInfoData;
            })
            .catch(() => {
                modelInfoData = { models: {} };
                return modelInfoData;
            });
    }
    return modelInfoPromise;
}

function formatEntry(entry) {
    if (!entry) {
        return ["选择模型后显示模型信息"];
    }

    const stems = Array.isArray(entry.stems) && entry.stems.length ? entry.stems.join(", ") : "-";
    return [
        `分类: ${entry.model_class || "-"}`,
        `支持音轨: ${stems}`,
        `体积: ${entry.size || "-"}`,
        `备注: ${entry.note || "-"}`,
        `推荐星级: ${entry.rating || "-"}`,
    ];
}

function stemValues(entry) {
    const stems = Array.isArray(entry?.stems) ? entry.stems.filter(Boolean) : [];
    const values = stems.length ? stems : ["auto"];
    return Array.from(new Set(["auto", ...values, "custom"]));
}

function normalizeModelKey(value) {
    return String(value || "").replaceAll("\\", "/").replace(/^\.\//, "");
}

function uniqueEntry(candidates) {
    const unique = Array.from(new Set(candidates.filter(Boolean)));
    return unique.length === 1 ? unique[0] : null;
}

function findOnlineModelEntry(data, selected) {
    if (!selected) {
        return null;
    }

    const models = data.models || {};
    const normalized = normalizeModelKey(selected);
    if (models[normalized]) {
        return models[normalized];
    }

    const candidates = [];
    for (const [key, value] of Object.entries(models)) {
        const normalizedKey = normalizeModelKey(key);
        const choiceKey = normalizeModelKey(value?.choice_key || `${value?.model_class || ""}/${value?.model_name || ""}`);
        if (normalized === normalizedKey || normalized === choiceKey) {
            candidates.push(value);
        }
    }

    return uniqueEntry(candidates);
}

function currentOnlineInfoLines(node) {
    if (!modelInfoData) {
        loadModelInfo().then(() => app.graph.setDirtyCanvas(true, true));
        return ["模型信息加载中..."];
    }
    const modelWidget = node.widgets?.find((widget) => widget.name === "model");
    return formatEntry(findOnlineModelEntry(modelInfoData, modelWidget?.value));
}

function findModelEntry(data, selected) {
    if (!selected) {
        return null;
    }

    const models = data.models || {};
    const normalized = normalizeModelKey(selected);
    const withoutPretrain = normalized.replace(/^pretrain\//, "");
    const filename = normalized.split("/").pop();
    const candidates = [];
    for (const [key, value] of Object.entries(models)) {
        const normalizedKey = normalizeModelKey(key);
        const choiceKey = normalizeModelKey(value?.choice_key || `${value?.model_class || ""}/${value?.model_name || ""}`);
        const modelName = normalizeModelKey(value?.model_name);
        const targetPosition = normalizeModelKey(value?.target_position);
        const localPath = normalizeModelKey(value?.local_path);
        const isExact =
            normalizedKey === normalized ||
            normalizedKey === withoutPretrain ||
            choiceKey === normalized ||
            modelName === normalized ||
            modelName === filename ||
            targetPosition === normalized ||
            targetPosition === `pretrain/${withoutPretrain}` ||
            localPath === normalized;
        if (isExact) {
            candidates.push(value);
        }
    }

    return uniqueEntry(candidates);
}

function sourceNodeForInput(node, inputName) {
    const input = node.inputs?.find((item) => item.name === inputName || item.label === inputName || item.localized_name === inputName || item.name === "模型信息");
    const linkId = input?.link;
    if (linkId == null) {
        return null;
    }

    const link = app.graph.links?.[linkId] || app.graph.links?.find?.((item) => item?.id === linkId);
    if (!link) {
        return null;
    }
    return app.graph.getNodeById?.(link.origin_id) || null;
}

function selectedModelFromNode(node) {
    const widgetNames = ["model", "model_name", "model_path"];
    for (const name of widgetNames) {
        const widget = node?.widgets?.find((item) => item.name === name);
        if (widget?.value) {
            return widget.value;
        }
    }
    return null;
}

function setComboValues(widget, values) {
    if (!widget || !Array.isArray(values) || !values.length) {
        return;
    }

    widget.options = widget.options || {};
    if (Array.isArray(widget.options.values)) {
        widget.options.values.splice(0, widget.options.values.length, ...values);
    } else {
        widget.options.values = [...values];
    }
    if (Array.isArray(widget.options.items)) {
        widget.options.items.splice(0, widget.options.items.length, ...values);
    } else if (widget.options.items) {
        widget.options.items = [...values];
    }
    if (Array.isArray(widget.values)) {
        widget.values.splice(0, widget.values.length, ...values);
    }
    if (!values.includes(widget.value)) {
        widget.value = values[0];
    }
    widget.callback?.(widget.value);
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

function fitLine(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) {
        return text;
    }
    let value = text;
    while (value.length > 1 && ctx.measureText(`${value}...`).width > maxWidth) {
        value = value.slice(0, -1);
    }
    return `${value}...`;
}

function addInfoWidget(node) {
    const infoWidgets = node.widgets?.filter((widget) => widget.type === "comfy_msst_model_info") || [];
    const existing = infoWidgets[0];
    if (existing) {
        existing.comfyMsstSourceNode = node;
        for (const duplicate of infoWidgets.slice(1)) {
            const index = node.widgets.indexOf(duplicate);
            if (index >= 0) {
                node.widgets.splice(index, 1);
            }
        }
        return existing;
    }

    const widget = {
        name: "模型信息",
        type: "comfy_msst_model_info",
        comfyMsstSourceNode: node,
        lines: ["选择模型后显示模型信息"],
        computeSize(width) {
            return [width, Math.max(92, 24 + this.lines.length * 17)];
        },
        draw(ctx, _node, width, y, height) {
            this.lines = currentOnlineInfoLines(this.comfyMsstSourceNode || _node);
            const x = 10;
            const boxWidth = width - 20;
            ctx.save();
            drawRoundedRect(ctx, x, y + 2, boxWidth, height - 6, 6);
            ctx.fillStyle = "#262626";
            ctx.fill();
            ctx.strokeStyle = "#4a4a4a";
            ctx.stroke();

            ctx.fillStyle = "#d7d7d7";
            ctx.font = "12px sans-serif";
            let textY = y + 22;
            for (const line of this.lines) {
                ctx.fillText(fitLine(ctx, line, boxWidth - 20), x + 10, textY);
                textY += 17;
            }
            ctx.restore();
        },
    };
    node.addCustomWidget(widget);
    return widget;
}

function refreshNodeInfo(node, infoWidget) {
    loadModelInfo().then((data) => {
        infoWidget.lines = currentOnlineInfoLines(node);
        node.setSize([node.size[0], node.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);
    });
}

function refreshStemChoices(node) {
    const stemWidget = node.widgets?.find((widget) => widget.name === "stem_name");
    if (!stemWidget) {
        return;
    }

    if (!stemWidget.comfyMsstAllValues) {
        stemWidget.comfyMsstAllValues = [...(stemWidget.options?.values || [])];
    }

    const source = sourceNodeForInput(node, "model_info");
    const selected = selectedModelFromNode(source);
    const signature = source ? `${source.id}:${selected || ""}` : "none";
    if (node.comfyMsstStemSignature === signature) {
        return;
    }
    node.comfyMsstStemSignature = signature;

    if (!source) {
        setComboValues(stemWidget, stemWidget.comfyMsstAllValues);
        app.graph.setDirtyCanvas(true, true);
        return;
    }

    loadModelInfo().then((data) => {
        const entry = findModelEntry(data, selected);
        setComboValues(stemWidget, entry ? stemValues(entry) : stemWidget.comfyMsstAllValues);
        app.graph.setDirtyCanvas(true, true);
    });
}

function refreshConnectedStemNodes(sourceNode) {
    const linkIds = sourceNode.outputs?.flatMap((output) => output.links || []) || [];
    for (const linkId of linkIds) {
        const link = app.graph.links?.[linkId] || app.graph.links?.find?.((item) => item?.id === linkId);
        const target = link ? app.graph.getNodeById?.(link.target_id) : null;
        if (target?.type === "ComfyMSSTGetStem" || target?.constructor?.comfyClass === "ComfyMSSTGetStem") {
            target.comfyMsstStemSignature = "";
            refreshStemChoices(target);
        }
    }
}

function wrapModelWidgetCallback(node, afterChange) {
    const modelWidget = node.widgets?.find((widget) => ["model", "model_name", "model_path"].includes(widget.name));
    if (!modelWidget) {
        return;
    }

    modelWidget.comfyMsstAfterChangeCallbacks = modelWidget.comfyMsstAfterChangeCallbacks || [];
    if (afterChange && !modelWidget.comfyMsstAfterChangeCallbacks.includes(afterChange)) {
        modelWidget.comfyMsstAfterChangeCallbacks.push(afterChange);
    }

    if (modelWidget.comfyMsstWrapped) {
        return;
    }

    const originalCallback = modelWidget.callback;
    modelWidget.callback = (...args) => {
        const callbackResult = originalCallback?.apply(modelWidget, args);
        setTimeout(() => {
            for (const callback of modelWidget.comfyMsstAfterChangeCallbacks || []) {
                callback();
            }
            refreshConnectedStemNodes(node);
        }, 0);
        return callbackResult;
    };
    modelWidget.comfyMsstWrapped = true;
}

function isNodeClass(node, className) {
    return node?.type === className || node?.comfyClass === className || node?.constructor?.comfyClass === className;
}

function isModelInfoSourceNode(node) {
    return [
        "ComfyMSSTOnlineModelLoader",
        "ComfyMSSTModelFromCatalog",
        "ComfyMSSTModelFromPaths",
        "ComfyMSSTVRModelFromCatalog",
        "ComfyMSSTVRModelFromPath",
    ].some((className) => isNodeClass(node, className));
}

function enhanceGetStemNode(node) {
    if (!isNodeClass(node, "ComfyMSSTGetStem") || node.comfyMsstGetStemEnhanced) {
        return;
    }
    node.comfyMsstGetStemEnhanced = true;

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        this.comfyMsstStemSignature = "";
        setTimeout(() => refreshStemChoices(this), 0);
        return result;
    };

    const onGraphConfigured = node.onGraphConfigured;
    node.onGraphConfigured = function () {
        const result = onGraphConfigured?.apply(this, arguments);
        this.comfyMsstStemSignature = "";
        setTimeout(() => refreshStemChoices(this), 0);
        return result;
    };

    const onDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function () {
        const result = onDrawForeground?.apply(this, arguments);
        refreshStemChoices(this);
        return result;
    };

    setTimeout(() => refreshStemChoices(node), 0);
}

app.registerExtension({
    name: "Comfy.MSST.ModelInfo",
    nodeCreated(node) {
        if (isModelInfoSourceNode(node)) {
            wrapModelWidgetCallback(node, () => {});
        }
        enhanceGetStemNode(node);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ComfyMSSTOnlineModelLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                const infoWidget = addInfoWidget(this);
                wrapModelWidgetCallback(this, () => refreshNodeInfo(this, infoWidget));
                refreshNodeInfo(this, infoWidget);
                return result;
            };
            return;
        }

        if (["ComfyMSSTModelFromCatalog", "ComfyMSSTModelFromPaths", "ComfyMSSTVRModelFromCatalog", "ComfyMSSTVRModelFromPath"].includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                wrapModelWidgetCallback(this, () => {});
                return result;
            };
            return;
        }

        if (nodeData.name === "ComfyMSSTGetStem") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                setTimeout(() => refreshStemChoices(this), 0);
                return result;
            };

            const onGraphConfigured = nodeType.prototype.onGraphConfigured;
            nodeType.prototype.onGraphConfigured = function () {
                const result = onGraphConfigured?.apply(this, arguments);
                this.comfyMsstStemSignature = "";
                setTimeout(() => refreshStemChoices(this), 0);
                return result;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const result = onConnectionsChange?.apply(this, arguments);
                this.comfyMsstStemSignature = "";
                setTimeout(() => refreshStemChoices(this), 0);
                return result;
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function () {
                const result = onDrawForeground?.apply(this, arguments);
                refreshStemChoices(this);
                return result;
            };
        }
    },
});
