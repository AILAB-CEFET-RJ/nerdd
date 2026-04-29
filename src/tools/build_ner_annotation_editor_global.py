#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an interactive HTML editor for manual review and correction of NER spans.

Enhancements over the basic editor:
- explicit global action: remove the same mention+label everywhere;
- explicit global action: change the same mention+old-label to a new label everywhere;
- preview of how many spans/records each global action will affect;
- export of reusable global cleanup rules;
- optional initial banlist applied at startup.

The editor works fully in-browser after the HTML is generated. The exported corrected
corpus is downloaded by the browser as JSON.

Usage example:
=============
python src/tools/build_ner_annotation_editor_global.py \
  --input data/dd_corpus_small_train.json \
  --output artifacts/annotation_review/dd_corpus_small_train_editor_global.html \
  --title "Review global - dd_corpus_small_train" \
  --labels Person,Location,Organization \
  --banlist-output annotation_banlist_train.json \
  --rules-output annotation_global_rules_train.json

"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


DEFAULT_LABELS = ["Person", "Location", "Organization"]

PALETTE = [
    "#0B6E4F",
    "#1D4E89",
    "#8B1E3F",
    "#8C510A",
    "#5C2E91",
    "#006D77",
    "#264653",
    "#7A3E00",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive HTML editor for NER annotations."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input corpus file (JSON list/object or JSONL).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML editor file.",
    )
    parser.add_argument(
        "--title",
        default="NER Annotation Editor",
        help="Page title.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label list (example: Person,Location,Organization).",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=0,
        help="Maximum number of records to include (0 = all).",
    )
    parser.add_argument(
        "--banlist-input",
        default="",
        help="Optional JSON file with initial remove rules: {label: [normalized_terms...]}",
    )
    parser.add_argument(
        "--banlist-output",
        default="annotation_banlist.json",
        help="Default filename used when exporting remove rules from the HTML editor.",
    )
    parser.add_argument(
        "--rules-output",
        default="annotation_global_rules.json",
        help="Default filename used when exporting all global cleanup rules from the HTML editor.",
    )
    return parser.parse_args()


def _parse_jsonl(text: str) -> list[dict]:
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
    return rows


def load_corpus(path: str | Path) -> list[dict]:
    payload = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return _parse_jsonl(payload)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("Unsupported input format: expected JSON object/list or JSONL.")


def normalize_span(span: dict) -> dict | None:
    start = span.get("start")
    end = span.get("end")
    label = span.get("label")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    if end <= start:
        return None
    normalized = {"start": start, "end": end, "label": label.strip()}
    # Preserve useful metadata if present, but the editor only manipulates offsets/label.
    for key in ("text", "seed_origin", "confidence", "score", "source"):
        if key in span:
            normalized[key] = span[key]
    return normalized


def get_text(record: dict) -> str:
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_spans(record: dict) -> list:
    spans = record.get("spans")
    if isinstance(spans, list):
        return spans
    entities = record.get("entities")
    if isinstance(entities, list):
        return entities
    ner = record.get("ner")
    if isinstance(ner, list):
        return ner
    return []


def sanitize_spans(text: str, spans: list) -> list[dict]:
    valid = []
    for raw in spans:
        if not isinstance(raw, dict):
            continue
        span = normalize_span(raw)
        if not span:
            continue
        if span["start"] < 0 or span["end"] > len(text):
            continue
        valid.append(span)
    return sorted(valid, key=lambda s: (s["start"], s["end"], s["label"]))


def normalize_records(rows: list[dict]) -> list[dict]:
    normalized = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        out = dict(row)
        text = get_text(out)
        out["text"] = text
        out["spans"] = sanitize_spans(text, get_spans(out))
        # Keep a stable reference for review even if sample_id is absent.
        if "_editor_row_index" not in out:
            out["_editor_row_index"] = idx
        normalized.append(out)
    return normalized


def build_label_colors(labels: list[str]) -> dict[str, str]:
    labels = [label for label in labels if label]
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def load_initial_banlist(path: str | Path) -> dict[str, list[str]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Banlist input must be a JSON object: {label: [terms...]}")
    out: dict[str, list[str]] = {}
    for label, terms in payload.items():
        if not isinstance(label, str) or not isinstance(terms, list):
            continue
        cleaned = []
        seen = set()
        for term in terms:
            if not isinstance(term, str):
                continue
            t = term.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            cleaned.append(t)
        out[label] = cleaned
    return out


def build_html(
    title: str,
    records: list[dict],
    labels: list[str],
    label_colors: dict[str, str],
    initial_banlist: dict[str, list[str]],
    banlist_output_name: str,
    rules_output_name: str,
) -> str:
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f7f7f8;
      --text: #1f2937;
      --muted: #6b7280;
      --card: #ffffff;
      --line: #d1d5db;
      --accent: #111827;
      --danger: #9f1239;
      --ok: #065f46;
      --soft-danger: #fff1f2;
      --soft-info: #eff6ff;
    }
    body {
      margin: 18px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    h3 { margin: 12px 0 8px 0; }
    .muted { color: var(--muted); margin: 0 0 14px 0; }
    .topbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    button, select, input {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }
    button:hover { border-color: #9ca3af; cursor: pointer; }
    button.danger { border-color: #fecdd3; background: #fff1f2; color: var(--danger); }
    button.info { border-color: #bfdbfe; background: #eff6ff; color: #1d4ed8; }
    .layout { display: grid; grid-template-columns: 1fr 390px; gap: 12px; }
    .card { border: 1px solid var(--line); border-radius: 8px; background: var(--card); padding: 12px; }
    .record-title { margin: 0 0 8px 0; font-size: 14px; font-weight: 600; }
    .record-meta { margin: 0 0 8px 0; color: var(--muted); font-size: 12px; }
    .record-text { line-height: 1.75; white-space: pre-wrap; font-size: 15px; }
    .entity {
      color: #fff; border-radius: 4px; padding: 0 4px; margin: 0 1px;
      display: inline-block; cursor: pointer; border: 1px solid transparent; user-select: none;
    }
    .entity::after { content: attr(data-label); margin-left: 6px; font-size: 10px; opacity: 0.95; }
    .entity.selected { border-color: #111827; box-shadow: 0 0 0 2px #11182733; }
    .legend-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
    .swatch { width: 28px; height: 12px; border-radius: 3px; border: 1px solid #00000033; display: inline-block; }
    .selected-box {
      font-size: 13px; line-height: 1.6; border: 1px dashed var(--line); border-radius: 6px;
      padding: 8px; margin-bottom: 8px; min-height: 64px; background: #fafafa;
    }
    .global-box { background: var(--soft-info); border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px; margin-bottom: 10px; }
    .global-box.dangerish { background: var(--soft-danger); border-color: #fecdd3; }
    .help { margin-top: 10px; color: var(--muted); font-size: 12px; line-height: 1.5; }
    .status { margin: 8px 0 0 0; font-size: 12px; color: #374151; min-height: 18px; }
    .status.error { color: var(--danger); }
    .status.ok { color: var(--ok); }
    .small { font-size: 12px; color: var(--muted); }
    code { background: #f3f4f6; border-radius: 4px; padding: 1px 4px; }
    @media (max-width: 1040px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>__TITLE__</h1>
  <p class="muted">Click an entity to remove it, change its label, or apply a global correction. Export the corrected dataset when done.</p>

  <div class="topbar">
    <button id="prevBtn" type="button">Previous</button>
    <button id="nextBtn" type="button">Next</button>
    <span id="counter" class="muted"></span>
    <label for="gotoInput" class="muted" style="margin:0;">Go to:</label>
    <input id="gotoInput" type="number" min="1" step="1" style="width:88px;" />
    <button id="gotoBtn" type="button">Go</button>
    <button id="exportBtn" type="button">Export corrected JSON</button>
  </div>

  <div class="layout">
    <section class="card">
      <h2 id="recordTitle" class="record-title"></h2>
      <p id="recordMeta" class="record-meta"></p>
      <div id="recordText" class="record-text"></div>
    </section>

    <aside class="card">
      <h3 style="margin-top:0;">Selected Entity / Selection</h3>
      <div id="selectedBox" class="selected-box">No entity selected.</div>
      <div class="topbar" style="margin-bottom:8px;">
        <select id="labelSelect"></select>
        <button id="addBtn" type="button">Add selected text</button>
        <button id="changeBtn" type="button">Change label</button>
        <button id="removeBtn" type="button">Remove</button>
      </div>

      <h3>Global corrections</h3>
      <div class="global-box dangerish">
        <div id="globalPreview" class="small">Select an entity to preview global actions.</div>
        <div class="topbar" style="margin-top:8px; margin-bottom:0;">
          <button id="removeEverywhereBtn" type="button" class="danger">Remove same text+label everywhere</button>
          <button id="changeEverywhereBtn" type="button" class="info">Change same text+label everywhere</button>
        </div>
      </div>
      <div class="global-box">
        <div class="small">Global rules are reusable. Export them and pass the remove rules back through <code>--banlist-input</code>.</div>
        <div class="topbar" style="margin-top:8px; margin-bottom:0;">
          <button id="exportBanlistBtn" type="button">Export remove rules</button>
          <button id="exportRulesBtn" type="button">Export all global rules</button>
        </div>
      </div>

      <div id="statusBox" class="status"></div>

      <h3>Legend</h3>
      <div id="legend"></div>

      <h3>Remove rules</h3>
      <div id="banlistBox" class="selected-box">No remove rules.</div>

      <h3>Change-label rules</h3>
      <div id="changeRulesBox" class="selected-box">No change rules.</div>

      <p class="help">
        Keyboard shortcuts: <br>
        <code>[</code> previous record, <code>]</code> next record,<br>
        <code>Delete</code> remove selected entity, <code>N</code> add selected text.<br><br>
        Recommended use: for generic false positives such as <code>polícia</code> labeled as <code>Location</code>, select the entity and use <i>Remove same text+label everywhere</i>.
      </p>
    </aside>
  </div>

  <script>
  const initialRecords = __RECORDS_JSON__;
  const initialLabels = __LABELS_JSON__;
  const labelColors = __LABEL_COLORS_JSON__;
  const initialBanlist = __INITIAL_BANLIST_JSON__;
  const banlistDownloadName = __BANLIST_DOWNLOAD_NAME__;
  const rulesDownloadName = __RULES_DOWNLOAD_NAME__;

  const state = {
    records: initialRecords,
    labels: initialLabels,
    current: 0,
    selectedSpanIndex: null,
    pendingSelection: null,
    banlist: {},
    changeRules: [],
    actionLog: [],
  };

  const $recordTitle = document.getElementById("recordTitle");
  const $recordMeta = document.getElementById("recordMeta");
  const $recordText = document.getElementById("recordText");
  const $counter = document.getElementById("counter");
  const $selectedBox = document.getElementById("selectedBox");
  const $labelSelect = document.getElementById("labelSelect");
  const $legend = document.getElementById("legend");
  const $statusBox = document.getElementById("statusBox");
  const $gotoInput = document.getElementById("gotoInput");
  const $banlistBox = document.getElementById("banlistBox");
  const $changeRulesBox = document.getElementById("changeRulesBox");
  const $globalPreview = document.getElementById("globalPreview");

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function normalizeTerm(value) {
    return String(value || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .trim()
      .replace(/\s+/g, " ");
  }

  function normalizeSpans(record) {
    const text = String(record.text || "");
    const spans = Array.isArray(record.spans) ? record.spans : [];
    const cleaned = spans
      .filter((s) => Number.isInteger(s.start) && Number.isInteger(s.end))
      .filter((s) => typeof s.label === "string" && s.label.trim().length > 0)
      .filter((s) => s.start >= 0 && s.end > s.start && s.end <= text.length)
      .map((s) => ({ ...s, start: s.start, end: s.end, label: s.label.trim() }))
      .sort((a, b) => (a.start - b.start) || (a.end - b.end) || a.label.localeCompare(b.label));
    record.spans = cleaned;
  }

  function setStatus(message, kind = "") {
    $statusBox.textContent = message || "";
    $statusBox.className = "status";
    if (kind) $statusBox.classList.add(kind);
  }

  function ensureLabelColor(label) {
    if (labelColors[label]) return;
    const keys = Object.keys(labelColors);
    const palette = ["#0B6E4F", "#1D4E89", "#8B1E3F", "#8C510A", "#5C2E91", "#006D77", "#264653", "#7A3E00"];
    labelColors[label] = palette[keys.length % palette.length];
  }

  function getCurrentRecord() { return state.records[state.current]; }

  function getMentionFromSpan(record, span) {
    if (!record || !span) return "";
    const text = String(record.text || "");
    if (!Number.isInteger(span.start) || !Number.isInteger(span.end)) return "";
    if (span.start < 0 || span.end <= span.start || span.end > text.length) return "";
    return text.slice(span.start, span.end);
  }

  function getCurrentSelectedSpan() {
    const record = getCurrentRecord();
    const idx = state.selectedSpanIndex;
    if (idx === null || !record || !Array.isArray(record.spans) || !record.spans[idx]) return null;
    return record.spans[idx];
  }

  function countMatchingSpans(label, termNorm) {
    let spans = 0;
    const records = new Set();
    for (let recordIdx = 0; recordIdx < state.records.length; recordIdx += 1) {
      const record = state.records[recordIdx];
      if (!Array.isArray(record.spans)) continue;
      for (const span of record.spans) {
        if (normalizeTerm(span.label) !== normalizeTerm(label)) continue;
        if (normalizeTerm(getMentionFromSpan(record, span)) !== termNorm) continue;
        spans += 1;
        records.add(recordIdx);
      }
    }
    return { spans, records: records.size };
  }

  function renderLegend() {
    const labels = [...state.labels].sort();
    $legend.innerHTML = labels.map((label) => {
      const color = labelColors[label] || "#444";
      return `<div class="legend-row"><span class="swatch" style="background:${color}"></span><code>${escapeHtml(label)}</code></div>`;
    }).join("");
  }

  function renderLabelSelect() {
    $labelSelect.innerHTML = state.labels.map((label) => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`).join("");
  }

  function renderBanlist() {
    const labels = Object.keys(state.banlist || {}).sort();
    if (labels.length === 0) { $banlistBox.textContent = "No remove rules."; return; }
    $banlistBox.innerHTML = labels.map((label) => {
      const terms = Array.from(state.banlist[label]).sort();
      return `<div style="margin-bottom:6px;"><b>${escapeHtml(label)}</b>: ${escapeHtml(terms.join(", "))}</div>`;
    }).join("");
  }

  function renderChangeRules() {
    if (!state.changeRules.length) { $changeRulesBox.textContent = "No change rules."; return; }
    $changeRulesBox.innerHTML = state.changeRules.map((rule) => {
      return `<div style="margin-bottom:6px;"><code>${escapeHtml(rule.term)}</code>: ${escapeHtml(rule.from_label)} → ${escapeHtml(rule.to_label)} <span class="small">(${rule.affected_spans} spans)</span></div>`;
    }).join("");
  }

  function renderGlobalPreview() {
    const record = getCurrentRecord();
    const span = getCurrentSelectedSpan();
    if (!record || !span) {
      $globalPreview.textContent = "Select an entity to preview global actions.";
      return;
    }
    const mention = getMentionFromSpan(record, span);
    const termNorm = normalizeTerm(mention);
    const count = countMatchingSpans(span.label, termNorm);
    $globalPreview.innerHTML =
      `Selected <code>${escapeHtml(mention)}</code> as <code>${escapeHtml(span.label)}</code>. ` +
      `A global action will affect <b>${count.spans}</b> span(s) in <b>${count.records}</b> record(s).`;
  }

  function renderSelected() {
    const record = getCurrentRecord();
    const idx = state.selectedSpanIndex;
    const pending = state.pendingSelection;
    if (idx === null || !record || !record.spans || !record.spans[idx]) {
      if (pending) {
        const mention = record.text.slice(pending.start, pending.end);
        $selectedBox.innerHTML =
          `<div><b>Selection:</b> ${escapeHtml(mention)}</div>` +
          `<div><b>Offsets:</b> ${pending.start}-${pending.end}</div>` +
          `<div><b>Action:</b> choose label and click <i>Add selected text</i>.</div>`;
      } else {
        $selectedBox.textContent = "No entity selected.";
      }
      renderGlobalPreview();
      return;
    }
    const span = record.spans[idx];
    const mention = getMentionFromSpan(record, span);
    const extra = span.seed_origin ? `<div><b>Origin:</b> ${escapeHtml(span.seed_origin)}</div>` : "";
    $selectedBox.innerHTML =
      `<div><b>Text:</b> ${escapeHtml(mention)}</div>` +
      `<div><b>Label:</b> ${escapeHtml(span.label)}</div>` +
      `<div><b>Offsets:</b> ${span.start}-${span.end}</div>` + extra;
    $labelSelect.value = span.label;
    renderGlobalPreview();
  }

  function renderText() {
    const record = getCurrentRecord();
    if (!record) { $recordText.innerHTML = ""; return; }
    const text = String(record.text || "");
    const spans = Array.isArray(record.spans) ? record.spans : [];
    let cursor = 0;
    let html = "";
    for (let i = 0; i < spans.length; i += 1) {
      const span = spans[i];
      if (span.start < cursor) continue;
      if (cursor < span.start) html += escapeHtml(text.slice(cursor, span.start));
      const mention = escapeHtml(text.slice(span.start, span.end));
      const label = escapeHtml(span.label);
      const color = labelColors[span.label] || "#444";
      const selectedClass = i === state.selectedSpanIndex ? " selected" : "";
      const title = span.seed_origin ? `${span.label} | ${span.seed_origin}` : span.label;
      html += `<span class="entity${selectedClass}" title="${escapeHtml(title)}" data-label="${label}" data-span-index="${i}" style="background:${color};">${mention}</span>`;
      cursor = span.end;
    }
    if (cursor < text.length) html += escapeHtml(text.slice(cursor));
    $recordText.innerHTML = html;

    Array.from($recordText.querySelectorAll(".entity")).forEach((el) => {
      el.addEventListener("click", () => {
        const idx = Number(el.getAttribute("data-span-index"));
        state.selectedSpanIndex = Number.isInteger(idx) ? idx : null;
        state.pendingSelection = null;
        setStatus("");
        render();
      });
    });
  }

  function getSelectionOffsets() {
    const record = getCurrentRecord();
    if (!record) return null;
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    if (range.collapsed) return null;
    if (!$recordText.contains(range.startContainer) || !$recordText.contains(range.endContainer)) return null;
    const pre = range.cloneRange();
    pre.selectNodeContents($recordText);
    pre.setEnd(range.startContainer, range.startOffset);
    const start = pre.toString().length;
    const selectedText = range.toString();
    const end = start + selectedText.length;
    if (end <= start || start < 0 || end > record.text.length) return null;
    return { start, end };
  }

  function hasOverlap(start, end, spans) {
    return spans.some((s) => Math.max(s.start, start) < Math.min(s.end, end));
  }

  function updatePendingSelection() {
    const offsets = getSelectionOffsets();
    if (!offsets) return;
    state.pendingSelection = offsets;
    state.selectedSpanIndex = null;
    setStatus("");
    renderSelected();
  }

  function render() {
    const total = state.records.length;
    if (total === 0) {
      $recordTitle.textContent = "No records"; $recordMeta.textContent = ""; $recordText.textContent = ""; $counter.textContent = "0 / 0";
      state.selectedSpanIndex = null; renderSelected(); return;
    }
    const record = getCurrentRecord();
    normalizeSpans(record);
    if (state.selectedSpanIndex !== null && state.selectedSpanIndex >= record.spans.length) state.selectedSpanIndex = null;
    $recordTitle.textContent = `Record #${state.current + 1}`;
    const source = record.source_id || record.sample_id || record._editor_row_index;
    $recordMeta.textContent = `entities=${record.spans.length}` + (source !== undefined && source !== "" ? ` | id=${source}` : "");
    $counter.textContent = `${state.current + 1} / ${total}`;
    $gotoInput.value = String(state.current + 1);
    renderText(); renderSelected(); renderBanlist(); renderChangeRules();
  }

  function changeLabel() {
    const span = getCurrentSelectedSpan();
    if (!span) return;
    const newLabel = $labelSelect.value.trim();
    if (!newLabel) return;
    span.label = newLabel;
    if (!state.labels.includes(newLabel)) { state.labels.push(newLabel); ensureLabelColor(newLabel); renderLegend(); renderLabelSelect(); }
    setStatus("Entity label updated.", "ok"); render();
  }

  function removeSelected() {
    const record = getCurrentRecord();
    const idx = state.selectedSpanIndex;
    if (idx === null || !record || !record.spans[idx]) return;
    record.spans.splice(idx, 1);
    state.selectedSpanIndex = null;
    setStatus("Entity removed.", "ok"); render();
  }

  function addEntityFromSelection() {
    const record = getCurrentRecord();
    if (!record) return;
    let selection = state.pendingSelection;
    if (!selection) { selection = getSelectionOffsets(); if (selection) state.pendingSelection = selection; }
    if (!selection) { setStatus("Select a text span first, then click Add selected text.", "error"); renderSelected(); return; }
    const label = $labelSelect.value.trim();
    if (!label) { setStatus("Choose a label before adding an entity.", "error"); return; }
    if (hasOverlap(selection.start, selection.end, record.spans)) { setStatus("Selected text overlaps an existing entity. Remove or adjust first.", "error"); return; }
    record.spans.push({ start: selection.start, end: selection.end, label, seed_origin: "manual_editor_added" });
    normalizeSpans(record);
    state.pendingSelection = null; state.selectedSpanIndex = null;
    if (!state.labels.includes(label)) { state.labels.push(label); ensureLabelColor(label); renderLegend(); renderLabelSelect(); }
    setStatus("Entity added.", "ok"); render(); window.getSelection()?.removeAllRanges();
  }

  function applyRemoveRule(label, normalizedTerm) {
    if (!state.banlist[label]) state.banlist[label] = new Set();
    state.banlist[label].add(normalizedTerm);
    let removed = 0;
    for (const record of state.records) {
      if (!Array.isArray(record.spans)) continue;
      const kept = [];
      for (const span of record.spans) {
        const sameLabel = normalizeTerm(span.label) === normalizeTerm(label);
        const sameTerm = normalizeTerm(getMentionFromSpan(record, span)) === normalizedTerm;
        if (sameLabel && sameTerm) { removed += 1; continue; }
        kept.push(span);
      }
      record.spans = kept;
    }
    return removed;
  }

  function removeSameEverywhere() {
    const record = getCurrentRecord();
    const span = getCurrentSelectedSpan();
    if (!record || !span) { setStatus("Select an entity before applying a global removal.", "error"); return; }
    const label = span.label;
    const mention = getMentionFromSpan(record, span);
    const termNorm = normalizeTerm(mention);
    const count = countMatchingSpans(label, termNorm);
    const ok = window.confirm(
      `Remove same text+label everywhere?\n\nText: '${mention}'\nNormalized: '${termNorm}'\nLabel: '${label}'\nAffected: ${count.spans} spans in ${count.records} records`
    );
    if (!ok) return;
    const removed = applyRemoveRule(label, termNorm);
    state.actionLog.push({ action: "remove_same_text_label_everywhere", label, term: termNorm, display_text: mention, affected_spans: removed });
    state.selectedSpanIndex = null; state.pendingSelection = null;
    setStatus(`Global remove applied (${label}::${termNorm}). Removed ${removed} spans.`, "ok"); render();
  }

  function changeSameEverywhere() {
    const record = getCurrentRecord();
    const span = getCurrentSelectedSpan();
    if (!record || !span) { setStatus("Select an entity before applying a global label change.", "error"); return; }
    const oldLabel = span.label;
    const newLabel = $labelSelect.value.trim();
    if (!newLabel) { setStatus("Choose the target label first.", "error"); return; }
    if (normalizeTerm(oldLabel) === normalizeTerm(newLabel)) { setStatus("Target label is the same as current label.", "error"); return; }
    const mention = getMentionFromSpan(record, span);
    const termNorm = normalizeTerm(mention);
    const count = countMatchingSpans(oldLabel, termNorm);
    const ok = window.confirm(
      `Change same text+label everywhere?\n\nText: '${mention}'\nNormalized: '${termNorm}'\nFrom: '${oldLabel}'\nTo: '${newLabel}'\nAffected: ${count.spans} spans in ${count.records} records`
    );
    if (!ok) return;

    let changed = 0;
    for (const rec of state.records) {
      if (!Array.isArray(rec.spans)) continue;
      for (const sp of rec.spans) {
        const sameLabel = normalizeTerm(sp.label) === normalizeTerm(oldLabel);
        const sameTerm = normalizeTerm(getMentionFromSpan(rec, sp)) === termNorm;
        if (sameLabel && sameTerm) {
          sp.label = newLabel;
          sp.seed_origin = sp.seed_origin || "manual_editor_global_label_change";
          changed += 1;
        }
      }
    }
    if (!state.labels.includes(newLabel)) { state.labels.push(newLabel); ensureLabelColor(newLabel); renderLegend(); renderLabelSelect(); }
    const rule = { action: "change_same_text_label_everywhere", term: termNorm, display_text: mention, from_label: oldLabel, to_label: newLabel, affected_spans: changed };
    state.changeRules.push(rule);
    state.actionLog.push(rule);
    state.selectedSpanIndex = null; state.pendingSelection = null;
    setStatus(`Global label change applied. Changed ${changed} spans.`, "ok"); render();
  }

  function previousRecord() { if (state.current <= 0) return; state.current -= 1; state.selectedSpanIndex = null; state.pendingSelection = null; setStatus(""); render(); }
  function nextRecord() { if (state.current >= state.records.length - 1) return; state.current += 1; state.selectedSpanIndex = null; state.pendingSelection = null; setStatus(""); render(); }
  function goToRecord() {
    const total = state.records.length; if (total === 0) return;
    const raw = Number($gotoInput.value); if (!Number.isFinite(raw)) { setStatus("Enter a valid record number.", "error"); return; }
    const wanted = Math.floor(raw); if (wanted < 1 || wanted > total) { setStatus(`Record number must be between 1 and ${total}.`, "error"); return; }
    state.current = wanted - 1; state.selectedSpanIndex = null; state.pendingSelection = null; setStatus(""); render();
  }

  function downloadJson(filename, data) {
    const payload = JSON.stringify(data, null, 2);
    const blob = new Blob([payload], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }

  function exportJson() { downloadJson("annotations_corrected.json", state.records); }
  function exportBanlist() {
    const obj = {};
    const labels = Object.keys(state.banlist || {}).sort();
    for (const label of labels) obj[label] = Array.from(state.banlist[label]).sort();
    downloadJson(banlistDownloadName, obj);
  }
  function exportRules() {
    const removeRules = {};
    const labels = Object.keys(state.banlist || {}).sort();
    for (const label of labels) removeRules[label] = Array.from(state.banlist[label]).sort();
    downloadJson(rulesDownloadName, { remove_rules: removeRules, change_label_rules: state.changeRules, action_log: state.actionLog });
  }

  document.getElementById("changeBtn").addEventListener("click", changeLabel);
  document.getElementById("addBtn").addEventListener("click", addEntityFromSelection);
  document.getElementById("removeBtn").addEventListener("click", removeSelected);
  document.getElementById("removeEverywhereBtn").addEventListener("click", removeSameEverywhere);
  document.getElementById("changeEverywhereBtn").addEventListener("click", changeSameEverywhere);
  document.getElementById("prevBtn").addEventListener("click", previousRecord);
  document.getElementById("nextBtn").addEventListener("click", nextRecord);
  document.getElementById("gotoBtn").addEventListener("click", goToRecord);
  document.getElementById("exportBtn").addEventListener("click", exportJson);
  document.getElementById("exportBanlistBtn").addEventListener("click", exportBanlist);
  document.getElementById("exportRulesBtn").addEventListener("click", exportRules);
  $gotoInput.addEventListener("keydown", (evt) => { if (evt.key === "Enter") goToRecord(); });

  document.addEventListener("keydown", (evt) => {
    if (evt.key === "[") previousRecord();
    if (evt.key === "]") nextRecord();
    if (evt.key === "Delete" || evt.key === "Backspace") removeSelected();
    if (evt.key === "n" || evt.key === "N") addEntityFromSelection();
  });
  $recordText.addEventListener("mouseup", updatePendingSelection);
  $recordText.addEventListener("keyup", updatePendingSelection);

  renderLegend(); renderLabelSelect();
  state.banlist = {};
  Object.keys(initialBanlist || {}).forEach((label) => {
    const terms = Array.isArray(initialBanlist[label]) ? initialBanlist[label] : [];
    state.banlist[label] = new Set(terms.filter((x) => typeof x === "string" && x.trim().length > 0));
  });
  // Apply loaded remove rules at startup.
  Object.keys(state.banlist).forEach((label) => {
    const terms = Array.from(state.banlist[label]);
    terms.forEach((termNorm) => applyRemoveRule(label, normalizeTerm(termNorm)));
  });
  render();
  </script>
</body>
</html>"""
    return (
        template.replace("__TITLE__", escape(title))
        .replace("__RECORDS_JSON__", json.dumps(records, ensure_ascii=False))
        .replace("__LABELS_JSON__", json.dumps(labels, ensure_ascii=False))
        .replace("__LABEL_COLORS_JSON__", json.dumps(label_colors, ensure_ascii=False))
        .replace("__INITIAL_BANLIST_JSON__", json.dumps(initial_banlist, ensure_ascii=False))
        .replace("__BANLIST_DOWNLOAD_NAME__", json.dumps(Path(banlist_output_name).name, ensure_ascii=False))
        .replace("__RULES_DOWNLOAD_NAME__", json.dumps(Path(rules_output_name).name, ensure_ascii=False))
    )


def main() -> None:
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if not labels:
        labels = list(DEFAULT_LABELS)

    rows = load_corpus(args.input)
    rows = normalize_records(rows)
    if args.max_reports > 0:
        rows = rows[: args.max_reports]

    label_colors = build_label_colors(labels)
    initial_banlist = load_initial_banlist(args.banlist_input)
    html_content = build_html(
        args.title,
        rows,
        labels,
        label_colors,
        initial_banlist,
        args.banlist_output,
        args.rules_output,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"[ok] Interactive editor saved to: {args.output}")


if __name__ == "__main__":
    main()
