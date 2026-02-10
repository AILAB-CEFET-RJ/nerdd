#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def parse_args():
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
        help="Optional JSON file with initial banlist: {label: [normalized_terms...]}",
    )
    parser.add_argument(
        "--banlist-output",
        default="annotation_banlist.json",
        help="Default filename used when exporting banlist from the HTML editor.",
    )
    return parser.parse_args()


def _parse_jsonl(text):
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


def load_corpus(path):
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


def normalize_span(span):
    start = span.get("start")
    end = span.get("end")
    label = span.get("label")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    if end <= start:
        return None
    return {"start": start, "end": end, "label": label.strip()}


def get_text(record):
    for key in ("text", "relato", "texto", "description", "descricao"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_spans(record):
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


def sanitize_spans(text, spans):
    valid = []
    for raw in spans:
        span = normalize_span(raw)
        if not span:
            continue
        if span["start"] < 0 or span["end"] > len(text):
            continue
        valid.append(span)
    return sorted(valid, key=lambda s: (s["start"], s["end"]))


def normalize_records(rows):
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out = dict(row)
        text = get_text(out)
        out["text"] = text
        out["spans"] = sanitize_spans(text, get_spans(out))
        normalized.append(out)
    return normalized


def build_label_colors(labels):
    labels = [l for l in labels if l]
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def load_initial_banlist(path):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Banlist input must be a JSON object: {label: [terms...]}")
    out = {}
    for label, terms in payload.items():
        if not isinstance(label, str):
            continue
        if not isinstance(terms, list):
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


def build_html(title, records, labels, label_colors, initial_banlist, banlist_output_name):
    template = """<!doctype html>
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
    }
    body {
      margin: 18px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    .muted { color: var(--muted); margin: 0 0 14px 0; }
    .topbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    button, select {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }
    button:hover { border-color: #9ca3af; cursor: pointer; }
    .layout {
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 12px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--card);
      padding: 12px;
    }
    .record-title {
      margin: 0 0 8px 0;
      font-size: 14px;
      font-weight: 600;
    }
    .record-meta {
      margin: 0 0 8px 0;
      color: var(--muted);
      font-size: 12px;
    }
    .record-text {
      line-height: 1.75;
      white-space: pre-wrap;
      font-size: 15px;
    }
    .entity {
      color: #fff;
      border-radius: 4px;
      padding: 0 4px;
      margin: 0 1px;
      display: inline-block;
      cursor: pointer;
      border: 1px solid transparent;
      user-select: none;
    }
    .entity .tag {
      margin-left: 6px;
      font-size: 10px;
      opacity: 0.95;
    }
    .entity::after {
      content: attr(data-label);
      margin-left: 6px;
      font-size: 10px;
      opacity: 0.95;
    }
    .entity.selected {
      border-color: #111827;
      box-shadow: 0 0 0 2px #11182733;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
      font-size: 13px;
    }
    .swatch {
      width: 28px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid #00000033;
      display: inline-block;
    }
    .selected-box {
      font-size: 13px;
      line-height: 1.6;
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 8px;
      margin-bottom: 8px;
      min-height: 64px;
      background: #fafafa;
    }
    .help {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .status {
      margin: 8px 0 0 0;
      font-size: 12px;
      color: #374151;
      min-height: 18px;
    }
    .status.error { color: #9f1239; }
    .status.ok { color: #065f46; }
    @media (max-width: 1040px) {
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <h1>__TITLE__</h1>
  <p class="muted">Click an entity to remove it or change its label. Export the corrected dataset when done.</p>

  <div class="topbar">
    <button id="prevBtn" type="button">Previous</button>
    <button id="nextBtn" type="button">Next</button>
    <span id="counter" class="muted"></span>
    <label for="gotoInput" class="muted" style="margin:0;">Go to:</label>
    <input id="gotoInput" type="number" min="1" step="1" style="width:88px; padding:6px 8px; border:1px solid #d1d5db; border-radius:6px;" />
    <button id="gotoBtn" type="button">Go</button>
    <button id="exportBtn" type="button">Export JSON</button>
  </div>

  <div class="layout">
    <section class="card">
      <h2 id="recordTitle" class="record-title"></h2>
      <p id="recordMeta" class="record-meta"></p>
      <div id="recordText" class="record-text"></div>
    </section>

    <aside class="card">
      <h3 style="margin-top:0;">Selected Entity</h3>
      <div id="selectedBox" class="selected-box">No entity selected.</div>
      <div class="topbar" style="margin-bottom:8px;">
        <select id="labelSelect"></select>
        <button id="addBtn" type="button">Add entity</button>
        <button id="changeBtn" type="button">Change label</button>
        <button id="removeBtn" type="button">Remove</button>
        <button id="banBtn" type="button">Ban term</button>
      </div>
      <div id="statusBox" class="status"></div>

      <h3 style="margin:12px 0 8px 0;">Legend</h3>
      <div id="legend"></div>

      <h3 style="margin:12px 0 8px 0;">Banlist</h3>
      <div class="topbar" style="margin-bottom:8px;">
        <button id="exportBanlistBtn" type="button">Export banlist</button>
      </div>
      <div id="banlistBox" class="selected-box">No ban rules.</div>

      <p class="help">
        Keyboard shortcuts: <br>
        <code>[</code> previous record, <code>]</code> next record,<br>
        <code>Delete</code> remove selected entity, <code>N</code> add selected text,<br>
        <code>B</code> ban selected term globally.
      </p>
    </aside>
  </div>

  <script>
  const initialRecords = __RECORDS_JSON__;
  const initialLabels = __LABELS_JSON__;
  const labelColors = __LABEL_COLORS_JSON__;
  const initialBanlist = __INITIAL_BANLIST_JSON__;
  const banlistDownloadName = __BANLIST_DOWNLOAD_NAME__;

  const state = {
    records: initialRecords,
    labels: initialLabels,
    current: 0,
    selectedSpanIndex: null,
    pendingSelection: null,
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
      .map((s) => ({ start: s.start, end: s.end, label: s.label.trim() }))
      .sort((a, b) => (a.start - b.start) || (a.end - b.end));
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

  function getCurrentRecord() {
    return state.records[state.current];
  }

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

  function renderLegend() {
    const labels = [...state.labels].sort();
    $legend.innerHTML = labels
      .map((label) => {
        const color = labelColors[label] || "#444";
        return `<div class="legend-row"><span class="swatch" style="background:${color}"></span><code>${escapeHtml(label)}</code></div>`;
      })
      .join("");
  }

  function renderLabelSelect() {
    $labelSelect.innerHTML = state.labels
      .map((label) => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`)
      .join("");
  }

  function renderBanlist() {
    const labels = Object.keys(state.banlist || {}).sort();
    if (labels.length === 0) {
      $banlistBox.textContent = "No ban rules.";
      return;
    }
    const html = labels.map((label) => {
      const terms = Array.from(state.banlist[label]).sort();
      return `<div style="margin-bottom:6px;"><b>${escapeHtml(label)}</b>: ${escapeHtml(terms.join(", "))}</div>`;
    }).join("");
    $banlistBox.innerHTML = html;
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
          `<div><b>Action:</b> choose label and click <i>Add entity</i>.</div>`;
      } else {
        $selectedBox.textContent = "No entity selected.";
      }
      return;
    }
    const span = record.spans[idx];
    const mention = getMentionFromSpan(record, span);
    $selectedBox.innerHTML =
      `<div><b>Text:</b> ${escapeHtml(mention)}</div>` +
      `<div><b>Label:</b> ${escapeHtml(span.label)}</div>` +
      `<div><b>Offsets:</b> ${span.start}-${span.end}</div>`;
    $labelSelect.value = span.label;
  }

  function renderText() {
    const record = getCurrentRecord();
    if (!record) {
      $recordText.innerHTML = "";
      return;
    }
    const text = String(record.text || "");
    const spans = Array.isArray(record.spans) ? record.spans : [];
    let cursor = 0;
    let html = "";
    for (let i = 0; i < spans.length; i += 1) {
      const span = spans[i];
      if (span.start < cursor) continue;
      if (cursor < span.start) {
        html += escapeHtml(text.slice(cursor, span.start));
      }
      const mention = escapeHtml(text.slice(span.start, span.end));
      const label = escapeHtml(span.label);
      const color = labelColors[span.label] || "#444";
      const selectedClass = i === state.selectedSpanIndex ? " selected" : "";
      html += `<span class="entity${selectedClass}" data-label="${label}" data-span-index="${i}" style="background:${color};">${mention}</span>`;
      cursor = span.end;
    }
    if (cursor < text.length) {
      html += escapeHtml(text.slice(cursor));
    }
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
    if (end <= start) return null;
    if (start < 0 || end > record.text.length) return null;
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
      $recordTitle.textContent = "No records";
      $recordMeta.textContent = "";
      $recordText.textContent = "";
      $counter.textContent = "0 / 0";
      state.selectedSpanIndex = null;
      renderSelected();
      return;
    }
    const record = getCurrentRecord();
    normalizeSpans(record);
    if (state.selectedSpanIndex !== null && state.selectedSpanIndex >= record.spans.length) {
      state.selectedSpanIndex = null;
    }
    $recordTitle.textContent = `Record #${state.current + 1}`;
    $recordMeta.textContent = `entities=${record.spans.length}`;
    $counter.textContent = `${state.current + 1} / ${total}`;
    $gotoInput.value = String(state.current + 1);
    renderText();
    renderSelected();
    renderBanlist();
  }

  function changeLabel() {
    const record = getCurrentRecord();
    const span = getCurrentSelectedSpan();
    if (!record || !span) return;
    const newLabel = $labelSelect.value.trim();
    if (!newLabel) return;
    span.label = newLabel;
    if (!state.labels.includes(newLabel)) {
      state.labels.push(newLabel);
      ensureLabelColor(newLabel);
      renderLegend();
      renderLabelSelect();
    }
    setStatus("Entity label updated.", "ok");
    render();
  }

  function removeSelected() {
    const record = getCurrentRecord();
    const idx = state.selectedSpanIndex;
    if (idx === null || !record || !record.spans[idx]) return;
    record.spans.splice(idx, 1);
    state.selectedSpanIndex = null;
    setStatus("Entity removed.", "ok");
    render();
  }

  function addEntityFromSelection() {
    const record = getCurrentRecord();
    if (!record) return;
    let selection = state.pendingSelection;
    if (!selection) {
      selection = getSelectionOffsets();
      if (selection) state.pendingSelection = selection;
    }
    if (!selection) {
      setStatus("Select a text span first, then click Add entity.", "error");
      renderSelected();
      return;
    }
    const label = $labelSelect.value.trim();
    if (!label) {
      setStatus("Choose a label before adding an entity.", "error");
      return;
    }
    if (hasOverlap(selection.start, selection.end, record.spans)) {
      setStatus("Selected text overlaps an existing entity. Remove or adjust first.", "error");
      return;
    }
    record.spans.push({ start: selection.start, end: selection.end, label });
    normalizeSpans(record);
    state.pendingSelection = null;
    state.selectedSpanIndex = null;
    if (!state.labels.includes(label)) {
      state.labels.push(label);
      ensureLabelColor(label);
      renderLegend();
      renderLabelSelect();
    }
    setStatus("Entity added.", "ok");
    render();
    window.getSelection()?.removeAllRanges();
  }

  function applyBanRule(label, normalizedTerm) {
    if (!state.banlist[label]) state.banlist[label] = new Set();
    state.banlist[label].add(normalizedTerm);

    let removed = 0;
    for (const record of state.records) {
      if (!Array.isArray(record.spans)) continue;
      const kept = [];
      for (const span of record.spans) {
        const sameLabel = normalizeTerm(span.label) === normalizeTerm(label);
        if (!sameLabel) {
          kept.push(span);
          continue;
        }
        const mention = getMentionFromSpan(record, span);
        const mentionNorm = normalizeTerm(mention);
        if (mentionNorm && mentionNorm === normalizedTerm) {
          removed += 1;
          continue;
        }
        kept.push(span);
      }
      record.spans = kept;
    }
    return removed;
  }

  function banSelectedTerm() {
    const record = getCurrentRecord();
    const span = getCurrentSelectedSpan();
    if (!record || !span) {
      setStatus("Select an entity before banning a term.", "error");
      return;
    }
    const label = span.label;
    const mention = getMentionFromSpan(record, span);
    const termNorm = normalizeTerm(mention);
    if (!termNorm) {
      setStatus("Could not derive selected term.", "error");
      return;
    }

    const previewCount = state.records.reduce((acc, rec) => {
      if (!Array.isArray(rec.spans)) return acc;
      return acc + rec.spans.filter((sp) => {
        if (normalizeTerm(sp.label) !== normalizeTerm(label)) return false;
        return normalizeTerm(getMentionFromSpan(rec, sp)) === termNorm;
      }).length;
    }, 0);

    const ok = window.confirm(
      `Ban term for label '${label}'?\n\nTerm: '${mention}'\nNormalized: '${termNorm}'\nOccurrences to remove: ${previewCount}`
    );
    if (!ok) return;

    const removed = applyBanRule(label, termNorm);
    state.selectedSpanIndex = null;
    state.pendingSelection = null;
    setStatus(`Ban rule added (${label}::${termNorm}). Removed ${removed} spans.`, "ok");
    render();
  }

  function previousRecord() {
    if (state.current <= 0) return;
    state.current -= 1;
    state.selectedSpanIndex = null;
    state.pendingSelection = null;
    setStatus("");
    render();
  }

  function nextRecord() {
    if (state.current >= state.records.length - 1) return;
    state.current += 1;
    state.selectedSpanIndex = null;
    state.pendingSelection = null;
    setStatus("");
    render();
  }

  function goToRecord() {
    const total = state.records.length;
    if (total === 0) return;
    const raw = Number($gotoInput.value);
    if (!Number.isFinite(raw)) {
      setStatus("Enter a valid record number.", "error");
      return;
    }
    const wanted = Math.floor(raw);
    if (wanted < 1 || wanted > total) {
      setStatus(`Record number must be between 1 and ${total}.`, "error");
      return;
    }
    state.current = wanted - 1;
    state.selectedSpanIndex = null;
    state.pendingSelection = null;
    setStatus("");
    render();
  }

  function exportJson() {
    const payload = JSON.stringify(state.records, null, 2);
    const blob = new Blob([payload], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotations_corrected.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportBanlist() {
    const obj = {};
    const labels = Object.keys(state.banlist || {}).sort();
    for (const label of labels) {
      obj[label] = Array.from(state.banlist[label]).sort();
    }
    const payload = JSON.stringify(obj, null, 2);
    const blob = new Blob([payload], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = banlistDownloadName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  document.getElementById("changeBtn").addEventListener("click", changeLabel);
  document.getElementById("addBtn").addEventListener("click", addEntityFromSelection);
  document.getElementById("removeBtn").addEventListener("click", removeSelected);
  document.getElementById("banBtn").addEventListener("click", banSelectedTerm);
  document.getElementById("prevBtn").addEventListener("click", previousRecord);
  document.getElementById("nextBtn").addEventListener("click", nextRecord);
  document.getElementById("gotoBtn").addEventListener("click", goToRecord);
  document.getElementById("exportBtn").addEventListener("click", exportJson);
  document.getElementById("exportBanlistBtn").addEventListener("click", exportBanlist);
  $gotoInput.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter") goToRecord();
  });

  document.addEventListener("keydown", (evt) => {
    if (evt.key === "[") previousRecord();
    if (evt.key === "]") nextRecord();
    if (evt.key === "Delete" || evt.key === "Backspace") removeSelected();
    if (evt.key === "n" || evt.key === "N") addEntityFromSelection();
    if (evt.key === "b" || evt.key === "B") banSelectedTerm();
  });

  $recordText.addEventListener("mouseup", () => {
    updatePendingSelection();
  });
  $recordText.addEventListener("keyup", () => {
    updatePendingSelection();
  });

  renderLegend();
  renderLabelSelect();
  state.banlist = {};
  Object.keys(initialBanlist || {}).forEach((label) => {
    const terms = Array.isArray(initialBanlist[label]) ? initialBanlist[label] : [];
    state.banlist[label] = new Set(terms.filter((x) => typeof x === "string" && x.trim().length > 0));
  });

  // Apply loaded banlist to current dataset at startup.
  Object.keys(state.banlist).forEach((label) => {
    const terms = Array.from(state.banlist[label]);
    terms.forEach((termNorm) => {
      applyBanRule(label, normalizeTerm(termNorm));
    });
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
    )


def main():
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
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"[ok] Interactive editor saved to: {args.output}")


if __name__ == "__main__":
    main()
