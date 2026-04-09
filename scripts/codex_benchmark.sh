#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  scripts/codex_benchmark.sh <benchmark_dir> next
  scripts/codex_benchmark.sh <benchmark_dir> open-next
  scripts/codex_benchmark.sh <benchmark_dir> complete-next
  scripts/codex_benchmark.sh <benchmark_dir> show <chunk_id>
  scripts/codex_benchmark.sh <benchmark_dir> show-latest
  scripts/codex_benchmark.sh <benchmark_dir> response-path <chunk_id>
  scripts/codex_benchmark.sh <benchmark_dir> ingest <chunk_id>
  scripts/codex_benchmark.sh <benchmark_dir> ingest-latest
  scripts/codex_benchmark.sh <benchmark_dir> status
  scripts/codex_benchmark.sh <benchmark_dir> build-output

Examples:
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 next
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 open-next
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 complete-next
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 show chunk_001
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 show-latest
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 response-path chunk_001
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 ingest chunk_001
  scripts/codex_benchmark.sh artifacts/benchmarks/codex_adjudication_t06_top1000 ingest-latest
EOF
  exit 1
fi

BENCH_DIR="${1%/}"
COMMAND="$2"
STATE_JSON="$BENCH_DIR/state.json"
CHUNKS_DIR="$BENCH_DIR/chunks"
RESP_DIR="$BENCH_DIR/manual_responses"

mkdir -p "$RESP_DIR"

step_note() {
  printf '\n[%s] %s\n' "$1" "$2"
}

seed_rule_note() {
  step_note "$1" "Regra critica: para decision='accept' ou 'accept_with_edits', entities_final so pode conter spans ja presentes em review_seed_entities."
  step_note "$1" "Nao adicione baseline_only, gliner2_only, normalizacoes, correcoes ortograficas ou spans semanticamente plausiveis fora do seed set."
}

latest_exported_chunk_id() {
  python3 - "$STATE_JSON" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state = json.loads(state_path.read_text(encoding="utf-8"))
exported = [chunk for chunk in state.get("chunks", []) if chunk.get("status") == "exported"]
if not exported:
    raise SystemExit("No exported chunk found in state.json")
latest = max(exported, key=lambda item: int(item.get("chunk_index_1based", 0)))
print(latest["chunk_id"])
PY
}

has_exported_chunk() {
  python3 - "$STATE_JSON" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state = json.loads(state_path.read_text(encoding="utf-8"))
print("yes" if any(chunk.get("status") == "exported" for chunk in state.get("chunks", [])) else "no")
PY
}

has_pending_chunk() {
  python3 - "$STATE_JSON" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state = json.loads(state_path.read_text(encoding="utf-8"))
print("yes" if any(chunk.get("status") == "pending" for chunk in state.get("chunks", [])) else "no")
PY
}

print_chunk_paths() {
  local chunk_id="$1"
  printf 'chunk_id=%s\n' "$chunk_id"
  printf 'chunk_path=%s\n' "$CHUNKS_DIR/$chunk_id.jsonl"
  printf 'response_path=%s\n' "$RESP_DIR/$chunk_id.jsonl"
}

reserve_next_chunk() {
  python3 src/tools/manage_codex_adjudication_benchmark.py next \
    --state-json "$STATE_JSON" >/tmp/codex_benchmark_next.json
  python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("/tmp/codex_benchmark_next.json").read_text(encoding="utf-8"))
print(payload["chunk_id"])
PY
}

open_chunk_flow() {
  local chunk_id="$1"
  step_note "open-next" "Chunk reservado. Revise o conteúdo abaixo e me envie no chat para adjudicação."
  print_chunk_paths "$chunk_id"
  seed_rule_note "open-next"
  step_note "open-next" "Quando eu devolver a resposta, salve-a com:"
  printf "cat > %s <<'EOF'\n" "$RESP_DIR/$chunk_id.jsonl"
  step_note "open-next" "Depois rode:"
  printf "bash scripts/codex_benchmark.sh %s ingest-latest\n" "$BENCH_DIR"
  printf "ou\n"
  printf "bash scripts/codex_benchmark.sh %s complete-next\n" "$BENCH_DIR"
  echo
  cat "$CHUNKS_DIR/$chunk_id.jsonl"
}

require_chunk_id() {
  if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "chunk_id is required for command '$COMMAND'" >&2
    exit 1
  fi
}

case "$COMMAND" in
  next)
    python3 src/tools/manage_codex_adjudication_benchmark.py next \
      --state-json "$STATE_JSON"
    step_note "next" "Agora rode 'show <chunk_id>' para ver o conteúdo do chunk reservado."
    ;;
  open-next)
    chunk_id="$(reserve_next_chunk)"
    open_chunk_flow "$chunk_id"
    ;;
  complete-next)
    if [[ "$(has_exported_chunk)" == "yes" ]]; then
      chunk_id="$(latest_exported_chunk_id)"
      response_path="$RESP_DIR/$chunk_id.jsonl"
      if [[ -f "$response_path" ]]; then
        step_note "complete-next" "Resposta encontrada para $chunk_id. Ingerindo antes de abrir o próximo chunk."
        python3 src/tools/manage_codex_adjudication_benchmark.py ingest \
          --state-json "$STATE_JSON" \
          --chunk-id "$chunk_id" \
          --response-jsonl "$response_path"
      else
        step_note "complete-next" "Nenhum arquivo de resposta encontrado para $chunk_id."
        step_note "complete-next" "Salve a resposta primeiro em:"
        printf "%s\n" "$response_path"
        exit 1
      fi
    fi

    if [[ "$(has_pending_chunk)" == "yes" ]]; then
      chunk_id="$(reserve_next_chunk)"
      open_chunk_flow "$chunk_id"
    else
      step_note "complete-next" "Não há mais chunks pendentes. Rode 'build-output' para consolidar o benchmark."
    fi
    ;;
  show)
    require_chunk_id "${3:-}"
    step_note "show" "Conteúdo do chunk $3. Envie-o no chat para adjudicação."
    cat "$CHUNKS_DIR/$3.jsonl"
    ;;
  show-latest)
    chunk_id="$(latest_exported_chunk_id)"
    step_note "show-latest" "Mostrando o chunk exportado mais recente."
    print_chunk_paths "$chunk_id"
    seed_rule_note "show-latest"
    step_note "show-latest" "Quando a resposta estiver salva, rode:"
    printf "bash scripts/codex_benchmark.sh %s ingest-latest\n" "$BENCH_DIR"
    echo
    cat "$CHUNKS_DIR/$chunk_id.jsonl"
    ;;
  response-path)
    require_chunk_id "${3:-}"
    step_note "response-path" "Agora cole a resposta adjudicada no arquivo abaixo."
    printf '%s\n' "$RESP_DIR/$3.jsonl"
    seed_rule_note "response-path"
    step_note "response-path" "Exemplo:"
    printf "cat > %s <<'EOF'\n" "$RESP_DIR/$3.jsonl"
    ;;
  ingest)
    require_chunk_id "${3:-}"
    python3 src/tools/manage_codex_adjudication_benchmark.py ingest \
      --state-json "$STATE_JSON" \
      --chunk-id "$3" \
      --response-jsonl "$RESP_DIR/$3.jsonl"
    step_note "ingest" "Chunk $3 ingerido com sucesso. Agora rode 'status' ou 'open-next'."
    ;;
  ingest-latest)
    chunk_id="$(latest_exported_chunk_id)"
    python3 src/tools/manage_codex_adjudication_benchmark.py ingest \
      --state-json "$STATE_JSON" \
      --chunk-id "$chunk_id" \
      --response-jsonl "$RESP_DIR/$chunk_id.jsonl"
    step_note "ingest-latest" "Chunk $chunk_id ingerido com sucesso. Agora rode 'status' ou 'open-next'."
    ;;
  status)
    python3 src/tools/manage_codex_adjudication_benchmark.py status \
      --state-json "$STATE_JSON"
    step_note "status" "Se ainda houver chunks pendentes, rode 'open-next' para continuar."
    ;;
  build-output)
    python3 src/tools/manage_codex_adjudication_benchmark.py build-output \
      --state-json "$STATE_JSON"
    step_note "build-output" "Saída consolidada gerada."
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    exit 1
    ;;
esac
