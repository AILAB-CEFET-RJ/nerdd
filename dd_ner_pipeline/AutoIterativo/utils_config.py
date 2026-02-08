# --- utils_config.py (ou dentro do treino.py) ---
from pathlib import Path
import json, shutil, os
from transformers import AutoConfig

def _atomic_write_json(payload: dict, dst_path: Path):
    tmp = dst_path.with_suffix(".tmp")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, dst_path)

def save_config_with_fallback(model, out_dir, labels,
                              template_path="config.json",
                              encoder_name_default="neuralmind/bert-base-portuguese-cased"):
    """
    Salva config.json via HF; se der erro, copia template; se ainda falhar, gera um mínimo viável.
    Retorna True/False informando sucesso final.
    """
    out_dir = Path(out_dir)
    cfg_file = out_dir / "config.json"
    ok = False

    # 1) Tenta salvar via HF (config do encoder + labels)
    try:
        base = getattr(model, "model", None) or getattr(model, "encoder", None) or getattr(model, "transformer", None)
        if base is not None and hasattr(base, "config"):
            cfg = base.config
        else:
            enc_name = getattr(getattr(model, "data_processor", None), "transformer_name", encoder_name_default)
            cfg = AutoConfig.from_pretrained(enc_name, trust_remote_code=True)

        id2label = {i: lab for i, lab in enumerate(labels)}
        label2id = {lab: i for i, lab in enumerate(labels)}
        cfg.id2label = id2label
        cfg.label2id = label2id
        cfg.num_labels = len(labels)

        cfg.save_pretrained(out_dir)  # cria config.json
        ok = cfg_file.exists()
    except Exception as e:
        print(f"[warn] save_pretrained(config) falhou: {e}")

    # 2) Fallback: copiar template externo
    if not ok and Path(template_path).exists():
        try:
            shutil.copyfile(template_path, cfg_file)
            ok = True
            print(f"[config] Template copiado -> {cfg_file}")
        except Exception as e:
            print(f"[warn] copy(template) falhou: {e}")

    # 3) Fallback: salvar config do encoder "puro"
    if not ok:
        try:
            enc_name = getattr(getattr(model, "data_processor", None), "transformer_name", encoder_name_default)
            AutoConfig.from_pretrained(enc_name, trust_remote_code=True).save_pretrained(out_dir)
            ok = cfg_file.exists()
        except Exception as e:
            print(f"[warn] salvar config do encoder falhou: {e}")

    # 4) Último recurso: JSON mínimo viável
    if not ok:
        try:
            payload = {
                "model_type": "bert",
                "num_labels": len(labels),
                "id2label": {str(i): lab for i, lab in enumerate(labels)},
                "label2id": {lab: i for i, lab in enumerate(labels)},
                "output_hidden_states": False,
                "output_attentions": False
            }
            _atomic_write_json(payload, cfg_file)
            ok = True
            print(f"[config] Mínimo viável gerado -> {cfg_file}")
        except Exception as e:
            print(f"[erro] não consegui criar nenhum config.json: {e}")

    # 5) Validação leve
    if ok:
        try:
            AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
            print(f"[config] OK -> {cfg_file}")
        except Exception as e:
            ok = False
            print(f"[warn] config.json inválido: {e}")

    return ok
