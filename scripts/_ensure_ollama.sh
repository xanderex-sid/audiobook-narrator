ensure_ollama() {
  local root="$1"
  export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
  if ! curl -fsS -m 2 "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo "starting ollama (model store: $root/models/ollama) ..."
    OLLAMA_MODELS="$root/models/ollama" nohup "$HOME/.local/bin/ollama" serve \
      > "$root/data/ollama-serve.log" 2>&1 &
    for _ in $(seq 1 20); do curl -fsS -m2 "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break; sleep 1; done
  fi
}
