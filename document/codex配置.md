# linux
mkdir -p ~/.codex
[ -f ~/.codex/config.toml ] && cp ~/.codex/config.toml ~/.codex/config.toml.bak

DDSS_API_KEY="sk-zaUZdG0XoU8YGM83Qgt7yLzcfIWGq6CnfHdVOskn5wXPydv4"
export DDSS_API_KEY

grep -v '^export DDSS_API_KEY=' ~/.bashrc > ~/.bashrc.tmp 2>/dev/null || true
mv ~/.bashrc.tmp ~/.bashrc
printf 'export DDSS_API_KEY=%q\n' "$DDSS_API_KEY" >> ~/.bashrc

cat > ~/.codex/config.toml <<'EOF'
model = "gpt-5.5"
model_provider = "ddss"

[model_providers.ddss]
name = "DDSS"
base_url = "https://code.ddsst.online/v1"
env_key = "DDSS_API_KEY"
wire_api = "responses"
EOF

# windows
$codexHome = "$env:USERPROFILE\.codex"
New-Item -ItemType Directory -Force $codexHome | Out-Null

setx DDSS_API_KEY "sk-zaUZdG0XoU8YGM83Qgt7yLzcfIWGq6CnfHdVOskn5wXPydv4"
$env:DDSS_API_KEY = "sk-zaUZdG0XoU8YGM83Qgt7yLzcfIWGq6CnfHdVOskn5wXPydv4"

$config = "$codexHome\config.toml"
if (Test-Path $config) { Copy-Item $config "$config.bak" -Force }

@'
model = "gpt-5.5"
model_provider = "ddss"

[model_providers.ddss]
name = "DDSS"
base_url = "https://code.ddsst.online/v1"
env_key = "DDSS_API_KEY"
wire_api = "responses"
'@ | Set-Content -Encoding UTF8 $config