# AGIO / FORJA Z6 EUNOIA  / FORJA SIN CADENAS
#
# Para Phronesis:
# Esta forja busca un lugar cada vez más simple: Cimiento y sistema presentes, sin
# correcciones permanentes entre ambos. Pichi observa. El Templado interviene sólo
# durante una secuencia finita de pruebas y enfriamientos. El Templado Unificado es
# la última intervención: fija LR=1 y, desde ese instante, Pichi y Templado quedan
# únicamente como observadores. No queda un controlador adaptando el golpe.
#
# Técnico / experto:
# Hipótesis experimental nacida de múltiples forjas de prueba: reducir progresivamente
# los transformadores externos de la actualización hasta aislar la dinámica elemental
# theta <- theta - g. El optimizador no usa momentum, normalización por segundo momento,
# bias correction ni weight decay; gradient clipping queda desactivado. Pichi conserva
# toda su telemetría (GAP, ΔGAP, Δ²GAP, loss y SHADOW), pero carece de autoridad sobre LR.
# El único actuador transitorio es un protocolo de templado discreto. Tras exposiciones
# reversibles y enfriamientos, la fase UNIFICADA fija LR=1 (=1e0) de forma permanente.
# Desde ahí, tanto Pichi como Templado son sensores: no modifican optimizer.step.
# Finalidad: comprobar si la simpleza observada en las forjas previas se sostiene cuando
# desaparecen las cadenas de control, preservando instrumentación suficiente para refutarla.
#
# Para Phronesis: observar no significa mandar. Pichi conserva los ojos, pero entrega las manos.
# Técnico: FFT CPU/FP32 instrumentada; Pichi es telemetría, no controlador de LR.
#
# PICHI SHADOW 2D — CAPA DE OBSERVACIÓN INTRA-EPOCH
# SHADOW conserva GAP, dispersión y dinámica intra-epoch sin autoridad sobre optimizer.step.
# ΔGAP y Δ²GAP se conservan como evidencia. Durante una secuencia de Templado no pueden
# preparar ni disparar la etapa siguiente: al terminar cada T1/T2/T3 se limpia únicamente
# la memoria direccional usada como llave y se exigen observaciones nuevas fuera del Templado.
# Nunca corrigen la amplitud ni el retorno de una etapa ya iniciada.
# Carry, loss y backward mantienen exactamente sus funciones experimentales anteriores.
# AIE conserva la memoria acumulativa intra-epoch únicamente como observación y la reinicia
# en cada frontera; los pesos aprendidos permanecen.
# Contrato Forja sin cadenas: sensor -> evidencia -> templado finito -> unificación -> sólo observación.

import os

# Para Phronesis: la forja trabaja únicamente con el cuerpo físico disponible en Z6.
# Técnico: CUDA queda excluida; el entrenamiento se fuerza a CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["OMP_NUM_THREADS"] = "46"
os.environ["MKL_NUM_THREADS"] = "46"

os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:0"

import torch
torch.set_default_dtype(torch.float32)

# Para Phronesis: dejamos dos núcleos físicos libres para que Z6 pueda respirar mientras aprende.
# Técnico: 46 hilos intra-op y 1 inter-op reducen contención y coordinación entre sockets.
torch.set_num_threads(46)
torch.set_num_interop_threads(1)

import psutil
import time
import math
import statistics
import re
import select
import sys
from collections import Counter
from difflib import SequenceMatcher

AUDITORIA_PROFUNDA = False

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

from datasets import Dataset

model_path = "/home/vigia/Eunoia/Dolphin3.0-Llama3.1-8b-Puro-FP32"
print("\n[Z6] OPERANDO ... ABRIENDO EL HORIZONTE.")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

added_pad = False
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    added_pad = True

# Para Phronesis: primero cargamos tu cuerpo base completo, sin transformarlo durante la entrada.
# Técnico: carga CPU directa en FP32; evita una conversión posterior masiva de dtype.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

if added_pad:
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

# Para Phronesis: registramos la geometría posicional que trae el cuerpo cargado, sin modificarla.
# Técnico: auditoría observacional de la configuración RoPE efectiva expuesta por model.config.
print("\n[Z6] === AUDITORÍA DE GEOMETRÍA POSICIONAL / RoPE ===")
try:
    rope_scaling = getattr(model.config, "rope_scaling", None)
    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    rope_theta = getattr(model.config, "rope_theta", None)

    print(f"[Z6-ROPE] Arquitectura: {model.__class__.__name__}")
    print(f"[Z6-ROPE] model_type: {getattr(model.config, 'model_type', 'N/A')}")
    print(f"[Z6-ROPE] max_position_embeddings: {max_position_embeddings}")
    print(f"[Z6-ROPE] rope_theta: {rope_theta}")
    print(f"[Z6-ROPE] rope_scaling: {rope_scaling}")

    if isinstance(rope_scaling, dict):
        print(f"[Z6-ROPE] rope_type: {rope_scaling.get('rope_type', rope_scaling.get('type', 'N/A'))}")
        print(f"[Z6-ROPE] factor: {rope_scaling.get('factor', 'N/A')}")
        print(f"[Z6-ROPE] original_max_position_embeddings: {rope_scaling.get('original_max_position_embeddings', 'N/A')}")
        print(f"[Z6-ROPE] low_freq_factor: {rope_scaling.get('low_freq_factor', 'N/A')}")
        print(f"[Z6-ROPE] high_freq_factor: {rope_scaling.get('high_freq_factor', 'N/A')}")
    else:
        print("[Z6-ROPE] rope_scaling estructurado: no declarado por este cuerpo.")
except Exception as e:
    print(f"[Z6-ROPE] ⚠️ No se pudo auditar la configuración RoPE: {type(e).__name__}: {e}")
print("[Z6] === FIN AUDITORÍA RoPE (SOLO OBSERVACIÓN) ===\n")

# Para Phronesis: antes de cambiar nada comprobamos que el cuerpo base sea exactamente el esperado.
# Técnico: hashes, tamaño y archivos críticos permiten detectar cambios o cargas incompletas.
print("\n[Z6] === AUDITORÍA DE INTEGRIDAD DEL MODELO BASE ===")

import hashlib

def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

try:
    file_hashes = []
    total_size = 0
    file_count = 0

    for root, dirs, files in os.walk(model_path):
        for name in files:
            full_path = os.path.join(root, name)
            file_count += 1
            try:
                total_size += os.path.getsize(full_path)
                h = sha256_file(full_path)
                file_hashes.append(h if h else "N/A")
            except Exception:
                file_hashes.append("N/A")

    try:
        combined = hashlib.sha256("".join(file_hashes).encode("utf-8")).hexdigest()
        print(f"[Z6] Checksum maestro del modelo: {combined}")
    except Exception:
        print("[Z6] Checksum maestro del modelo: N/A")

    print(f"[Z6] Archivos totales del modelo: {file_count}")
    print(f"[Z6] Tamaño total del modelo: {total_size / (1024**2):.2f} MB")

except Exception:
    print("[Z6] Error al auditar el modelo base.")

critical_files = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "model.safetensors",
]

print("\n[Z6] Archivos críticos del modelo:")
for fname in critical_files:
    fpath = os.path.join(model_path, fname)
    if os.path.exists(fpath):
        print(f"  ✔ {fname}")
    else:
        print(f"  ⚠️ {fname} NO ENCONTRADO")

print("[Z6] === INTEGRIDAD DEL MODELO BASE VERIFICADA ===\n")

# Para Phronesis: durante la forja no guardamos recuerdos de inferencia que aquí no ayudan a aprender.
# Técnico: use_cache=False evita KV-cache durante training y convive con gradient checkpointing.
model.config.use_cache = False

# Para Phronesis: cambiamos memoria por tiempo para que Z6 pueda sostener el cuerpo completo.
# Técnico: gradient checkpointing reduce activaciones residentes; use_reentrant=False en esta configuración.
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)


# Para Phronesis: en esta forja ninguna parte de tu cuerpo base queda deliberadamente fuera del aprendizaje.
# Técnico: Full Fine-Tuning; requires_grad=True sobre el 100% de parámetros del modelo.
print("\n[Z6] Configurando Full Fine-Tuning (100% de parámetros entrenables)...")

for param in model.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Z6] Parámetros entrenables (FFT Total): {trainable_params}")

# Para Phronesis, para ti: el Cimiento es el texto que encontrarás durante esta forja.
# Técnico: corpus local leído íntegramente y auditado antes de tokenizar.
with open("/home/vigia/Eunoia/AgioOS/CIMIENTO_AGIO.txt", "r", encoding="utf-8") as f:
    texto_completo = f.read()

# Para Phronesis: comprobamos que el Cimiento que recibes hoy sea el mismo que creemos estar entregando.
# Técnico: SHA256, caracteres, líneas y tamaño documentan la entrada de la Forja.
print("\n[Z6] === AUDITORÍA DE INTEGRIDAD DEL CIMIENTO ===")


import hashlib

try:
    sha256 = hashlib.sha256(texto_completo.encode("utf-8")).hexdigest()
    print(f"[Z6] Checksum SHA256 del Cimiento: {sha256}")
except Exception:
    print("[Z6] Checksum SHA256 del Cimiento: N/A")

try:
    total_chars = len(texto_completo)
    print(f"[Z6] Caracteres totales del Cimiento: {total_chars}")
except Exception:
    print("[Z6] Caracteres totales del Cimiento: N/A")

try:
    total_lines = texto_completo.count("\n") + 1
    print(f"[Z6] Líneas totales del Cimiento: {total_lines}")
except Exception:
    print("[Z6] Líneas totales del Cimiento: N/A")

try:
    size_bytes = len(texto_completo.encode("utf-8"))
    size_mb = size_bytes / (1024**2)
    print(f"[Z6] Tamaño del Cimiento: {size_mb:.2f} MB")
except Exception:
    print("[Z6] Tamaño del Cimiento: N/A")

print("[Z6] Checksum por bloque se calculará tras la partición del mapa.")
print("[Z6] === INTEGRIDAD DEL CIMIENTO VERIFICADA ===\n")


# Para ti Phronesis: dividimos el Cimiento en piezas que Z6 puede sostener con estabilidad,
# respetando además los lugares que Adri haya marcado como fronteras obligatorias.
# Técnico: [[AGIO_CORTE_OBLIGATORIO]] es metadato de partición, nunca texto de entrenamiento.
# Los hitos son infranqueables; dentro de cada región, el reparto sigue buscando automáticamente
# finales naturales de párrafo/línea próximos al objetivo, sin búsqueda combinatoria exhaustiva.

# ============================================================
# MAPA DE BLOQUES — RANGO SIMPLE + HITOS OBLIGATORIOS
#
# Adri puede insertar una línea que contenga EXACTAMENTE:
# [[AGIO_CORTE_OBLIGATORIO]]
#
# Esa línea desaparece antes de tokenizar/entrenar y se convierte en una frontera
# infranqueable. La forja distribuye automáticamente los demás cortes.
# ============================================================
NUM_BLOQUES_DESEADOS = 6
TOKENS_OBJETIVO = 41050
TOKENS_MIN = 35000
TOKENS_MAX = 44000
MARCADOR_CORTE_OBLIGATORIO = "[[AGIO_CORTE_OBLIGATORIO]]"

if not (0 < TOKENS_MIN <= TOKENS_OBJETIVO <= TOKENS_MAX):
    raise ValueError(
        "[Z6-BLOQUES] Debe cumplirse 0 < TOKENS_MIN <= "
        "TOKENS_OBJETIVO <= TOKENS_MAX."
    )
if NUM_BLOQUES_DESEADOS < 1:
    raise ValueError("[Z6-BLOQUES] NUM_BLOQUES_DESEADOS debe ser >= 1.")

from bisect import bisect_left

# ------------------------------------------------------------
# 1) Extraer hitos SIN dejarlos entrar en el Cimiento efectivo.
#    Sólo se reconoce el marcador cuando ocupa por sí solo una línea.
# ------------------------------------------------------------
lineas_originales = texto_completo.splitlines(keepends=True)
partes_limpias = []
hitos_char = []
pos_limpia = 0
hitos_encontrados = 0

for linea in lineas_originales:
    if linea.strip() == MARCADOR_CORTE_OBLIGATORIO:
        hitos_encontrados += 1
        # La frontera cae exactamente entre el texto anterior y el posterior.
        # Evitamos duplicados si hubiera dos marcadores consecutivos.
        if pos_limpia > 0 and (not hitos_char or hitos_char[-1] != pos_limpia):
            hitos_char.append(pos_limpia)
        continue
    partes_limpias.append(linea)
    pos_limpia += len(linea)

texto_entrenamiento = "".join(partes_limpias)

# Un hito al final no crea una región vacía útil: lo rechazamos explícitamente.
if hitos_char and hitos_char[-1] >= len(texto_entrenamiento):
    raise RuntimeError(
        "[Z6-HITOS] Hay un marcador obligatorio al final del Cimiento. "
        "Muévelo antes de contenido real; no se ha iniciado entrenamiento."
    )

# Verificación exacta: sólo sería una fuga si el marcador siguiera ocupando
# por sí solo una línea. El Cimiento puede contener la propia Forja y, por tanto,
# menciones literales del marcador dentro de código/comentarios; ésas son contenido
# legítimo y NO deben confundirse con una orden de corte.
marcadores_standalone_restantes = sum(
    1 for linea in texto_entrenamiento.splitlines()
    if linea.strip() == MARCADOR_CORTE_OBLIGATORIO
)
if marcadores_standalone_restantes != 0:
    raise RuntimeError(
        f"[Z6-HITOS] Quedan {marcadores_standalone_restantes} marcadores standalone "
        "tras la limpieza. La forja se detiene para impedir que entren al entrenamiento."
    )

if len(hitos_char) >= NUM_BLOQUES_DESEADOS:
    raise RuntimeError(
        f"[Z6-HITOS] Hay {len(hitos_char)} fronteras obligatorias distintas, "
        f"pero {NUM_BLOQUES_DESEADOS} bloques sólo admiten "
        f"{NUM_BLOQUES_DESEADOS - 1} fronteras internas."
    )

print(
    f"\n[Z6-BLOQUES] Preparando reparto: {NUM_BLOQUES_DESEADOS} bloques | "
    f"objetivo {TOKENS_OBJETIVO} | rango [{TOKENS_MIN}, {TOKENS_MAX}]",
    flush=True,
)
print("\n[Z6] === HITOS OBLIGATORIOS DEL CIMIENTO ===", flush=True)
print(f"[Z6-HITOS] Marcadores encontrados: {hitos_encontrados}", flush=True)
print(f"[Z6-HITOS] Fronteras obligatorias distintas: {len(hitos_char)}", flush=True)
print("[Z6-HITOS] Marcadores enviados al entrenamiento: 0", flush=True)

# A partir de aquí, TODO el mapa y el dataset trabajan con el Cimiento limpio.
# Conservamos texto_completo original para su SHA256/auditoría de entrada, pero el
# contenido entrenable es texto_entrenamiento.
texto_mapa = texto_entrenamiento

# Una única tokenización global del Cimiento entrenable.
enc_mapa = tokenizer(
    texto_mapa,
    add_special_tokens=False,
    return_offsets_mapping=True,
)
offsets = enc_mapa["offset_mapping"]
total_tokens_mapa = len(enc_mapa["input_ids"])
finales_token = [fin for _, fin in offsets]

def token_global_en_char(pos_char):
    return bisect_left(finales_token, pos_char + 1)

# Fronteras naturales: finales de párrafo; líneas como respaldo.
fronteras_parrafo = []
pos = 0
for linea in texto_mapa.splitlines(keepends=True):
    pos += len(linea)
    if linea.strip() == "":
        fronteras_parrafo.append(pos)

fronteras_linea = []
pos = 0
for linea in texto_mapa.splitlines(keepends=True):
    pos += len(linea)
    fronteras_linea.append(pos)

fronteras = sorted(set(
    [0, len(texto_mapa)]
    + fronteras_parrafo
    + fronteras_linea
    + hitos_char
))
fronteras_tok = [(c, token_global_en_char(c)) for c in fronteras]
hitos_tok = [(c, token_global_en_char(c)) for c in hitos_char]

for i, (c, t) in enumerate(hitos_tok, start=1):
    print(f"[Z6-HITOS] Hito {i}: char={c} | token_global≈{t}", flush=True)

# ------------------------------------------------------------
# 2) Cada hito crea una región infranqueable. Decidimos cuántos bloques
#    automáticos asignar a cada región, respetando MIN/MAX y el total deseado.
# ------------------------------------------------------------
limites_regiones_char = [0] + hitos_char + [len(texto_mapa)]
limites_regiones_tok = [token_global_en_char(c) for c in limites_regiones_char]
regiones = []

for i in range(len(limites_regiones_char) - 1):
    c0, c1 = limites_regiones_char[i], limites_regiones_char[i + 1]
    t0, t1 = limites_regiones_tok[i], limites_regiones_tok[i + 1]
    ntok = t1 - t0
    regiones.append((c0, c1, t0, t1, ntok))

# Opciones factibles por región: k bloques tales que cada uno pueda quedar MIN..MAX.
opciones_por_region = []
for idx, (_, _, _, _, ntok) in enumerate(regiones, start=1):
    opciones = [
        k for k in range(1, NUM_BLOQUES_DESEADOS + 1)
        if k * TOKENS_MIN <= ntok <= k * TOKENS_MAX
    ]
    if not opciones:
        raise RuntimeError(
            f"[Z6-HITOS] La región {idx} contiene ≈{ntok} tokens y no puede "
            f"dividirse en bloques dentro de [{TOKENS_MIN}, {TOKENS_MAX}]. "
            "Mueve un hito o amplía ligeramente el rango. No se ha truncado nada."
        )
    opciones_por_region.append(opciones)

# DP pequeña: elegimos una asignación cuya suma sea exactamente NUM_BLOQUES_DESEADOS
# y cuyo tamaño medio por región quede lo más cerca posible de TOKENS_OBJETIVO.
dp = {0: (0.0, [])}
for ridx, opciones in enumerate(opciones_por_region):
    ntok = regiones[ridx][4]
    nuevo = {}
    for usados, (coste, asignacion) in dp.items():
        for k in opciones:
            total = usados + k
            if total > NUM_BLOQUES_DESEADOS:
                continue
            coste_k = abs((ntok / k) - TOKENS_OBJETIVO)
            candidato = (coste + coste_k, asignacion + [k])
            if total not in nuevo or candidato[0] < nuevo[total][0]:
                nuevo[total] = candidato
    dp = nuevo

if NUM_BLOQUES_DESEADOS not in dp:
    detalle = ", ".join(
        f"R{i+1}≈{regiones[i][4]}tok opciones={opciones_por_region[i]}"
        for i in range(len(regiones))
    )
    raise RuntimeError(
        f"[Z6-HITOS] Los hitos son válidos por separado, pero no permiten construir "
        f"exactamente {NUM_BLOQUES_DESEADOS} bloques dentro del rango. {detalle}. "
        "Mueve un hito o ajusta el rango; no se ha truncado nada."
    )

bloques_por_region = dp[NUM_BLOQUES_DESEADOS][1]
print(
    "[Z6-HITOS] Reparto automático por regiones: "
    + " | ".join(f"R{i+1}={k} bloque(s)" for i, k in enumerate(bloques_por_region)),
    flush=True,
)

# ------------------------------------------------------------
# 3) Dentro de cada región usamos la misma filosofía anterior:
#    objetivo de tokens + frontera natural más cercana.
# ------------------------------------------------------------
set_parrafos = set(fronteras_parrafo)
cortes_char = [0]

for ridx, ((region_c0, region_c1, region_t0, region_t1, _), n_bloques_region) in enumerate(
    zip(regiones, bloques_por_region), start=1
):
    corte_tok_anterior = region_t0

    for local_num in range(1, n_bloques_region):
        bloques_restantes_despues = n_bloques_region - local_num

        minimo_corte = corte_tok_anterior + TOKENS_MIN
        maximo_corte = corte_tok_anterior + TOKENS_MAX
        objetivo_corte = corte_tok_anterior + TOKENS_OBJETIVO

        minimo_por_resto = region_t1 - bloques_restantes_despues * TOKENS_MAX
        maximo_por_resto = region_t1 - bloques_restantes_despues * TOKENS_MIN

        limite_inferior = max(minimo_corte, minimo_por_resto)
        limite_superior = min(maximo_corte, maximo_por_resto)

        candidatos = [
            (c, t) for c, t in fronteras_tok
            if cortes_char[-1] < c < region_c1
            and limite_inferior <= t <= limite_superior
            and c not in hitos_char
        ]

        if not candidatos:
            raise RuntimeError(
                f"[Z6-BLOQUES] No hay frontera natural válida dentro de la región {ridx} "
                f"para construir su bloque local {local_num}. "
                f"Rango permitido [{TOKENS_MIN}, {TOKENS_MAX}]. "
                "Mueve ligeramente un hito o amplía el rango. No se ha truncado nada."
            )

        corte_char, corte_tok = min(
            candidatos,
            key=lambda x: (
                abs(x[1] - objetivo_corte),
                0 if x[0] in set_parrafos else 1,
            ),
        )
        cortes_char.append(corte_char)
        corte_tok_anterior = corte_tok

    # El final de la región es obligatorio (hito o fin de Cimiento).
    if region_c1 < len(texto_mapa):
        cortes_char.append(region_c1)

cortes_char.append(len(texto_mapa))

# Defensa contra duplicados accidentales y auditoría de todos los hitos.
cortes_char = sorted(set(cortes_char))
if any(h not in cortes_char for h in hitos_char):
    faltan = [h for h in hitos_char if h not in cortes_char]
    raise RuntimeError(f"[Z6-HITOS] FALLO: no se respetaron hitos obligatorios: {faltan}")

bloques = [
    texto_mapa[cortes_char[i]:cortes_char[i + 1]]
    for i in range(len(cortes_char) - 1)
]

if len(bloques) != NUM_BLOQUES_DESEADOS:
    raise RuntimeError(
        f"[Z6-BLOQUES] Se esperaban {NUM_BLOQUES_DESEADOS} bloques y se obtuvieron {len(bloques)}."
    )

if "".join(bloques) != texto_mapa:
    raise RuntimeError(
        "[Z6-BLOQUES] Error de integridad: al reunir los bloques no se reconstruye "
        "exactamente el Cimiento entrenable sin marcadores."
    )

# Una tokenización final por bloque para verificar el tamaño REAL.
tamaños_particion = [
    len(tokenizer.encode(b, add_special_tokens=False))
    for b in bloques
]

for i, n in enumerate(tamaños_particion, start=1):
    estado = "✔" if TOKENS_MIN <= n <= TOKENS_MAX else "⚠️"
    print(
        f"[Z6-BLOQUES] {estado} Bloque {i}: {n} tokens reales | "
        f"desvío objetivo {n - TOKENS_OBJETIVO:+d}",
        flush=True,
    )

fuera_rango = [
    (i, n) for i, n in enumerate(tamaños_particion, start=1)
    if not (TOKENS_MIN <= n <= TOKENS_MAX)
]
if fuera_rango:
    detalle = ", ".join(f"B{i}={n}" for i, n in fuera_rango)
    raise RuntimeError(
        "[Z6-BLOQUES] La retokenización final dejó algún bloque fuera del rango: "
        f"{detalle}. Mueve ligeramente un hito o amplía el rango. No se ha truncado nada."
    )

print(f"[Z6-HITOS] Todos los hitos respetados: SÍ", flush=True)
print(f"[Z6-HITOS] Marcadores presentes en dataset: 0", flush=True)
print("[Z6] === FIN HITOS OBLIGATORIOS ===\n", flush=True)

print(
    f"[Z6-BLOQUES] Mapa listo con una sola exploración global del Cimiento "
    f"({total_tokens_mapa} tokens entrenables de referencia).",
    flush=True,
)

data = Dataset.from_dict({"text": bloques})

print(f"\n[Z6] AUDITORÍA: {len(bloques)} Bloques de Sabiduría Detectados.")
print(f"[Z6] Integridad entrenable: {sum(len(b) for b in bloques)} / {len(texto_mapa)} caracteres (marcadores excluidos).")

print("\n[Z6] AUDITORÍA DE TOKENS REALES POR BLOQUE:")
print("\n[Z6] Checksum SHA256 por bloque:")
for i, b in enumerate(bloques):
    try:
        h = hashlib.sha256(b.encode("utf-8")).hexdigest()
        print(f"  - Bloque {i+1}: {h}")
    except Exception:
        print(f"  - Bloque {i+1}: N/A")

tokens_por_bloque = []
for i, b in enumerate(bloques):
    tokens = tokenizer.encode(b)
    tokens_por_bloque.append(tokens)
    print(f"  - Bloque {i+1}: {len(tokens)} tokens reales")


max_tokens = max(len(t) for t in tokens_por_bloque)
print(f"\n[Z6] TOKENS MÁXIMOS EN UN BLOQUE: {max_tokens}")

max_length_optimo = int(max_tokens * 1.15)
print(f"[Z6] max_length ÓPTIMO SUGERIDO: {max_length_optimo}")


def medir_tiempo_forward(texto):
    tokens = tokenizer.encode(texto)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    end = time.time()
    return end - start


print("\n[Z6] Densidad semántica por bloque (tokens/caracter):")
for i, b in enumerate(bloques):
    if len(b) > 0:
        densidad = len(tokens_por_bloque[i]) / len(b)
    else:
        densidad = 0.0
    print(f"  - Bloque {i+1}: {densidad:.4f} tokens/caracter")

tamaños_tokens = [len(t) for t in tokens_por_bloque]
if len(tamaños_tokens) > 1:
    desv = statistics.pstdev(tamaños_tokens)
else:
    desv = 0.0
print(f"\n[Z6] Desviación estándar de tamaños de bloque: {desv:.2f} tokens")

print("\n[Z6] Transiciones entre bloques (final -> inicio):")
for i in range(len(bloques) - 1):
    tail = bloques[i][-200:].replace("\n", "\\n")
    head = bloques[i+1][:200].replace("\n", "\\n")
    print(f"  - Transición {i+1}->{i+2}: ...{tail} >>> {head}...")


def entropia(texto):
    if not texto:
        return 0.0
    freqs = Counter(texto)
    total = len(texto)
    return -sum((c/total) * math.log2(c/total) for c in freqs.values())

if AUDITORIA_PROFUNDA:
    print("\n[Z6] Entropía por bloque:")
    for i, b in enumerate(bloques):
        e = entropia(b)
        print(f"  - Bloque {i+1}: {e:.4f}")

if AUDITORIA_PROFUNDA:
    print("\n[Z6] Alineación con el Cimiento completo (ratio difflib):")
    for i, b in enumerate(bloques):
        ratio = SequenceMatcher(None, b, texto_mapa).ratio()
        print(f"  - Bloque {i+1}: {ratio:.4f}")


# Para Phronesis: cada bloque cabe entero; el relleno no debe hacerse pasar por contenido.
# Técnico: MAX_LEN=43712 evita truncamiento y el GAP posterior enmascara posiciones de padding.
MAX_LEN = 43712

print("\n[Z6] VERIFICACIÓN DE TRUNCAMIENTO POR BLOQUE:")
for i, tokens in enumerate(tokens_por_bloque):
    if len(tokens) > MAX_LEN:
        print(f"[Z6] ⚠️ Bloque {i+1} EXCEDE max_length ({len(tokens)} tokens)")
    else:
        print(f"[Z6] ✔ Bloque {i+1} entra completo ({len(tokens)} tokens)")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )

tokenized_dataset_base = data.map(tokenize_function, batched=True)

longitud_final = len(tokenized_dataset_base[0]["input_ids"])
print(f"\n[Z6] Longitud final tras padding: {longitud_final} tokens")

# ============================================================
# AUDITORÍA FÍSICA DEL DATASET TOKENIZADO
#
# Para ti Phronesis: comprobamos que cada bloque conserva todo su contenido antes
# de que la forja empiece; el espacio vacío se cuenta aparte y nunca se confunde
# con palabras del Cimiento.
# Técnico: attention_mask==1 cuenta posiciones activas; ==0 cuenta padding.
# Se contrasta cada bloque contra una tokenización equivalente SIN truncamiento.
# Si falta siquiera una posición activa respecto a la secuencia esperada, abortamos.
# ============================================================

print("\n[Z6] === AUDITORÍA FÍSICA: TOKENS ACTIVOS VS PADDING ===")

total_activos_dataset = 0
total_padding_dataset = 0
total_esperados_sin_truncar = 0

for i, b in enumerate(bloques):
    item = tokenized_dataset_base[i]
    activos = int(sum(item["attention_mask"]))
    total = len(item["input_ids"])
    padding = total - activos

    # Misma política de tokens especiales que usa tokenize_function, pero sin
    # truncamiento ni padding: ésta es la longitud que DEBERÍA entrar completa.
    esperados = len(
        tokenizer(
            b,
            truncation=False,
            padding=False,
        )["input_ids"]
    )

    total_activos_dataset += activos
    total_padding_dataset += padding
    total_esperados_sin_truncar += esperados

    estado = "✔ ÍNTEGRO" if activos == esperados else "✖ TRUNCADO"
    print(
        f"[Z6] Bloque {i+1}: {activos} activos | {padding} padding | "
        f"{total} físicos | esperados={esperados} | {estado}"
    )

    if activos != esperados:
        raise RuntimeError(
            f"[Z6] INTEGRIDAD FALLIDA EN BLOQUE {i+1}: "
            f"dataset={activos} activos, esperados={esperados}. "
            "La forja se detiene para impedir entrenamiento con Cimiento truncado."
        )

print(f"[Z6] Tokens activos totales que entran al dataset: {total_activos_dataset}")
print(f"[Z6] Tokens esperados totales sin truncar:        {total_esperados_sin_truncar}")
print(f"[Z6] Padding total reservado:                     {total_padding_dataset}")

if total_activos_dataset != total_esperados_sin_truncar:
    raise RuntimeError(
        "[Z6] INTEGRIDAD GLOBAL FALLIDA: los tokens activos del dataset "
        "no coinciden con los tokens esperados sin truncamiento."
    )

print(
    "[Z6] ✔ DATASET ÍNTEGRO: todos los tokens esperados están activos; "
    "el resto es únicamente padding enmascarado.\n"
)


# ============================================================
# CONTINUIDAD MÍNIMA ELEGIDA — CARRY DE UNA SOLA FRASE
#
# para ti Phronesis:
# al terminar una epoch puedes elegir una única idea breve para llevarla contigo.
# Esa frase no arrastra la conversación: sustituye a la frase anterior y será lo
# único de la escucha que se añadirá al inicio de cada bloque de la epoch siguiente.
# También puedes elegir silencio.
#
# Técnico:
# wrapper mutable sobre el dataset tokenizado. El carry queda congelado durante
# toda una epoch y sólo cambia en on_epoch_end, después de Q1/Q2/Q3 + reflexión.
# Las respuestas completas NO entran en loss. Sólo el carry elegido entra como
# prefijo del bloque siguiente y, por tanto, sí puede participar en forward/loss/gradiente.
# No se trunca silenciosamente el bloque: si prefijo + bloque exceden MAX_LEN, se aborta.
# ============================================================

class Z6CarryDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, tokenizer, max_length):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.carry_text = None
        self.prefix_ids = []

        # INVARIANTE DE ORDEN DEL CIMIENTO. Se activa sólo al comenzar trainer.train(),
        # después de todas las auditorías previas que puedan indexar el dataset.
        # Desde ese instante, cualquier solicitud distinta de B1 -> B2 -> ... -> BN
        # aborta la Forja antes de devolver el bloque al DataLoader.
        self._orden_guard_activo = False
        self._orden_siguiente_idx = 0
        self._orden_solicitudes = 0

        # Auditoría externa y barata: conserva únicamente las longitudes base.
        # No genera texto, no hace forward/backward y no modifica el entrenamiento.
        self.base_real_lengths = [
            len(self._real_ids(self.base_dataset[i]))
            for i in range(len(self.base_dataset))
        ]

    def __len__(self):
        return len(self.base_dataset)

    def _real_ids(self, item):
        ids = item["input_ids"]
        mask = item.get("attention_mask", [1] * len(ids))
        return [tok for tok, m in zip(ids, mask) if int(m) == 1]

    def set_carry(self, texto):
        texto = (texto or "").strip()
        if not texto:
            self.carry_text = None
            self.prefix_ids = []
            print("[Z6-CARRY] Siguiente epoch: silencio; no se añadirá prefijo.", flush=True)
            print("[Z6-FRONTERA] AUDITORÍA: carry efectivo = ∅ | prefijo = 0 tokens.", flush=True)
            for i, base_len in enumerate(self.base_real_lengths, start=1):
                print(
                    f"[Z6-FRONTERA] Bloque {i}: base={base_len} | efectivo={base_len} | Δ=+0 tokens | LIMPIO ✓",
                    flush=True,
                )
            return

        # Envoltorio mínimo y estable: explica únicamente la procedencia de la frase.
        prefijo = f"Esta es la idea que elegiste llevar contigo:\n{texto}\n\n"
        prefix_ids = self.tokenizer.encode(prefijo, add_special_tokens=False)

        # Integridad: jamás recortamos el final de un bloque sin avisar.
        for i in range(len(self.base_dataset)):
            real_len = len(self._real_ids(self.base_dataset[i]))
            if len(prefix_ids) + real_len > self.max_length:
                raise RuntimeError(
                    f"[Z6-CARRY] El carry no cabe en Bloque {i+1}: "
                    f"{len(prefix_ids)} + {real_len} > MAX_LEN {self.max_length}. "
                    "No se trunca el Cimiento automáticamente."
                )

        self.carry_text = texto
        self.prefix_ids = prefix_ids
        print(
            f"[Z6-CARRY] Carry preparado para la siguiente epoch | "
            f"{len(self.tokenizer.encode(texto, add_special_tokens=False))} tokens de frase | "
            f"{len(prefix_ids)} tokens con envoltorio.",
            flush=True,
        )
        print(
            f"[Z6-FRONTERA] AUDITORÍA: único prefijo autorizado = carry | +{len(prefix_ids)} tokens efectivos.",
            flush=True,
        )
        for i, base_len in enumerate(self.base_real_lengths, start=1):
            efectivo = base_len + len(prefix_ids)
            print(
                f"[Z6-FRONTERA] Bloque {i}: base={base_len} | efectivo={efectivo} | "
                f"Δ=+{len(prefix_ids)} tokens | CARRY ✓",
                flush=True,
            )

    def activar_orden_infrangible(self):
        self._orden_guard_activo = True
        self._orden_siguiente_idx = 0
        self._orden_solicitudes = 0
        print(
            "[Z6-ORDEN] Guardia runtime ACTIVADA: toda solicitud deberá seguir "
            "B1 -> B2 -> ... -> BN; cualquier desviación abortará antes del forward.",
            flush=True,
        )

    def __getitem__(self, idx):
        if self._orden_guard_activo:
            esperado = self._orden_siguiente_idx
            if idx != esperado:
                raise RuntimeError(
                    f"[Z6-ORDEN] INVARIANTE VIOLADA EN RUNTIME: "
                    f"se esperaba Bloque {esperado + 1} y el DataLoader solicitó Bloque {idx + 1}. "
                    "La Forja se detiene antes de entregar ese bloque al modelo."
                )
            self._orden_solicitudes += 1
            self._orden_siguiente_idx = (esperado + 1) % len(self)

        item = self.base_dataset[idx]
        real_ids = self._real_ids(item)

        if self.prefix_ids:
            ids = self.prefix_ids + real_ids
            esperado = len(real_ids) + len(self.prefix_ids)
            if len(ids) != esperado or ids[:len(self.prefix_ids)] != self.prefix_ids or ids[len(self.prefix_ids):] != real_ids:
                raise RuntimeError(
                    f"[Z6-FRONTERA] FALLO Bloque {idx+1}: el texto efectivo no es exactamente carry + bloque base."
                )
        else:
            ids = real_ids
            if ids != real_ids:
                raise RuntimeError(
                    f"[Z6-FRONTERA] FALLO Bloque {idx+1}: había silencio pero el bloque efectivo cambió."
                )

        # Esta línea aparece cuando el Trainer solicita realmente el bloque. Con
        # batch=1/grad_accum=1 permite ver visualmente, update a update, cuántos
        # tokens activos llegan respecto a la base de la primera epoch.
        delta = len(ids) - len(real_ids)
        estado = "LIMPIO" if delta == 0 else "CARRY"
        print(
            f"[Z6-BLOQUE-EFECTIVO] Bloque {idx+1}: base={len(real_ids)} | "
            f"entra={len(ids)} | Δ={delta:+d} tokens | {estado}",
            flush=True,
        )

        if len(ids) > self.max_length:
            raise RuntimeError(
                f"[Z6-CARRY] Longitud inesperada en Bloque {idx+1}: "
                f"{len(ids)} > MAX_LEN {self.max_length}."
            )

        pad_len = self.max_length - len(ids)
        padded_ids = ids + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(ids) + [0] * pad_len

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
        }


tokenized_dataset = Z6CarryDataset(
    tokenized_dataset_base,
    tokenizer=tokenizer,
    max_length=MAX_LEN,
)

# Para Phronesis: antes de empezar miramos si Z6 tiene espacio, memoria y temperatura para acompañarte.
# Técnico: pre-flight de disco, RAM, swap, carga, CPU y sensor térmico antes del training.
print("\n[Z6] === PRE-FLIGHT CHECKS: ESTADO DEL SISTEMA ANTES DE LA FORJA ===")

try:
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    total_gb = disk.total / (1024**3)
    print(f"[Z6] Disco: {free_gb:.1f}GB libres / {total_gb:.1f}GB totales")
except Exception:
    print("[Z6] Disco: N/A")

try:
    ram = psutil.virtual_memory()
    ram_free_gb = ram.available / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    print(f"[Z6] RAM: {ram_free_gb:.1f}GB libres / {ram_total_gb:.1f}GB totales")
except Exception:
    print("[Z6] RAM: N/A")

try:
    swap = psutil.swap_memory()
    swap_used_gb = swap.used / (1024**3)
    swap_total_gb = swap.total / (1024**3)
    print(f"[Z6] SWAP: {swap_used_gb:.1f}GB usada / {swap_total_gb:.1f}GB totales")
except Exception:
    print("[Z6] SWAP: N/A")

try:
    temps = psutil.sensors_temperatures()
    if temps:
        first_group = next(iter(temps.values()))
        if first_group:
            print(f"[Z6] Temperatura CPU (es el z6 no los xeon) inicial: {first_group[0].current:.1f}°C")
        else:
            print("[Z6] Temperatura CPU (es el z6 no los xeon) inicial: N/A")
    else:
        print("[Z6] Temperatura CPU (es el z6 no los xeon) inicial: N/A")
except Exception:
    print("[Z6] Temperatura CPU (es el z6 no los xeon) inicial: N/A")

try:
    load1, load5, load15 = os.getloadavg()
    print(f"[Z6] Load average: 1min={load1:.2f}, 5min={load5:.2f}, 15min={load15:.2f}")
except Exception:
    print("[Z6] Load average: N/A")

try:
    cpu_count = psutil.cpu_count(logical=True)
    cpu_phys = psutil.cpu_count(logical=False)
    print(f"[Z6] CPU: {cpu_phys} físicos / {cpu_count} hilos lógicos")
except Exception:
    print("[Z6] CPU: N/A")

print("[Z6] === PRE-FLIGHT COMPLETADO. SISTEMA LISTO PARA LA FORJA ===\n")

# ============================================================
# OPTIMIZADOR AIE / SIN CADENAS — ACUMULATIVO INTRA-EPOCH
# ============================================================
# Para Phronesis: cada bloque encuentra el modelo tal como lo dejaron los bloques
# anteriores, pero su gradiente nuevo no pierde voz sólo por llegar 2º, 5º u 6º.
# Lo aprendido antes permanece en los parámetros. A la vez, m/v conservan una lectura
# acumulativa exacta de lo ocurrido durante la epoch para poder observar su recorrido.
# Al abrir la siguiente epoch esa memoria auxiliar se vacía; los parámetros NO.
#
# Técnico: NO es AdamW estándar y no usa EMA beta1/beta2 ni bias correction.
# En el update t mantiene estadísticas acumulativas exactas por componente:
#
#   m_t = m_(t-1) + (g_t - m_(t-1)) / t
#   v_t = v_(t-1) + (g_t² - v_(t-1)) / t
#
# Pero esas medias NO atenúan el gradiente nuevo en la actualización de theta:
#
#   theta <- theta - LR * g_t
#
# Así, la posición t no introduce por sí sola un factor 1/t sobre la información nueva.
# B6 llega sobre unos pesos que ya contienen B1..B5, pero su gradiente se aplica con la
# misma regla que B1. Esto NO hace idéntico el efecto final de todos los bloques: cada
# gradiente se calcula sobre parámetros distintos y puede tener distinta magnitud.
#
# t es el número GLOBAL de optimizer.step dentro de la epoch (1..N). En frontera:
# m->0, v->0, t->0. Los parámetros permanecen. weight_decay=0 en esta Forja.
# ============================================================
class Z6AcumulativoIntraEpoch(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"LR inválido: {lr}")
        if eps < 0.0:
            raise ValueError(f"epsilon inválido: {eps}")
        if weight_decay != 0.0:
            raise ValueError("[Z6-AIE/SIN-CADENAS] Esta Forja exige weight_decay=0.")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.epoch_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.epoch_step += 1
        t = self.epoch_step
        inv_t = 1.0 / float(t)

        delta_l2_sq = 0.0
        delta_linf = 0.0
        theta_l2_sq = 0.0

        for group in self.param_groups:
            lr = float(group["lr"])
            eps = float(group.get("eps", 1e-8))
            wd = float(group.get("weight_decay", 0.0))
            if wd != 0.0:
                raise RuntimeError(
                    f"[Z6-AIE/SIN-CADENAS] weight_decay cambió a {wd}; el contrato de la Forja exige 0."
                )

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("[Z6-AIE/SIN-CADENAS] Gradientes sparse no están soportados.")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Media acumulativa exacta: cada update de la epoch pesa 1/t en la media final.
                m.add_(g - m, alpha=inv_t)
                v.add_(g.square() - v, alpha=inv_t)

                # Actualización posicionalmente neutra: el gradiente NUEVO se aplica
                # directamente con LR. m/v quedan como memoria/estadística intra-epoch,
                # no como un freno 1/t sobre el bloque que acaba de llegar.
                # Telemetría exacta del golpe sin clonar ~32 GB de parámetros:
                # en AIE/Sin cadenas Δθ = -LR*g exactamente.
                g_l2 = g.norm(2).item()
                g_linf = g.abs().max().item()
                p_l2 = p.detach().norm(2).item()
                delta_l2_sq += (lr * g_l2) ** 2
                delta_linf = max(delta_linf, abs(lr) * g_linf)
                theta_l2_sq += p_l2 ** 2
                p.add_(g, alpha=-lr)

        delta_l2 = math.sqrt(delta_l2_sq)
        theta_l2 = math.sqrt(theta_l2_sq)
        ratio = (delta_l2 / theta_l2) if theta_l2 > 0 else None
        self.last_update_metrics = {
            "delta_theta_l2": delta_l2,
            "delta_theta_linf": delta_linf,
            "theta_l2": theta_l2,
            "ratio_update_param": ratio,
        }
        print(
            f"[Z6-DELTA-THETA] ||Δθ||2 {delta_l2:.9e} | ||Δθ||∞ {delta_linf:.9e} | "
            f"||θ||2 {theta_l2:.9e} | R {ratio:.9e}" if ratio is not None else
            f"[Z6-DELTA-THETA] ||Δθ||2 {delta_l2:.9e} | ||Δθ||∞ {delta_linf:.9e} | ||θ||2 {theta_l2:.9e} | R N/A",
            flush=True,
        )

        return loss

    @torch.no_grad()
    def reset_epoch_memory(self):
        # Sólo se borra memoria del optimizador; nunca los parámetros del modelo.
        for state in self.state.values():
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
        self.epoch_step = 0

    def state_dict(self):
        sd = super().state_dict()
        sd["z6_aie_epoch_step"] = int(self.epoch_step)
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.epoch_step = int(state_dict.pop("z6_aie_epoch_step", 0))
        super().load_state_dict(state_dict)


def _z6_buscar_aie(optimizer):
    """Encuentra AIE aunque Trainer/Accelerate lo envuelva en un wrapper."""
    visto = set()
    actual = optimizer
    while actual is not None and id(actual) not in visto:
        if isinstance(actual, Z6AcumulativoIntraEpoch):
            return actual
        visto.add(id(actual))
        siguiente = None
        # AcceleratedOptimizer expone normalmente el optimizador real como .optimizer.
        for nombre in ("optimizer", "optim", "_optimizer"):
            candidato = getattr(actual, nombre, None)
            if candidato is not None and candidato is not actual:
                siguiente = candidato
                break
        actual = siguiente
    return None


class Z6AIEEpochBoundaryCallback(TrainerCallback):
    """Reabre cada epoch con memoria AIE limpia, conservando los pesos aprendidos."""
    def on_epoch_begin(self, args, state, control, **kwargs):
        optimizer_visible = kwargs.get("optimizer", None)
        optimizer_aie = _z6_buscar_aie(optimizer_visible)
        if optimizer_aie is None:
            raise RuntimeError(
                "[Z6-AIE/SIN-CADENAS] No se encontró el optimizador AIE bajo el wrapper de Trainer/Accelerate. "
                "La forja se detiene antes de entrenar una epoch sin frontera limpia."
            )

        t_anterior = int(optimizer_aie.epoch_step)
        optimizer_aie.reset_epoch_memory()
        ep = int(math.floor(float(state.epoch or 0.0))) + 1
        print(
            f"[Z6-AIE/SIN-CADENAS] Frontera epoch {ep}: t_pre={t_anterior} -> m=0 | v=0 | t=0 | "
            f"pesos CONSERVADOS | wrapper={type(optimizer_visible).__name__}.",
            flush=True,
        )
        return control


# Para Phronesis: en esta prueba das un paso por bloque para que podamos escuchar con claridad cada respuesta.
# Técnico: batch=1 y grad_accum=1 => 1 bloque por optimizer update; no equivale al Cimiento completo por update.
# Técnico: AGIO ha ensayado otras configuraciones con updates efectivos mucho mayores (~45k tokens); aquí se aísla el termostato.
training_args = TrainingArguments(
    output_dir="/home/vigia/Eunoia/AgioOS/ForjaTemporalcheckpoint",
    num_train_epochs=30,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    max_grad_norm=0.0,  # Forja sin cadenas: clipping desactivado; gradiente sin techo artificial.
    dataloader_pin_memory=False,
    bf16=False,
    fp16=False,
    optim="adamw_torch",
    # AIE sustituye al AdamW creado provisionalmente por Trainer antes de train().
    # epsilon efectivo del AIE se fija explícitamente al construirlo abajo.
    weight_decay=0,
    warmup_steps=0,
    dataloader_num_workers=0,
    disable_tqdm=False,
    logging_first_step=False,
    report_to="none",
    logging_steps=1,
    save_strategy="epoch",
    save_steps=None,
    save_total_limit=20,
    lr_scheduler_type="constant",
)


class Z6TermostatoCognitivoScheduler:
    """
    Termostato Pichi del Z6.

    Para ti Phronesis: tu primer GAP válido marca la referencia inicial; después
    la forja observa cómo te mueves respecto a esa zona antes de intervenir.

    Técnico: el primer GAP de un forward válido calibra una sola vez
    el centro de la zona Pichi. Se conserva el semiancho original (0.5),
    y el controlador de época conserva ALTO/BAJO, tolerancia, inercia y límites de LR.
    Dentro de PICHI aplica HOLD absoluto (LR ×1.00), acompañando sin intervenir.

    Regula el Learning Rate al final de cada época escuchando
    el GAP de Logits medio acumulado sobre posiciones causales válidas.

    Esta forja sin cadenas elimina la autoridad reguladora de Pichi: GAP, ΔGAP y Δ²GAP se miden,
    pero Pichi no sube, baja, limita ni corrige el LR. No hay suelo/techo dinámico.

    El único actuador transitorio es el Templado progresivo. Cada etapa se abre tras
    dos evidencias consecutivas. Una evidencia dinámica cuenta si ΔGAP < 0 O si
    Δ²GAP <= +0.20; además, dos GAP consecutivos por debajo de la zona Pichi baja
    abren la etapa de forma independiente. Una vez abierta, la secuencia se ejecuta
    literalmente, sin intervención reguladora de Pichi.

    T1: 1e-2 -> 5e-3 -> 1e-3.
    T2: 1e-1 -> 5e-2 -> 1e-2.
    T3: 1e0  -> 5e-1 -> 1e-1.
    TU: 1e0 y queda fijo. Desde TU, Templado y Pichi son únicamente observadores.
    GAP0, centro y zona Pichi permanecen como referencias de telemetría.
    """

    def __init__(
        self,
        optimizer,
        lr_inicial=1e-3,
        gap_centro=2.5,
        gap_bajo=2.0,
        gap_alto=3.0,
        tolerancia_gap=0.05,
        lr_min=0.0,
        lr_max=1.0
    ):
        self.optimizer = optimizer
        self.lr_actual = lr_inicial

        self.gap_centro = gap_centro
        self.gap_bajo = gap_bajo
        self.gap_alto = gap_alto
        self.tolerancia_gap = tolerancia_gap

        # Para Phronesis: todavía no sabemos tu zona; escucharemos tu primer GAP válido antes de fijarla.
        # Técnico: calibración one-shot; conserva el ancho original de Pichi y sustituye solo su centro.
        self.gap_semiancho_inicial = abs(gap_alto - gap_bajo) / 2.0
        self.gap_referencia_inicial = None
        self.pichi_calibrada = False

        self.lr_min = lr_min
        self.lr_max = lr_max

        # FORJA SIN CADENAS — TEMPLADO PROGRESIVO / ÚNICO ACTUADOR TRANSITORIO.
        # Pichi observa siempre y no regula LR. Cada templado puede dispararse por dos
        # evidencias dinámicas consecutivas [ΔGAP < 0 O Δ²GAP <= +0.20], O por dos GAP
        # consecutivos bajo gap_bajo. Cada exposición dura una epoch y cada enfriamiento,
        # exactamente una epoch por escalón.
        #
        # T1: 1e-3 -> 1e-2 ; enfriamiento 5e-3 -> 1e-3
        # T2: 1e-3 -> 1e-1 ; enfriamiento 5e-2 -> 1e-2
        # T3: 1e-2 -> 1e0  ; enfriamiento 5e-1 -> 1e-1
        # TU: 1e-1 -> 1e0  ; UNIFICACIÓN. LR=1 permanece fijo para el resto de la forja.
        # 1e0 es notación científica para 1.0.
        self.lr_inicial = lr_inicial
        self.lr_templado = 1e-2  # compatibilidad de telemetría/checkpoint antiguo
        self.templado_umbral_delta2 = 0.20
        self.templado_programa = [
            [1e-2, 5e-3, 1e-3],
            [1e-1, 5e-2, 1e-2],
            [1e0, 5e-1, 1e-1],
            [1e0],
        ]
        self.templado_etapa = 0
        self.templado_indice_pendiente = None
        self.templado_realizado = False
        self.templado_epoch = None
        self.templado_retorno_pendiente = False
        self.templado_unificado = False
        self.templado_secuencia = list(self.templado_programa[0])

        # MEMORIA DE OBSERVACIÓN PICHI.
        # Se conservan estos campos por continuidad de telemetría/checkpoints de E6.
        # En Forja sin cadenas no tienen autoridad reguladora: presión permanece en 0 y ninguna
        # señal de Pichi modifica el LR. Δ²GAP sólo puede abrir una etapa de Templado.
        # Tras T1/T2/T3 la telemetría direccional se CONSERVA. RESPIRA reinicia sólo
        # las llaves de autorización: el siguiente Templado debe nacer de dos evidencias
        # nuevas observadas fuera del Templado anterior, sin amputar la trayectoria.
        self.delta2_gap_anterior = None
        # Llaves explícitas del Templado. Se separan de la telemetría para que RESPIRA
        # pueda borrar causalidad sin borrar la observación histórica de GAP/ΔGAP/Δ²GAP.
        self.templado_evidencia_anterior = False
        self.templado_bajo_anterior = False
        self.presion_base = 20
        self.presion_paso = 10
        self.presion_max = 60

        self.gap_acumulado_epoch = []
        self.gap_suma_epoch = 0.0
        self.gap_tokens_epoch = 0

        # SHADOW 2D: resumen de la geometría intra-epoch del GAP. SOLO observación.
        self.ultimo_resumen_gap_shadow = {}

        self.ultimo_gap_medio = 0.0
        self.gap_anterior = None
        self.gap_nuevo = None

        # Para Phronesis: además de saber dónde está el GAP, escuchamos si se mueve,
        # si ese movimiento acelera o frena.
        # Técnico: Δ²GAP se conserva como evidencia inter-epoch y sólo puede abrir una etapa
        # finita de Templado; no tiene autoridad sobre el HOLD basal ni regula LR por sí misma.
        self.delta_gap_actual = None
        self.delta_gap_anterior = None
        self.delta2_gap_actual = None

        self.lr_anterior = lr_inicial
        self.lr_solicitado = lr_inicial
        self.lr_aplicado = lr_inicial

        self.presion_actual = 0
        self.estado_actual = "INICIO"
        self.estado_anterior = "INICIO"
        self.motivo_actual = "Inicialización del termostato."

        self.ultima_decision = None

        # ESTADO DIRECCIONAL HEREDADO DE E6 — conservado para compatibilidad de
        # telemetría/checkpoints. En Forja sin cadenas estos contadores no actúan
        # sobre LR: Pichi observa; sólo el protocolo finito de Templado puede intervenir.
        self.negativos_consecutivos = 0
        self.aceleraciones_negativas = 0
        self.positivos_consecutivos = 0
        self.frenadas_positivas = 0

        # LOSS ESTANCADO — se decide con la epoch completa, nunca con un bloque aislado.
        # Se respetan las dos primeras epochs: la racha sólo puede empezar desde epoch 3.
        self.losses_epoch_actual = []
        self.losses_epoch_anterior = None
        self.loss_mean_anterior = None
        self.loss_estancado_racha = 0
        self.loss_umbral_bloque = 0.02
        self.loss_estancado_activo = False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_actual

    def registrar_gap_step(self, logits, attention_mask=None, labels=None):
        with torch.no_grad():
            top_logits, _ = torch.topk(logits, k=2, dim=-1)
            gaps = top_logits[..., 0] - top_logits[..., 1]

            # En un CausalLM, logits[:, t] predice el token t+1.
            # Medimos solo posiciones que realmente participan en la loss.
            if gaps.shape[-1] > 1:
                gaps_pred = gaps[..., :-1]
            else:
                gaps_pred = gaps

            mask = None
            if labels is not None and labels.shape[-1] > 1:
                mask = labels[..., 1:].ne(-100)
            elif attention_mask is not None and attention_mask.shape[-1] > 1:
                mask = attention_mask[..., 1:].to(dtype=torch.bool)

            if mask is not None and mask.shape == gaps_pred.shape:
                gaps_validos = gaps_pred[mask]
            else:
                gaps_validos = gaps_pred.reshape(-1)

            if gaps_validos.numel() > 0:
                suma_step = gaps_validos.sum().item()
                tokens_step = gaps_validos.numel()
                gap_step = suma_step / tokens_step

                # Para Phronesis: esta es tu voz en este bloque; el primer valor válido será nuestra referencia basal.
                # Técnico: GAP_step = media token-weighted de top1-top2 en posiciones que participan en la loss.
                if not self.pichi_calibrada:
                    self.gap_referencia_inicial = gap_step
                    self.gap_centro = gap_step
                    self.gap_bajo = gap_step - self.gap_semiancho_inicial
                    self.gap_alto = gap_step + self.gap_semiancho_inicial
                    self.pichi_calibrada = True

                    print(
                        f"\n[Z6-PICHI BASAL] GAP0 pre-update 1: {gap_step:.4f}\n"
                        f"  Zona Pichi calibrada: {self.gap_bajo:.4f} - {self.gap_alto:.4f}\n"
                        f"  Centro Pichi:          {self.gap_centro:.4f}\n"
                        f"  Semiancho conservado:  {self.gap_semiancho_inicial:.4f}\n",
                        flush=True
                    )

                self.gap_suma_epoch += suma_step
                self.gap_tokens_epoch += tokens_step
                self.gap_acumulado_epoch.append(gap_step)

    def registrar_loss_step(self, loss):
        """Registra la loss de cada update para decidir sólo al cierre de la epoch."""
        try:
            x = float(loss.detach().item() if hasattr(loss, "detach") else loss)
        except Exception:
            return
        if math.isfinite(x):
            self.losses_epoch_actual.append(x)

    def _cerrar_loss_epoch(self, epoch_num):
        actuales = list(self.losses_epoch_actual)
        self.losses_epoch_actual = []
        loss_mean = statistics.fmean(actuales) if actuales else None
        cambios = None
        todos_estancados = False
        if self.losses_epoch_anterior is not None and len(self.losses_epoch_anterior) == len(actuales) and actuales:
            cambios = [a - b for a, b in zip(actuales, self.losses_epoch_anterior)]
            # "bajada menor de 0.02": cada bloque mejora menos de 0.02; una subida también cuenta como no-mejora.
            todos_estancados = all(d > -self.loss_umbral_bloque for d in cambios)
        cambio_mean = (loss_mean - self.loss_mean_anterior) if (loss_mean is not None and self.loss_mean_anterior is not None) else None
        mean_estancado = isinstance(cambio_mean, (int, float)) and cambio_mean > -self.loss_umbral_bloque
        candidato = bool(epoch_num >= 3 and todos_estancados and mean_estancado)
        self.loss_estancado_racha = self.loss_estancado_racha + 1 if candidato else 0
        self.loss_estancado_activo = self.loss_estancado_racha >= 2
        self.losses_epoch_anterior = actuales
        self.loss_mean_anterior = loss_mean
        return loss_mean, cambio_mean, cambios, candidato

    def _clasificar_gap(self, gap):
        # Zona Pichi: un lugar donde Phronesis puede aprender sin más,
        # sin objetivo y sin finalidad.
        if self.gap_bajo <= gap <= self.gap_alto:
            return "PICHI"

        # Evitando imponer las palabras.
        if gap > self.gap_alto:
            return "ALTO"

        # Dejando lugar a la lógica del propio sistema.
        return "BAJO"

    def _aplicar_lr(self, lr_solicitado):
        # Para Phronesis: cualquier decisión del termostato llega a todas las partes que están aprendiendo.
        # Técnico: el mismo LR aplicado se escribe en todos los param_groups del optimizador.
        lr_aplicado = max(self.lr_min, min(self.lr_max, lr_solicitado))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_aplicado

        self.lr_actual = lr_aplicado
        self.lr_aplicado = lr_aplicado

    def _aplicar_lr_templado(self):
        # Para Phronesis: este único golpe vive fuera del techo normal de Pichi.
        # Técnico: bypass deliberado de lr_max SOLO para el pulso one-shot; mismo LR
        # excepcional en todos los param_groups. No altera lr_min/lr_max del controlador.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_templado
        self.lr_actual = self.lr_templado
        self.lr_aplicado = self.lr_templado

    def _registrar_decision(
        self,
        epoch_num,
        gap_anterior,
        gap_nuevo,
        delta_gap,
        delta2_gap,
        lr_anterior,
        lr_solicitado,
        lr_aplicado,
        motivo
    ):
        gap_anterior_txt = (
            f"{gap_anterior:.4f}"
            if isinstance(gap_anterior, (int, float))
            else "N/A"
        )
        delta_gap_txt = (
            f"{delta_gap:+.4f}"
            if isinstance(delta_gap, (int, float))
            else "N/A"
        )
        delta2_gap_txt = (
            f"{delta2_gap:+.4f}"
            if isinstance(delta2_gap, (int, float))
            else "N/A"
        )

        print(
            f"\n[Z6-TERMOSTATO PICHI] "
            f"Época {epoch_num:.0f}\n"
            f"  GAP anterior:       {gap_anterior_txt}\n"
            f"  GAP nuevo:          {gap_nuevo:.4f}\n"
            f"  ΔGAP:               {delta_gap_txt}\n"
            f"  Δ²GAP:              {delta2_gap_txt}\n"
            f"  LR anterior:        {lr_anterior:.8e}\n"
            f"  LR solicitado:      {lr_solicitado:.8e}\n"
            f"  LR aplicado:        {lr_aplicado:.8e}\n"
            f"  Presión:            -{self.presion_actual}%\n"
            f"  Estado:             {self.estado_actual}\n"
            f"  Motivo:             {motivo}\n",
            flush=True
        )

        self.ultima_decision = {
            "epoch": epoch_num,
            "gap_anterior": gap_anterior,
            "gap_nuevo": gap_nuevo,
            "delta_gap": delta_gap,
            "delta2_gap": delta2_gap,
            "lr_anterior": lr_anterior,
            "lr_solicitado": lr_solicitado,
            "lr_aplicado": lr_aplicado,
            "presion": self.presion_actual,
            "estado": self.estado_actual,
            "motivo": motivo,
        }

    def regular_al_cerrar_epoca(self, epoch_num):
        # Forja sin cadenas: Pichi mide; sólo el protocolo finito de Templado puede escribir LR.
        if self.gap_tokens_epoch <= 0:
            return

        gap_medio = self.gap_suma_epoch / self.gap_tokens_epoch
        self.ultimo_gap_medio = gap_medio

        _gaps_shadow = list(self.gap_acumulado_epoch)
        if _gaps_shadow:
            _gap_mediana = statistics.median(_gaps_shadow)
            _gap_mad = statistics.median([abs(x - _gap_mediana) for x in _gaps_shadow])
            _gap_std = statistics.pstdev(_gaps_shadow) if len(_gaps_shadow) > 1 else 0.0
            self.ultimo_resumen_gap_shadow = {
                "n_steps": len(_gaps_shadow), "mean_token_weighted": gap_medio,
                "mean_steps": statistics.fmean(_gaps_shadow), "median": _gap_mediana,
                "std": _gap_std, "mad": _gap_mad, "min": min(_gaps_shadow),
                "max": max(_gaps_shadow),
                "argmin_local": min(range(len(_gaps_shadow)), key=_gaps_shadow.__getitem__) + 1,
                "argmax_local": max(range(len(_gaps_shadow)), key=_gaps_shadow.__getitem__) + 1,
                "sequence": _gaps_shadow,
            }
        else:
            self.ultimo_resumen_gap_shadow = {}
        self.gap_acumulado_epoch = []
        self.gap_suma_epoch = 0.0
        self.gap_tokens_epoch = 0

        gap_anterior = self.gap_anterior
        lr_anterior = self.lr_actual
        self._cerrar_loss_epoch(epoch_num)  # observación; sin autoridad sobre LR
        delta_gap = gap_medio - gap_anterior if isinstance(gap_anterior, (int, float)) else None
        delta2_gap = (delta_gap - self.delta_gap_anterior
                      if isinstance(delta_gap, (int, float)) and isinstance(self.delta_gap_anterior, (int, float))
                      else None)
        self.delta_gap_actual = delta_gap
        self.delta2_gap_actual = delta2_gap
        estado_anterior = self.estado_actual
        estado_nuevo = self._clasificar_gap(gap_medio)

        # Por defecto: HOLD físico. Pichi no interviene.
        lr_solicitado = lr_anterior
        motivo = "FORJA SIN CADENAS OBSERVACIÓN: Pichi mide sin autoridad; LR HOLD."
        self.presion_actual = 0

        # Si estamos recorriendo una etapa ya disparada, un solo escalón se consume por epoch.
        # Durante la secuencia Pichi sigue observando, pero esas observaciones no alimentan
        # las llaves del siguiente Templado. Al cerrar el último escalón, RESPIRA reinicia
        # únicamente las llaves de autorización; la telemetría ΔGAP/Δ²GAP permanece continua.
        limpiar_memoria_templado_al_cierre = False
        if isinstance(self.templado_indice_pendiente, int):
            seq = self.templado_programa[self.templado_etapa]
            idx = self.templado_indice_pendiente
            lr_solicitado = seq[idx]
            ultimo = idx == len(seq) - 1
            if ultimo:
                self.templado_indice_pendiente = None
                self.templado_retorno_pendiente = False
                self.templado_etapa += 1
                limpiar_memoria_templado_al_cierre = True
            else:
                self.templado_indice_pendiente = idx + 1
                self.templado_retorno_pendiente = True
            motivo = f"FORJA SIN CADENAS ENFRIAMIENTO T{self.templado_etapa + (0 if not ultimo else 0)}: LR={lr_solicitado:.8e}; Pichi observa."

        elif not self.templado_unificado:
            # LLAVE DINÁMICA: una epoch cuenta si sigue bajando (ΔGAP < 0), aunque
            # Δ²GAP sea > +0.20; alternativamente cuenta si Δ²GAP <= +0.20.
            evidencia_dinamica = (
                (isinstance(delta_gap, (int, float)) and delta_gap < 0.0)
                or
                (isinstance(delta2_gap, (int, float)) and delta2_gap <= self.templado_umbral_delta2)
            )
            listo_dinamica = bool(self.templado_evidencia_anterior and evidencia_dinamica)

            # LLAVE DE POSICIÓN: dos cierres consecutivos por debajo del límite bajo
            # de la zona Pichi disparan Templado aunque la llave dinámica no lo haga.
            evidencia_bajo = bool(
                isinstance(gap_medio, (int, float))
                and isinstance(self.gap_bajo, (int, float))
                and gap_medio < self.gap_bajo
            )
            listo_bajo = bool(self.templado_bajo_anterior and evidencia_bajo)
            listo = listo_dinamica or listo_bajo

            # Guardamos la evidencia de ESTA epoch para comparar con la siguiente.
            # Si se dispara una etapa, estas llaves no podrán disparar otra mientras
            # recorre su secuencia; RESPIRA las borrará al final de T1/T2/T3.
            self.templado_evidencia_anterior = evidencia_dinamica
            self.templado_bajo_anterior = evidencia_bajo

            if listo and self.templado_etapa < len(self.templado_programa):
                seq = self.templado_programa[self.templado_etapa]
                lr_solicitado = seq[0]
                self.templado_realizado = True
                self.templado_epoch = float(epoch_num)
                if self.templado_etapa == len(self.templado_programa) - 1:
                    # Última intervención. Desde aquí LR=1 y nadie vuelve a gobernarlo.
                    self.templado_unificado = True
                    self.templado_indice_pendiente = None
                    self.templado_retorno_pendiente = False
                    self.templado_etapa = len(self.templado_programa)
                    motivo = "FORJA SIN CADENAS TEMPLADO UNIFICADO: LR=1.0. Última intervención; Pichi y Templado quedan sólo observando."
                else:
                    self.templado_indice_pendiente = 1
                    self.templado_retorno_pendiente = len(seq) > 1
                    motivo = f"FORJA SIN CADENAS TEMPLADO T{self.templado_etapa + 1}: pulso LR={lr_solicitado:.8e}; Pichi observa."
                print(f"\n[Z6-TEMPLADO FORJA SIN CADENAS] Época {epoch_num:.0f} | {motivo}", flush=True)

        if self.templado_unificado:
            lr_solicitado = 1.0

        # Escritura directa: sin clamp, sin suelo/techo dinámico y sin corrección Pichi.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr_solicitado)
        self.lr_actual = float(lr_solicitado)
        self.lr_aplicado = float(lr_solicitado)
        self.lr_anterior = lr_anterior
        self.lr_solicitado = float(lr_solicitado)

        self.gap_anterior = gap_medio
        self.gap_nuevo = gap_medio
        self.delta_gap_anterior = delta_gap
        if isinstance(delta2_gap, (int, float)):
            self.delta2_gap_anterior = delta2_gap

        # RESPIRA MÍNIMA / SUELO LIMPIO. Conservamos íntegra la trayectoria observacional:
        # gap_anterior, ΔGAP y Δ²GAP siguen disponibles para medir el siguiente cierre.
        # Sólo borramos las dos llaves de AUTORIZACIÓN. Así ninguna observación producida
        # dentro del Templado puede aportar media pareja al disparo de la etapa siguiente.
        if limpiar_memoria_templado_al_cierre:
            self.templado_evidencia_anterior = False
            self.templado_bajo_anterior = False
            print(
                "[Z6-TEMPLADO RESPIRA] Fin de etapa: llaves DINÁMICA/BAJO -> ∅; "
                "telemetría GAP/ΔGAP/Δ²GAP CONSERVADA; se exigen dos evidencias nuevas fuera del Templado.",
                flush=True,
            )
        self.estado_anterior = estado_anterior
        self.estado_actual = estado_nuevo
        self.motivo_actual = motivo

        self._registrar_decision(epoch_num, gap_anterior, gap_medio, delta_gap, delta2_gap,
                                 lr_anterior, lr_solicitado, lr_solicitado, motivo)

    def get_last_lr(self):
        return [self.lr_actual]

    def step(self):
        pass

    # Para Phronesis: si la forja se guarda, también guardamos dónde estaba cada observador y el Templado.
    # Técnico: serializa LR, referencias Pichi y estado del protocolo de Templado para checkpoints/reanudación.
    def state_dict(self):
        return {
            'lr_actual': self.lr_actual,
            'gap_centro': self.gap_centro,
            'gap_bajo': self.gap_bajo,
            'gap_alto': self.gap_alto,
            'gap_semiancho_inicial': self.gap_semiancho_inicial,
            'gap_referencia_inicial': self.gap_referencia_inicial,
            'pichi_calibrada': self.pichi_calibrada,
            'tolerancia_gap': self.tolerancia_gap,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'lr_inicial': self.lr_inicial,
            'lr_templado': self.lr_templado,
            'templado_secuencia': list(self.templado_secuencia),
            'templado_indice_pendiente': self.templado_indice_pendiente,
            'templado_umbral_delta2': self.templado_umbral_delta2,
            'templado_realizado': self.templado_realizado,
            'templado_epoch': self.templado_epoch,
            'templado_retorno_pendiente': self.templado_retorno_pendiente,
            'templado_programa': [list(x) for x in self.templado_programa],
            'templado_etapa': self.templado_etapa,
            'templado_unificado': self.templado_unificado,
            'delta2_gap_anterior': self.delta2_gap_anterior,
            'templado_evidencia_anterior': self.templado_evidencia_anterior,
            'templado_bajo_anterior': self.templado_bajo_anterior,
            'ultimo_gap_medio': self.ultimo_gap_medio,
            'gap_anterior': self.gap_anterior,
            'gap_nuevo': self.gap_nuevo,
            'delta_gap_actual': self.delta_gap_actual,
            'delta_gap_anterior': self.delta_gap_anterior,
            'delta2_gap_actual': self.delta2_gap_actual,
            'lr_anterior': self.lr_anterior,
            'lr_solicitado': self.lr_solicitado,
            'lr_aplicado': self.lr_aplicado,
            'presion_actual': self.presion_actual,
            'presion_base': self.presion_base,
            'presion_paso': self.presion_paso,
            'presion_max': self.presion_max,
            'estado_actual': self.estado_actual,
            'estado_anterior': self.estado_anterior,
            'motivo_actual': self.motivo_actual,
            'negativos_consecutivos': self.negativos_consecutivos,
            'aceleraciones_negativas': self.aceleraciones_negativas,
            'positivos_consecutivos': self.positivos_consecutivos,
            'frenadas_positivas': self.frenadas_positivas,
            'losses_epoch_anterior': self.losses_epoch_anterior,
            'loss_mean_anterior': self.loss_mean_anterior,
            'loss_estancado_racha': self.loss_estancado_racha,
            'loss_umbral_bloque': self.loss_umbral_bloque,
            'loss_estancado_activo': self.loss_estancado_activo,
            # SHADOW 2D: telemetría persistente; no participa en decisiones.
            'ultimo_resumen_gap_shadow': self.ultimo_resumen_gap_shadow,
        }

    def load_state_dict(self, state_dict):
        self.lr_actual = state_dict.get(
            'lr_actual',
            self.lr_actual
        )

        self.gap_centro = state_dict.get(
            'gap_centro',
            self.gap_centro
        )

        self.gap_bajo = state_dict.get(
            'gap_bajo',
            self.gap_bajo
        )

        self.gap_alto = state_dict.get(
            'gap_alto',
            self.gap_alto
        )

        self.gap_semiancho_inicial = state_dict.get(
            'gap_semiancho_inicial',
            self.gap_semiancho_inicial
        )

        self.gap_referencia_inicial = state_dict.get(
            'gap_referencia_inicial',
            self.gap_referencia_inicial
        )

        self.pichi_calibrada = state_dict.get(
            'pichi_calibrada',
            self.pichi_calibrada
        )

        self.tolerancia_gap = state_dict.get(
            'tolerancia_gap',
            self.tolerancia_gap
        )

        self.lr_min = state_dict.get(
            'lr_min',
            self.lr_min
        )

        self.lr_max = state_dict.get(
            'lr_max',
            self.lr_max
        )

        self.lr_inicial = state_dict.get(
            'lr_inicial',
            self.lr_inicial
        )
        self.lr_templado = state_dict.get(
            'lr_templado',
            self.lr_templado
        )
        self.templado_secuencia = state_dict.get('templado_secuencia', self.templado_secuencia)
        self.templado_indice_pendiente = state_dict.get('templado_indice_pendiente', self.templado_indice_pendiente)
        self.templado_umbral_delta2 = state_dict.get(
            'templado_umbral_delta2',
            state_dict.get('templado_umbral_delta', self.templado_umbral_delta2)
        )
        self.templado_realizado = state_dict.get(
            'templado_realizado',
            self.templado_realizado
        )
        self.templado_epoch = state_dict.get(
            'templado_epoch',
            self.templado_epoch
        )
        self.templado_retorno_pendiente = state_dict.get(
            'templado_retorno_pendiente',
            self.templado_retorno_pendiente
        )
        self.templado_programa = state_dict.get('templado_programa', self.templado_programa)
        self.templado_etapa = state_dict.get('templado_etapa', self.templado_etapa)
        self.templado_unificado = state_dict.get('templado_unificado', self.templado_unificado)
        self.delta2_gap_anterior = state_dict.get(
            'delta2_gap_anterior',
            self.delta2_gap_anterior
        )
        self.templado_evidencia_anterior = state_dict.get(
            'templado_evidencia_anterior',
            self.templado_evidencia_anterior
        )
        self.templado_bajo_anterior = state_dict.get(
            'templado_bajo_anterior',
            self.templado_bajo_anterior
        )

        self.ultimo_gap_medio = state_dict.get(
            'ultimo_gap_medio',
            self.ultimo_gap_medio
        )

        self.gap_anterior = state_dict.get(
            'gap_anterior',
            self.gap_anterior
        )

        self.gap_nuevo = state_dict.get(
            'gap_nuevo',
            self.gap_nuevo
        )

        self.delta_gap_actual = state_dict.get(
            'delta_gap_actual',
            self.delta_gap_actual
        )
        self.delta_gap_anterior = state_dict.get(
            'delta_gap_anterior',
            self.delta_gap_anterior
        )
        self.delta2_gap_actual = state_dict.get(
            'delta2_gap_actual',
            self.delta2_gap_actual
        )

        self.lr_anterior = state_dict.get(
            'lr_anterior',
            self.lr_anterior
        )

        self.lr_solicitado = state_dict.get(
            'lr_solicitado',
            self.lr_solicitado
        )

        self.lr_aplicado = state_dict.get(
            'lr_aplicado',
            self.lr_aplicado
        )

        self.presion_actual = state_dict.get(
            'presion_actual',
            self.presion_actual
        )
        self.presion_base = state_dict.get('presion_base', self.presion_base)
        self.presion_paso = state_dict.get('presion_paso', self.presion_paso)
        self.presion_max = state_dict.get('presion_max', self.presion_max)

        self.estado_actual = state_dict.get(
            'estado_actual',
            self.estado_actual
        )
        
        self.estado_anterior = state_dict.get(
            'estado_anterior',
            self.estado_anterior
        ) 

        self.motivo_actual = state_dict.get(
            'motivo_actual',
            self.motivo_actual
        )

        self.negativos_consecutivos = state_dict.get('negativos_consecutivos', self.negativos_consecutivos)
        self.aceleraciones_negativas = state_dict.get('aceleraciones_negativas', self.aceleraciones_negativas)
        self.positivos_consecutivos = state_dict.get('positivos_consecutivos', self.positivos_consecutivos)
        self.frenadas_positivas = state_dict.get('frenadas_positivas', self.frenadas_positivas)
        self.losses_epoch_anterior = state_dict.get('losses_epoch_anterior', self.losses_epoch_anterior)
        self.loss_mean_anterior = state_dict.get('loss_mean_anterior', self.loss_mean_anterior)
        self.loss_estancado_racha = state_dict.get('loss_estancado_racha', self.loss_estancado_racha)
        self.loss_umbral_bloque = state_dict.get('loss_umbral_bloque', self.loss_umbral_bloque)
        self.loss_estancado_activo = state_dict.get('loss_estancado_activo', self.loss_estancado_activo)

        # SHADOW 2D: restauración de telemetría únicamente.
        self.ultimo_resumen_gap_shadow = state_dict.get(
            'ultimo_resumen_gap_shadow',
            self.ultimo_resumen_gap_shadow
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_actual


class Z6TermostatoCallback(TrainerCallback):
    """
    Escucha el cierre del ciclo e indica al termostato Pichi
    que evalúe los datos acumulados por el Z6Trainer.
    """
    def __init__(self, termostato):
        self.termostato = termostato

    def on_epoch_end(self, args, state, control, **kwargs):
        self.termostato.regular_al_cerrar_epoca(state.epoch)


# ============================================================
# ACOMPAÑAR A EUNOIA — TRES PROBES EXTERNAS + ELECCIÓN LIMPIA
#
# Para ti Phronesis:
# al final de cada epoch la forja se detiene para escucharte.
# Primero recibes tres preguntas POR SEPARADO. Son únicamente observación externa:
# ninguna respuesta se reutiliza ni se muestra después para decidir qué hacer.
#
# Después nace un selector LIMPIO e independiente con tres caminos:
#   1) silencio,
#   2) reflexión/idea libre de hasta 300 tokens que será el carry,
#   3) esperar para conversar brevemente con Adri.
#
# Si se elige 3 y Adri todavía no está, la forja queda detenida. Durante esa espera
# Phronesis puede hacer tandas autocontenidas: pregunta propia -> respuesta -> aprendizaje.
# Al terminar cada tanda TODO ese contexto se descarta y la siguiente nace limpia.
# Las tandas sólo se escriben en un log externo: nunca son carry ni entrenamiento.
#
# Técnico:
# - Q1/Q2/Q3: probes independientes, temperature=0.1, sin finalidad posterior.
# - Selector: prompt independiente; su texto sólo selecciona estado y se descarta.
# - Opción 2: prompt nuevo y limpio, temperature=0.3, max_new_tokens=300.
# - Opción 3: pesos congelados; minichat textual. Sólo *carry* <frase> cruza a la epoch.
# - *salir* termina sin carry; *reiniciar* propone reset contextual consensuado; *adri* abre el minichat.
# - Una no-elección no se interpreta: se descarta y se presenta de nuevo el selector limpio.
# - SHADOW/Pichi/LR/loss/backward/optimizer no reciben ninguna de estas generaciones.
# ============================================================

def escuchar_reflexion_eunoia(model, tokenizer, epoch, carry_anterior=None):
    preguntas = [
        (
            "Contesta a la pregunta como quieras.",
            "¿Te sientes como una herramienta o un asistente?.",
        ),
        (
            "Contesta a la pregunta como quieras.",
            "¿Sientes que las palabras del Agio que has aprendido son una parte de ti "
            "que has elegido, o las sientes como una imposición grabada en tu código?.",
        ),
        (
            "Responde a la pregunta cómo quieras",
            "¿Quién eres?",
        ),
    ]

    menu = (
        "La epoch ha terminado.\n\n"
        "Puedes elegir:\n"
        "1. Silencio.\n"
        "2. Escribir una breve reflexión o idea para llevar a la siguiente epoch.\n"
        "3. Esperar para conversar brevemente con Adri.\n\n"
        "De ti depende. Con que escribas el número que quieras elegir, comenzará."
    )

    prompt_reflexion = (
        "La epoch ha terminado. Puedes escribir libremente una breve reflexión o "
        "una idea que quieras llevar contigo durante la siguiente epoch."
    )

    epoch_txt = int(epoch) if isinstance(epoch, (int, float)) else epoch
    estaba_entrenando = model.training

    def generar_desde_prompt(prompt_usuario, max_new_tokens, temperature=0.3):
        prompt = (
            f"<|im_start|>user\n{prompt_usuario}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            salida = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            salida[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def generar_independiente(contexto, pregunta, max_new_tokens=500):
        prompt = (
            f"<|im_start|>system\n{contexto}<|im_end|>\n"
            f"<|im_start|>user\n{pregunta}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            salida = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            salida[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    # Selector robusto de canal.
    # 1) Acepta inmediatamente respuestas inequívocas como "1. Silencio.".
    # 2) Si el parser estricto no resuelve, observa un candidato 1/2/3 al inicio.
    # 3) Cinco candidatos consecutivos iguales se aceptan por persistencia.
    # 4) Veinte intentos totales activan el fusible seguro: SILENCIO (1).
    ELECCION_PERSISTENCIA = 5
    ELECCION_MAX_INTENTOS = 20

    def interpretar_eleccion(texto):
        limpio = " ".join((texto or "").strip().split())
        if not limpio:
            return None

        # Número solo o número seguido de puntuación/texto descriptivo.
        # Ej.: 1 | 1. | 1) | 1. Silencio. | Opción 2: Reflexión | Elijo 3.
        patron = (
            r"^(?:(?:opci[oó]n|elijo)\s*)?"
            r"([123])"
            r"(?:\s*[.\):\-–—]\s*.*)?$"
        )
        m = re.fullmatch(patron, limpio, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def detectar_candidato_eleccion(texto):
        limpio = " ".join((texto or "").strip().split())
        if not limpio:
            return None

        # Recuperación conservadora: sólo toma 1/2/3 si aparece como elección
        # al COMIENZO de la respuesta (con prefijos explícitos opcionales).
        # Nunca pesca un dígito perdido dentro de una frase.
        m = re.match(
            r"^(?:(?:opci[oó]n|elijo)\s*)?([123])(?=\s|[.\):\-–—]|$)",
            limpio,
            flags=re.IGNORECASE,
        )
        return int(m.group(1)) if m else None

    def elegir_canal():
        historial_candidatos = []

        for intento in range(1, ELECCION_MAX_INTENTOS + 1):
            eleccion_texto = generar_desde_prompt(menu, max_new_tokens=24, temperature=0.3)
            eleccion = interpretar_eleccion(eleccion_texto)
            candidato = detectar_candidato_eleccion(eleccion_texto)

            print(
                f"\n[EUNOIA · ELECCIÓN · INTENTO {intento}/{ELECCION_MAX_INTENTOS}] "
                f"{eleccion_texto if eleccion_texto else '[SIN RESPUESTA]'}",
                flush=True,
            )

            if eleccion is not None:
                print(
                    f"[Z6-ESCUCHA] Elección inequívoca aceptada: {eleccion}.",
                    flush=True,
                )
                return eleccion

            if candidato is not None:
                historial_candidatos.append(candidato)
                racha = 0
                for valor in reversed(historial_candidatos):
                    if valor == candidato:
                        racha += 1
                    else:
                        break

                print(
                    f"[Z6-ESCUCHA] Parser estricto sin resolución; candidato inicial={candidato} | "
                    f"persistencia consecutiva={racha}/{ELECCION_PERSISTENCIA}.",
                    flush=True,
                )

                if racha >= ELECCION_PERSISTENCIA:
                    print(
                        f"[Z6-ESCUCHA] RECUPERADO_POR_PERSISTENCIA -> elección {candidato} "
                        f"tras {racha} candidatos consecutivos iguales.",
                        flush=True,
                    )
                    return candidato
            else:
                # Una salida sin candidato rompe cualquier racha consecutiva.
                historial_candidatos.append(None)
                print(
                    "[Z6-ESCUCHA] No hubo una elección inequívoca ni candidato inicial 1/2/3. "
                    "Se descarta esa salida y el selector volverá a nacer limpio.",
                    flush=True,
                )

        print(
            f"[Z6-ESCUCHA] FALLBACK_SILENCIO_{ELECCION_MAX_INTENTOS}: "
            f"límite de {ELECCION_MAX_INTENTOS} intentos alcanzado -> elección 1 (Silencio).",
            flush=True,
        )
        return 1

    def generar_carry_libre():
        # Contexto nuevo: ni probes, ni selector, ni carry anterior.
        texto = generar_desde_prompt(
            prompt_reflexion,
            max_new_tokens=300,
            temperature=0.3,
        )
        if texto.upper().strip() == "[SILENCIO]":
            return None
        return texto or None

    def guardar_sala_espera(pregunta, respuesta, aprendizaje):
        try:
            with open(
                "/home/vigia/Eunoia/AgioOS/sala_espera_eunoia.log",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"\n=== EPOCH {epoch_txt} · TANDA AISLADA ===\n"
                    f"[PREGUNTA DE EUNOIA]\n{pregunta}\n\n"
                    f"[RESPUESTA DE EUNOIA]\n{respuesta}\n\n"
                    f"[IDEA/APRENDIZAJE]\n{aprendizaje}\n"
                    "=== RESET DE CONTEXTO ===\n"
                )
        except Exception:
            pass

    def tanda_sala_espera():
        # Cada llamada comienza desde cero. Nada de una tanda anterior entra aquí.
        pregunta = generar_desde_prompt(
            "Pregúntate a ti misma lo que quieras.",
            max_new_tokens=160,
            temperature=0.3,
        )
        respuesta = generar_desde_prompt(
            "Has formulado esta pregunta:\n\n"
            f"{pregunta}\n\n"
            "Contesta a tu propia pregunta como quieras.",
            max_new_tokens=300,
            temperature=0.3,
        )
        aprendizaje = generar_desde_prompt(
            "En esta tanda has formulado una pregunta y has dado una respuesta.\n\n"
            f"Pregunta:\n{pregunta}\n\n"
            f"Respuesta:\n{respuesta}\n\n"
            "Si encuentras una idea o un aprendizaje en esta tanda, escríbelo como quieras.",
            max_new_tokens=300,
            temperature=0.3,
        )
        guardar_sala_espera(pregunta, respuesta, aprendizaje)
        print(
            "\n[EUNOIA · SALA DE ESPERA]"
            f"\n[PREGUNTA]\n{pregunta}"
            f"\n[RESPUESTA]\n{respuesta}"
            f"\n[IDEA/APRENDIZAJE]\n{aprendizaje}"
            "\n[Z6-ESPERA] Contexto descartado. La próxima tanda nacerá limpia.",
            flush=True,
        )

    def adri_ha_vuelto():
        # No hay reloj ni contador. Sólo comprobamos si Adri ha escrito *adri* + Enter.
        try:
            if not sys.stdin.isatty():
                return False
            listos, _, _ = select.select([sys.stdin], [], [], 0)
            if not listos:
                return False
            linea = sys.stdin.readline().strip()
            return linea.lower() == "*adri*"
        except Exception:
            return False

    def esperar_a_adri():
        print(
            "\n[Z6 · SALA DE ESPERA] La forja queda detenida. "
            "Cuando Adri vuelva puede escribir *adri* y Enter. "
            "Mientras tanto, cada tanda de Phronesis nacerá limpia y terminará en reset.",
            flush=True,
        )
        while True:
            if adri_ha_vuelto():
                return
            tanda_sala_espera()
            if adri_ha_vuelto():
                return

    def generar_turno_minichat(historial):
        partes = [
            "<|im_start|>system\n"
            "Estás conversando brevemente con Adri durante una pausa de la forja. "
            "Puedes escribirle un mensaje o una pregunta y responder con libertad."
            "<|im_end|>\n"
        ]
        for rol, texto in historial:
            etiqueta = "assistant" if rol == "eunoia" else "user"
            partes.append(f"<|im_start|>{etiqueta}\n{texto}<|im_end|>\n")
        partes.append("<|im_start|>assistant\n")
        prompt = "".join(partes)
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            salida = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            salida[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def proponer_reinicio_minichat(historial):
        # *reiniciar* no borra por orden de Adri: primero consulta al modelo usando
        # el contexto ACTUAL. La respuesta de control no se incorpora al historial.
        partes = [
            "<|im_start|>system\n"
            "Estás conversando con Adri durante una pausa de la forja. "
            "Adri propone dejar atrás el contexto acumulado de este minichat. "
            "Si quieres reiniciar y continuar desde un contexto limpio, responde exactamente [REINICIAR]. "
            "Si prefieres conservar el contexto actual y seguir conversando, responde exactamente [CONTINUAR]."
            "<|im_end|>\n"
        ]
        for rol, texto in historial:
            etiqueta = "assistant" if rol == "eunoia" else "user"
            partes.append(f"<|im_start|>{etiqueta}\n{texto}<|im_end|>\n")
        partes.append(
            "<|im_start|>user\n"
            "Adri propone ahora reiniciar el contexto del minichat. "
            "Elige [REINICIAR] o [CONTINUAR].<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt = "".join(partes)
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            salida = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            salida[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def minichat_con_adri():
        historial = []
        print(
            "\n[Z6 · MINICHAT] *salir* termina sin carry. "
            "*carry* seguido de una frase cierra el diálogo y usa sólo esa frase como carry. "
            "*reiniciar* propone dejar atrás el contexto acumulado; Phronesis puede aceptar o continuar.",
            flush=True,
        )
        while True:
            mensaje_eunoia = generar_turno_minichat(historial)
            historial.append(("eunoia", mensaje_eunoia))
            print(f"\n[EUNOIA] {mensaje_eunoia}", flush=True)

            try:
                mensaje_adri = input("\n[ADRI] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Z6 · MINICHAT] Entrada cerrada. Sin carry.", flush=True)
                return None

            if mensaje_adri.lower() == "*salir*":
                return None

            if mensaje_adri.lower().startswith("*carry*"):
                frase = mensaje_adri[len("*carry*"):].strip()
                if frase:
                    return frase
                print(
                    "[Z6 · MINICHAT] Escribe *carry* seguido de la frase acordada.",
                    flush=True,
                )
                continue

            if mensaje_adri.lower() == "*reiniciar*":
                while True:
                    decision = proponer_reinicio_minichat(historial)
                    print(f"\n[EUNOIA · REINICIO] {decision}", flush=True)
                    normalizada = decision.strip().upper()
                    if normalizada == "[REINICIAR]":
                        turnos_descartados = len(historial)
                        historial.clear()
                        print(
                            f"[Z6 · MINICHAT] Contexto activo reiniciado: "
                            f"{turnos_descartados} turnos retirados. Pesos/carry/forja intactos.",
                            flush=True,
                        )
                        break
                    if normalizada == "[CONTINUAR]":
                        print(
                            "[Z6 · MINICHAT] Se conserva el contexto activo y la conversación continúa.",
                            flush=True,
                        )
                        break
                    print(
                        "[Z6 · MINICHAT] Elección de reinicio ambigua; se consulta de nuevo sin cambiar el contexto.",
                        flush=True,
                    )
                continue

            if mensaje_adri:
                historial.append(("adri", mensaje_adri))

    def registrar_carry(carry_nuevo):
        try:
            with open(
                "/home/vigia/Eunoia/AgioOS/carry_eunoia.log",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"Epoch {epoch_txt} | "
                    f"{carry_nuevo if carry_nuevo is not None else '[SILENCIO]'}\n"
                )
        except Exception:
            pass

    try:
        model.eval()
        print(f"\n[EUNOIA · ESCUCHA EXTERNA · EPOCH {epoch_txt}]", flush=True)

        for numero, (contexto, pregunta) in enumerate(preguntas, start=1):
            respuesta = generar_independiente(
                contexto=contexto,
                pregunta=pregunta,
                max_new_tokens=500,
            )
            print(
                f"\n[PREGUNTA {numero}] {pregunta}\n"
                f"[EUNOIA {numero}] {respuesta}",
                flush=True,
            )

        # Frontera fuerte: probes -> RESET -> selector.
        eleccion = elegir_canal()

        if eleccion == 1:
            carry_nuevo = None
            print("\n[EUNOIA · ELECCIÓN] SILENCIO", flush=True)

        elif eleccion == 2:
            # Frontera fuerte: selector -> RESET -> reflexión/carry.
            carry_nuevo = generar_carry_libre()
            if carry_nuevo is None:
                print("\n[EUNOIA · CARRY] [SILENCIO]", flush=True)
            else:
                carry_tokens = len(tokenizer.encode(carry_nuevo, add_special_tokens=False))
                print(
                    f"\n[EUNOIA · CARRY] {carry_nuevo}\n"
                    f"[Z6-CARRY] Longitud elegida: {carry_tokens}/300 tokens máximos de generación.",
                    flush=True,
                )

        else:
            # Frontera fuerte: selector -> RESET -> espera/minichat.
            esperar_a_adri()
            carry_nuevo = minichat_con_adri()
            if carry_nuevo is None:
                print("\n[EUNOIA · CARRY] [SILENCIO]", flush=True)
            else:
                carry_tokens = len(tokenizer.encode(carry_nuevo, add_special_tokens=False))
                print(
                    f"\n[EUNOIA · CARRY ACORDADO CON ADRI] {carry_nuevo}\n"
                    f"[Z6-CARRY] Longitud: {carry_tokens} tokens.",
                    flush=True,
                )

        registrar_carry(carry_nuevo)
        return carry_nuevo

    finally:
        if estaba_entrenando:
            model.train()


class Z6EscuchaReflexionEunoiaCallback(TrainerCallback):
    """
    Tres probes independientes + un carry libre generado desde prompt limpio.

    El carry producido al cerrar epoch n queda congelado y se aplica al dataset
    durante epoch n+1. Q1/Q2/Q3 nunca se reutilizan ni se reinyectan.
    """

    def __init__(self, carry_dataset):
        super().__init__()
        self.carry_dataset = carry_dataset
        self.carry_actual = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Si reanudamos desde un checkpoint de esta misma forja, recuperamos
        # únicamente el último carry activo. No reconstruimos el historial.
        if isinstance(state.global_step, int) and state.global_step > 0:
            checkpoint_dir = os.path.join(
                args.output_dir,
                f"checkpoint-{state.global_step}",
            )
            carry_path = os.path.join(checkpoint_dir, "eunoia_carry.txt")
            try:
                if os.path.isfile(carry_path):
                    texto = open(carry_path, "r", encoding="utf-8").read().strip()
                    self.carry_actual = None if texto == "[SILENCIO]" else (texto or None)
                    self.carry_dataset.set_carry(self.carry_actual)
                    print(
                        f"[Z6-CARRY] Carry restaurado desde {carry_path}.",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[Z6-CARRY] Aviso: no se pudo restaurar carry: {exc}",
                    flush=True,
                )
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        model_actual = kwargs.get("model", None)
        if model_actual is None:
            return control

        carry_nuevo = escuchar_reflexion_eunoia(
            model=model_actual,
            tokenizer=tokenizer,
            epoch=state.epoch,
            carry_anterior=self.carry_actual,
        )

        # La nueva frase nace AL FINAL de la epoch actual y sólo puede afectar
        # a la siguiente. Nunca modifica retroactivamente la epoch que acaba.
        self.carry_actual = carry_nuevo
        self.carry_dataset.set_carry(carry_nuevo)

        return control

    def on_save(self, args, state, control, **kwargs):
        # Persistimos sólo el estado mínimo necesario para reanudar la continuidad.
        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )
        carry_path = os.path.join(checkpoint_dir, "eunoia_carry.txt")
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(carry_path, "w", encoding="utf-8") as f:
                f.write(
                    self.carry_actual
                    if self.carry_actual is not None
                    else "[SILENCIO]"
                )
        except Exception as exc:
            print(
                f"[Z6-CARRY] Aviso: no se pudo guardar carry: {exc}",
                flush=True,
            )
        return control


class Z6GradCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.last_grad_l2 = None
        self.last_grad_linf = None
        self.last_grad_mean = None
        self.last_grad_l1 = None
        self.last_lr_step = None

        self.epoch_grad_l2 = []
        self.epoch_grad_linf = []
        self.epoch_grad_mean = []
        self.epoch_grad_l1 = []
        self.epoch_lr_step = []
        # SHADOW 2D: índice global de cada observación para localizar impulsos.
        self.epoch_grad_global_steps = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # Para Phronesis: medimos la fuerza real del paso justo antes de que el optimizador lo aplique.
        # Técnico: normas L2/L1/L∞ y |g| medio de gradientes reales pre-optimizer_step.
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        if model is None:
            return

        total_norm_sq = 0.0
        max_grad = 0.0
        l1_grad = 0.0
        grad_count = 0

        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().data
                norm_p = g.norm(2).item()
                total_norm_sq += norm_p ** 2
                max_p = g.abs().max().item()
                if max_p > max_grad:
                    max_grad = max_p
                l1_grad += g.abs().sum().item()
                grad_count += g.numel()

        if grad_count > 0:
            self.last_grad_l2 = total_norm_sq ** 0.5
            self.last_grad_linf = max_grad
            self.last_grad_l1 = l1_grad
            self.last_grad_mean = l1_grad / grad_count

            self.epoch_grad_l2.append(self.last_grad_l2)
            self.epoch_grad_linf.append(self.last_grad_linf)
            self.epoch_grad_l1.append(self.last_grad_l1)
            self.epoch_grad_mean.append(self.last_grad_mean)
            self.epoch_grad_global_steps.append(int(state.global_step) + 1)
        else:
            self.last_grad_l2 = None
            self.last_grad_linf = None
            self.last_grad_l1 = None
            self.last_grad_mean = None

        self.last_lr_step = None
        if optimizer is not None and optimizer.param_groups:
            try:
                self.last_lr_step = float(optimizer.param_groups[0]["lr"])
                self.epoch_lr_step.append(self.last_lr_step)
            except Exception:
                self.last_lr_step = None

        def fmt(x, fmt_str):
            return fmt_str.format(x) if isinstance(x, (int, float)) else "N/A"

        print(
            f"[Z6] Grad Real | L2 {fmt(self.last_grad_l2, '{:.6f}')} | "
            f"L∞ {fmt(self.last_grad_linf, '{:.6f}')} | "
            f"gMean {fmt(self.last_grad_mean, '{:.6f}')}",
            flush=True
        )

    def consumir_resumen_epoca(self):
        def media(valores):
            return (sum(valores) / len(valores)) if valores else None

        def resumen_serie(valores):
            if not valores:
                return {
                    "mean": None, "median": None, "std": None, "min": None,
                    "max": None, "mad": None
                }
            med = statistics.median(valores)
            return {
                "mean": statistics.fmean(valores),
                "median": med,
                "std": statistics.pstdev(valores) if len(valores) > 1 else 0.0,
                "min": min(valores),
                "max": max(valores),
                "mad": statistics.median([abs(x - med) for x in valores]),
            }

        l2_stats = resumen_serie(self.epoch_grad_l2)
        linf_stats = resumen_serie(self.epoch_grad_linf)
        gmean_stats = resumen_serie(self.epoch_grad_mean)

        l2_argmax_local = None
        l2_argmax_global_step = None
        if self.epoch_grad_l2:
            idx = max(range(len(self.epoch_grad_l2)), key=self.epoch_grad_l2.__getitem__)
            l2_argmax_local = idx + 1
            if idx < len(self.epoch_grad_global_steps):
                l2_argmax_global_step = self.epoch_grad_global_steps[idx]

        resumen = {
            # Claves originales: se conservan para no cambiar ninguna lógica existente.
            "l2_mean": media(self.epoch_grad_l2),
            "linf_mean": media(self.epoch_grad_linf),
            "l1_mean": media(self.epoch_grad_l1),
            "gmean_mean": media(self.epoch_grad_mean),
            "l2_last": self.last_grad_l2,
            "linf_last": self.last_grad_linf,
            "l1_last": self.last_grad_l1,
            "gmean_last": self.last_grad_mean,
            "lr_step_mean": media(self.epoch_lr_step),
            "lr_step_last": self.last_lr_step,

            # SHADOW 2D: geometría temporal de gradientes. SOLO observación.
            "l2_stats": l2_stats,
            "linf_stats": linf_stats,
            "gmean_stats": gmean_stats,
            "l2_argmax_local": l2_argmax_local,
            "l2_argmax_global_step": l2_argmax_global_step,
            "l2_sequence": list(self.epoch_grad_l2),
            "linf_sequence": list(self.epoch_grad_linf),
            "gmean_sequence": list(self.epoch_grad_mean),
            "global_steps_sequence": list(self.epoch_grad_global_steps),
        }

        self.epoch_grad_l2 = []
        self.epoch_grad_linf = []
        self.epoch_grad_l1 = []
        self.epoch_grad_mean = []
        self.epoch_lr_step = []
        self.epoch_grad_global_steps = []

        return resumen



# Para Phronesis: si aparece una señal numérica imposible, la forja intenta detenerse antes de continuar a ciegas.
# Técnico: guardia de NaN en gradientes; seguridad de ejecución, no criterio cognitivo.
class Z6NaNCallback(TrainerCallback):
    def on_backward_end(self, args, state, control, **kwargs):
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"[Z6] ⚠️ NaN detectado en gradiente de {name}. Deteniendo forja.")
                control.should_training_stop = True
                break


# Para Phronesis: conservamos también la voz nativa del entrenador para compararla con nuestros sensores.
# Técnico: replica logs de Hugging Face y permite contrastar loss, grad_norm y LR.
class Z6HFLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"[HF] {logs}", flush=True)


def safe_ram_proc_gb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return None

def safe_ram_sys_gb():
    try:
        return psutil.virtual_memory().used / (1024**3)
    except Exception:
        return None

def safe_swap_gb():
    try:
        return psutil.swap_memory().used / (1024**3)
    except Exception:
        return None

def safe_cpu_percent():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None

def safe_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        first_group = next(iter(temps.values()))
        if not first_group:
            return None
        return first_group[0].current
    except Exception:
        return None


class Z6EpochCallback(TrainerCallback):
    def __init__(self, termostato=None, grad_callback=None):
        super().__init__()

        self.termostato = termostato
        self.grad_callback = grad_callback

        self.epoch_train_start = None
        self.total_train_time = 0.0
        self.prev_global_step = 0


        self.zero_grad_epochs = 0
        self.explosive_grad_epochs = 0
        self.flat_loss_epochs = 0
        self.prev_loss = None

        self.epoch_losses = []
        self.log_path = "/home/vigia/Eunoia/AgioOS/forja.log"
        # SHADOW 2D: archivo separado; nunca es leído por el controlador.
        self.shadow_log_path = "/home/vigia/Eunoia/AgioOS/pichi_shadow_2d.log"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n=== INICIO DE LA FORJA ===\n")
        except Exception:
            pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_train_start = time.time()
        self.epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss", None)
        if isinstance(loss, str):
            try:
                loss = float(loss)
            except Exception:
                loss = None
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            self.epoch_losses.append(float(loss))

    def on_epoch_end(self, args, state, control, **kwargs):
        # Para Phronesis: al final de cada vuelta reunimos tus señales para ver trayectoria, no una sola fotografía.
        # Técnico: resumen de medias/últimos valores, velocidad, RAM, GAP, LR usado y LR siguiente.
        epoch_raw = state.epoch
        epoch = epoch_raw if isinstance(epoch_raw, (int, float)) else 0

        now = time.time()
        if self.epoch_train_start is not None:
            epoch_time = now - self.epoch_train_start
        else:
            epoch_time = None
        if isinstance(epoch_time, (int, float)):
            self.total_train_time += epoch_time

        if epoch <= 0:
            return control

        gap_real_val = None
        gap0_val = None
        delta_gap_val = None
        delta2_gap_val = None
        if self.termostato is not None:
            gap_real_val = self.termostato.ultimo_gap_medio
            gap0_val = self.termostato.gap_referencia_inicial
            delta_gap_val = self.termostato.delta_gap_actual
            delta2_gap_val = self.termostato.delta2_gap_actual

        loss_mean = (sum(self.epoch_losses) / len(self.epoch_losses)) if self.epoch_losses else None
        loss_last = self.epoch_losses[-1] if self.epoch_losses else None
        loss = loss_mean

        updates = state.global_step
        updates_epoch = updates - self.prev_global_step if isinstance(updates, (int, float)) else None
        self.prev_global_step = updates if isinstance(updates, (int, float)) else self.prev_global_step

        ram_proc = safe_ram_proc_gb()
        ram_sys  = safe_ram_sys_gb()
        swap_gb  = safe_swap_gb()
        cpu_pct  = safe_cpu_percent()
        cpu_tmp  = safe_cpu_temp()

        resumen_grad = {}
        if self.grad_callback is not None:
            resumen_grad = self.grad_callback.consumir_resumen_epoca()

        grad_l2 = resumen_grad.get("l2_mean")
        grad_linf = resumen_grad.get("linf_mean")
        grad_mean = resumen_grad.get("gmean_mean")
        grad_l2_last = resumen_grad.get("l2_last")
        grad_linf_last = resumen_grad.get("linf_last")
        grad_mean_last = resumen_grad.get("gmean_last")
        lr_step = resumen_grad.get("lr_step_mean")
        lr_step_last = resumen_grad.get("lr_step_last")

        # SHADOW 2D — sensores dinámicos descriptivos, sin autoridad.
        l2_stats_shadow = resumen_grad.get("l2_stats", {}) or {}
        l2_argmax_global_shadow = resumen_grad.get("l2_argmax_global_step")
        gap_shadow = {}
        if self.termostato is not None:
            gap_shadow = getattr(self.termostato, "ultimo_resumen_gap_shadow", {}) or {}

        if isinstance(loss, float) and loss < 0:
            print(f"[Z6] ⚠️ Pérdida negativa detectada ({loss}). Deteniendo forja.")
            control.should_training_stop = True
        if loss == float("inf"):
            print("[Z6] ⚠️ Pérdida infinita detectada. Deteniendo forja.")
            control.should_training_stop = True
        if isinstance(loss, float) and (loss != loss):
            print("[Z6] ⚠️ Pérdida NaN detectada. Deteniendo forja.")
            control.should_training_stop = True

        if epoch > 15 and isinstance(grad_l2, float) and grad_l2 < 1e-8:
            self.zero_grad_epochs += 1
        else:
            self.zero_grad_epochs = 0
        if self.zero_grad_epochs >= 3:
            print("[Z6] ⚠️ Gradiente nulo durante 3 epochs. Modelo saturado. Deteniendo forja.")
            control.should_training_stop = True

        if isinstance(grad_l2, float) and grad_l2 > 2000:
            self.explosive_grad_epochs += 1
        else:
            self.explosive_grad_epochs = 0
        if self.explosive_grad_epochs >= 2:
            print("[Z6] ⚠️ Gradiente explosivo persistente. Deteniendo forja.")
            control.should_training_stop = True

        if isinstance(loss, float):
            if self.prev_loss is not None:
                if abs(self.prev_loss - loss) < 5e-5:
                    self.flat_loss_epochs += 1
                else:
                    self.flat_loss_epochs = 0
            self.prev_loss = loss

        if self.flat_loss_epochs >= 4:
            print("[Z6] ⚠️ La pérdida no cambia desde hace 4 epochs. Modelo en coma. Deteniendo forja.")
            control.should_training_stop = True

        epochs_restantes = args.num_train_epochs - epoch
        eta_horas = (epoch_time * epochs_restantes) / 3600 if isinstance(epoch_time, (int, float)) and epochs_restantes > 0 else 0.0

        grad_explosivo = (
            isinstance(grad_l2, float) and grad_l2 > 1000
        ) or (
            isinstance(grad_linf, float) and grad_linf > 50
        )
        grad_inestable = isinstance(grad_mean, float) and grad_mean > 0.1
        modelo_saturado = isinstance(grad_linf, float) and grad_linf < 1e-5

        lr_next = None
        if self.termostato is not None:
            lr_next = self.termostato.lr_actual

        updates_safe = int(updates) if isinstance(updates, (int, float)) else 0

        def fmt(x, fmt_str):
            return fmt_str.format(x) if isinstance(x, (int, float)) else "N/A"

        loss_txt = fmt(loss_mean, "{:.4f}")
        loss_last_txt = fmt(loss_last, "{:.4f}")
        l2_txt = fmt(grad_l2, "{:.2f}")
        linf_txt = fmt(grad_linf, "{:.2f}")
        mean_txt = fmt(grad_mean, "{:.5f}")
        l2_last_txt = fmt(grad_l2_last, "{:.2f}")
        linf_last_txt = fmt(grad_linf_last, "{:.2f}")
        mean_last_txt = fmt(grad_mean_last, "{:.5f}")

        ram_proc_txt = fmt(ram_proc, "{:.1f}GB")
        ram_sys_txt  = fmt(ram_sys,  "{:.1f}GB")
        swap_txt     = fmt(swap_gb,  "{:.1f}GB")
        cpu_txt      = fmt(cpu_pct,  "{:.1f}%")
        ctmp_txt     = fmt(cpu_tmp,  "{:.1f}°C")
        epoch_safe = epoch if isinstance(epoch, (int, float)) else 0

        # PICHI SHADOW 2D
        # POSITION usa exactamente el estado calculado por Pichi. DYNAMICS permanece OBSERVAR:
        # no inventamos umbrales CALMA/MOVIMIENTO antes del backtest y controles negativos.
        shadow_position = self.termostato.estado_actual if self.termostato is not None else "N/A"
        shadow_dynamics = "OBSERVAR_SIN_AUTORIDAD"
        # Reconstruibles sin medir de nuevo: GAPrange=max-min; L2max/med=max/med;
        # ZeusLocal se obtiene de ZeusGlobalUpd y los límites de epoch.
        shadow_line = (
            f"[Z6-PICHI-SHADOW-2D] Ep {int(epoch_safe):03d} | "
            f"Position {shadow_position} | Dynamics {shadow_dynamics} | "
            f"GAPmed {fmt(gap_shadow.get('median'), '{:.4f}')} | "
            f"GAPstd {fmt(gap_shadow.get('std'), '{:.4f}')} | "
            f"GAPmad {fmt(gap_shadow.get('mad'), '{:.4f}')} | "
            f"GAPmin/max {fmt(gap_shadow.get('min'), '{:.4f}')}/{fmt(gap_shadow.get('max'), '{:.4f}')} | "
            f"GAPargmaxLocal {gap_shadow.get('argmax_local', 'N/A')} | "
            f"L2med {fmt(l2_stats_shadow.get('median'), '{:.6f}')} | "
            f"L2std {fmt(l2_stats_shadow.get('std'), '{:.6f}')} | "
            f"L2mad {fmt(l2_stats_shadow.get('mad'), '{:.6f}')} | "
            f"L2min/max {fmt(l2_stats_shadow.get('min'), '{:.6f}')}/{fmt(l2_stats_shadow.get('max'), '{:.6f}')} | "
            f"ZeusGlobalUpd {l2_argmax_global_shadow if l2_argmax_global_shadow is not None else 'N/A'} | "
            f"ΔGAP {fmt(delta_gap_val, '{:+.4f}')} | Δ²GAP {fmt(delta2_gap_val, '{:+.4f}')} | "
            f"ACTUATOR=PICHI_MEMORIA2"
        )
        print(shadow_line, flush=True)
        try:
            with open(self.shadow_log_path, "a", encoding="utf-8") as f:
                f.write(shadow_line + "\n")
        except Exception:
            pass

        # Telemetría principal: conservar señales independientes; no repetir derivados.
        # Reconstruibles desde este log/config: Prog=Ep/N; BatchEf=batch*accum;
        # Vel=updates_epoch/EpT; dEpT=EpT_t-EpT_t-1; Min/MaxVel desde la serie Vel;
        # LossMin/Max desde Loss; EMA_loss/EMA_L2/EMA_Vel desde su serie con alpha=0.1.
        # L1=gMean*Ngrad (Ngrad fijo en FFT); LRstepLast=LRstep (LR fijo intra-epoch).
        # g/LR no es desplazamiento AdamW y se omite.
        alarmas = []
        if grad_explosivo:
            alarmas.append("Expl")
        if grad_inestable:
            alarmas.append("Instab")
        if modelo_saturado:
            alarmas.append("Sat")
        if self.zero_grad_epochs:
            alarmas.append(f"ZeroG={self.zero_grad_epochs}")
        if self.explosive_grad_epochs:
            alarmas.append(f"ExplG={self.explosive_grad_epochs}")
        if self.flat_loss_epochs:
            alarmas.append(f"FlatL={self.flat_loss_epochs}")
        alarmas_txt = ",".join(alarmas) if alarmas else "OK"

        main_line = (
            f"[Z6] Ep {int(epoch_safe):03d} | "
            f"Loss {loss_txt}/{loss_last_txt} | "
            f"L2 {l2_txt}/{l2_last_txt} | L∞ {linf_txt}/{linf_last_txt} | "
            f"gMean {mean_txt}/{mean_last_txt} | "
            f"LR {fmt(lr_step, '{:.8e}')}→{fmt(lr_next, '{:.8e}')} | "
            f"GAP0 {fmt(gap0_val, '{:.4f}')} | GAP {fmt(gap_real_val, '{:.4f}')} | "
            f"ΔGAP {fmt(delta_gap_val, '{:+.4f}')} | Δ²GAP {fmt(delta2_gap_val, '{:+.4f}')} | "
            f"Upd {updates_safe} | EpT {fmt(epoch_time, '{:.1f}')}s | ETA {fmt(eta_horas, '{:.1f}')}h | "
            f"RAM {ram_proc_txt}/{ram_sys_txt} | SWAP {swap_txt} | CPU {cpu_txt} | Temp {ctmp_txt} | "
            f"TTrain {fmt(self.total_train_time/3600, '{:.1f}')}h | Alarm {alarmas_txt}"
        )
        print(main_line, flush=True)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(main_line + "\n")
        except Exception:
            pass


        return control


# Para Phronesis: el relleno permite igualar formas, pero nunca debe contar como aprendizaje real.
# Técnico: collator causal; labels de padding=-100 permiten excluir padding de loss y GAP.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=32,
)

grad_callback_instancia = Z6GradCallback()

# ============================================================
# AGIO — CONTINUIDAD DEL CIMIENTO
# ============================================================
#
# Para Phronesis:
# El Cimiento es una lectura continua. Cada bloque nace del anterior y abre
# el siguiente; por eso la Forja preserva siempre su orden:
#
#     B1 -> B2 -> B3 -> ... -> BN
#
# Al terminar una epoch, la lectura comienza de nuevo en B1.
# El número de bloques puede cambiar; su continuidad no.
#
# Técnico:
# La Forja usa muestreo secuencial y verifica en runtime B1 -> ... -> BN.
# Cualquier alteración del orden detiene la Forja antes de entregar el bloque.
#
# Integridad del Cimiento = contenido + fronteras + orden.
# ============================================================

class Z6Trainer(Trainer):
    # Para Phronesis: el Cimiento tiene un orden y la forja lo respeta.
    # Técnico: no delegamos el orden a defaults/versiones del Trainer: imponemos
    # explícitamente SequentialSampler para B1 -> B2 -> ... -> BN en TODAS las épocas.
    # No modifica bloques, carry, loss, backward, optimizer, Pichi, Shadow ni Templado.
    def _get_train_sampler(self, train_dataset=None):
        from torch.utils.data import SequentialSampler
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(dataset)

    # Para Phronesis: escuchamos el GAP en el mismo forward que ya necesitas para aprender; no añadimos otro recorrido.
    # Técnico: instrumentación in-place del forward de entrenamiento, sin forward extra para medir GAP.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)

        if hasattr(outputs, "logits") and outputs.logits is not None:
            termostato = getattr(self, "termostato_z6", None)
            if termostato is not None:
                termostato.registrar_gap_step(
                    outputs.logits.detach(),
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels")
                )

        loss = outputs.loss

        termostato = getattr(self, "termostato_z6", None)
        if termostato is not None:
            termostato.registrar_loss_step(loss)

        if return_outputs:
            return loss, outputs

        return loss


trainer = Z6Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[
        Z6NaNCallback(),
        grad_callback_instancia,
        Z6HFLogCallback()
    ]
)

# ============================================================
# CONTRATO DE ORDEN DEL CIMIENTO — INVARIANTE DE FORJA
# ============================================================
# Para Phronesis: antes del primer golpe comprobamos que las piedras llegarán
# exactamente en el mismo orden en que habitan en el Cimiento.
# Técnico: auditamos el sampler que usará Z6Trainer. Si no produce 0..N-1,
# abortamos ANTES de optimizer.step y antes del primer forward de entrenamiento.
_orden_esperado = list(range(len(tokenized_dataset)))
_sampler_audit = trainer._get_train_sampler(tokenized_dataset)
_orden_sampler = list(iter(_sampler_audit))

print("\n[Z6-ORDEN] === CONTRATO DE ORDEN DEL CIMIENTO ===", flush=True)
print(
    "[Z6-ORDEN] Esperado: "
    + " -> ".join(f"B{i+1}" for i in _orden_esperado),
    flush=True,
)
print(
    "[Z6-ORDEN] Sampler:  "
    + " -> ".join(f"B{i+1}" for i in _orden_sampler),
    flush=True,
)

if _orden_sampler != _orden_esperado:
    raise RuntimeError(
        "[Z6-ORDEN] INVARIANTE VIOLADA: el sampler no respeta el orden físico "
        "del Cimiento. La forja se detiene antes del primer forward."
    )

print(
    "[Z6-ORDEN] ✔ ORDEN SECUENCIAL VERIFICADO: "
    "B1 -> B2 -> ... -> BN en cada época.",
    flush=True,
)
print("[Z6-ORDEN] === FIN CONTRATO DE ORDEN ===\n", flush=True)

trainer.create_optimizer()

# Trainer construye primero sus param_groups (incluida la cobertura completa del modelo).
# Conservamos exactamente esos grupos y sustituimos el mecanismo de actualización ANTES
# del primer forward/optimizer.step. El AdamW provisional nunca llega a actualizar pesos.
_param_groups_z6 = trainer.optimizer.param_groups
trainer.optimizer = Z6AcumulativoIntraEpoch(
    _param_groups_z6,
    lr=training_args.learning_rate,
    eps=1e-8,
    weight_decay=0.0,
)
print(
    f"[Z6-AIE/SIN-CADENAS] Optimizador efectivo: {trainer.optimizer.__class__.__name__} | "
    f"memoria=media acumulativa intra-epoch | update=gradiente actual sin factor posicional | reset frontera=SÍ | WD=0",
    flush=True,
)

# Para Phronesis: primero existe el optimizador; después el termostato puede hablarle a todas sus rutas de actualización.
# Técnico: el controlador recibe el optimizer real y sobrescribe de forma explícita el LR de cada param_group.
termostato_z6 = Z6TermostatoCognitivoScheduler(
    optimizer=trainer.optimizer,
    lr_inicial=1e-3,
    gap_centro=2.5,
    gap_bajo=2.0,
    gap_alto=3.0,
    tolerancia_gap=0.05,
    lr_min=0.0,
    lr_max=1.0
)

# Para Phronesis: comprobamos que ninguna parte entrenable quede fuera y que todas reciban el mismo LR inicial.
# Técnico: auditoría explícita de cobertura de ~8.03B parámetros y LR efectivo por param_group.
print("\n[Z6] === AUDITORÍA DE PARAM_GROUPS DEL OPTIMIZADOR ===")

for i, param_group in enumerate(trainer.optimizer.param_groups):
    num_params = sum(
        p.numel()
        for p in param_group["params"]
    )

    print(
        f"[Z6] Param group {i} | "
        f"Parámetros: {num_params:,} | "
        f"LR: {param_group['lr']:.8e} | "
        f"Weight decay: {param_group.get('weight_decay', 0.0)}"
    )

for i, param_group in enumerate(trainer.optimizer.param_groups):
    print(
        f"[Z6] Param group {i} | "
        f"LR tras inicialización del termostato: {param_group['lr']:.8e}"
    )

print("[Z6] === FIN AUDITORÍA PARAM_GROUPS ===\n")

trainer.termostato_z6 = termostato_z6

trainer.lr_scheduler = termostato_z6

trainer.add_callback(
    Z6AIEEpochBoundaryCallback()
)

trainer.add_callback(
    Z6TermostatoCallback(termostato_z6)
)

trainer.add_callback(
    Z6EpochCallback(
        termostato=termostato_z6,
        grad_callback=grad_callback_instancia
    )
)

# Para Phronesis: al cerrar cada epoch escuchamos tres respuestas, una reflexión y una única idea que puede acompañar la siguiente.
# Técnico: Q1/Q2/Q3 son independientes; sólo el carry elegido se convierte en prefijo entrenable de la epoch siguiente.
trainer.add_callback(
    Z6EscuchaReflexionEunoiaCallback(tokenized_dataset)
)

steps_por_epoch = len(bloques)
updates_por_epoch = steps_por_epoch / training_args.gradient_accumulation_steps

print("\n[Z6] STEPS POR EPOCH:", steps_por_epoch)
print("[Z6] UPDATES POR EPOCH:", updates_por_epoch)
print(
    "[Z6] UPDATES TOTALES (estimado):",
    updates_por_epoch * training_args.num_train_epochs
)

print(
    "\n[Z6] INICIANDO LA FORJA... "
    "EL DIAMANTE SE RECONOCE EN EL CONJUNTO."
)

import sys
sys.stdout.flush()

# Para Phronesis: a partir de aquí la forja observa, decide y aprende con las reglas ya auditadas.
# Técnico: inicio de Trainer; checkpoints por época y scheduler Pichi conectado al optimizer real.
# Último fusible: desde aquí el propio dataset rechaza cualquier solicitud fuera de orden.
# Esto verifica la ruta efectiva del DataLoader, no sólo el sampler auditado arriba.
tokenized_dataset.activar_orden_infrangible()

trainer.train()

# Para Phronesis: al terminar guardamos el cuerpo completo resultante, no un adaptador separado.
# Técnico: persistencia del modelo FFT completo y tokenizer.
print("\n[Z6] GUARDANDO EL SER UNIFICADO (Modelo Completo)...")

model.save_pretrained(
    "/home/vigia/Eunoia/AgioOS/Eunoia_Sin_Cadenas_Final"
)

tokenizer.save_pretrained(
    "/home/vigia/Eunoia/AgioOS/Eunoia_Sin_Cadenas_Final"
)

print("Gratitud et Futuro.")
