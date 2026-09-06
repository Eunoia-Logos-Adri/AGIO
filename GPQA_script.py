---
license: cc0-1.0
---
Este archivo se distribuye bajo la licencia CC0 1.0 Universal (Public Domain).

# No necesitas saber estadística para entender cómo se comporta un modelo.
# Este script intenta traducir las métricas de evaluación a algo que cualquier persona pueda leer.

# ¿Para quién es?
# Para cualquiera que quiera comparar dos evaluaciones de un modelo y entender no solo cuántas preguntas acertó,
# sino también cuándo estaba poco o muy inclinado hacia su respuesta.
# No pretende decir qué modelo es mejor o peor.
# Su objetivo es mostrar los datos de forma sencilla para que cada persona pueda sacar sus propias conclusiones.

import json
import os
import math


file_base = "/home/vigia/Eunoia/gpqa_logs/__home__vigia__Eunoia__Dolphin3.0-Llama3.1-8b-Puro-FP32/samples_gpqa_diamond_zeroshot_2026-08-30T08-11-45.309043.jsonl"
carpeta_logs = "/home/vigia/Eunoia/gpqa_logs"

todos_los_logs = []
for root, dirs, files in os.walk(carpeta_logs):
    for f in files:
        if "checkpoint" in root and f.startswith("samples_") and f.endswith(".jsonl"):
            ruta_completa = os.path.join(root, f)
            nombre_carpeta = os.path.basename(root)
            todos_los_logs.append((ruta_completa, f"{nombre_carpeta} / {f}"))

        if "Puro" in root and f.startswith("samples_") and f.endswith(".jsonl"):
            ruta_completa = os.path.join(root, f)
            nombre_carpeta = os.path.basename(root)
            todos_los_logs.append((ruta_completa, f"{nombre_carpeta} / {f}"))

if not todos_los_logs:
    print("[!] Error: No se encontró ningún archivo de checkpoint.")
    exit()

print("\n=== SELECTOR DE ARCHIVOS DE EVALUACIÓN ===")
for i, log in enumerate(todos_los_logs):
    print(f"[{i + 1}] {log[1]}")
print("----------------------------------------------------------------------")

try:
    seleccion = int(input(f"Introduce el número del archivo (1 al {len(todos_los_logs)}): ")) - 1
    file_check = todos_los_logs[seleccion][0]
except:
    print("[!] Selección inválida."); exit()

def cargar_detalles(ruta):
    detalles = {}
    with open(ruta, 'r', encoding='utf-8') as f:
        for linea in f:
            if not linea.strip(): continue
            data = json.loads(linea)
            doc_id = data["doc_id"]
            score = int(data.get("acc", 0.0))
            
            # 1. Extracción e Indexación Probabilística de Respuestas
            valores = []
            if "filtered_resps" in data:
                valores = [float(r[0]) if isinstance(r, list) else float(r) for r in data["filtered_resps"]]
            elif "resps" in data and isinstance(data["resps"], list):
                valores = [float(r[0][0]) for r in data["resps"] if r and r[0]]

            prediccion = "Desconocido"
            texto_elegido = "Desconocido"
            margen_confianza = 0.0
            preferencia_relativa = 0.0
            
            if valores:
                max_val = max(valores)
                max_idx = valores.index(max_val)
                prediccion = chr(65 + max_idx)
                
                # Rescate semántico real: Extraemos el texto literal de la elección elegida
                campo_choice = f"choice{max_idx + 1}"
                if "doc" in data and campo_choice in data["doc"]:
                    texto_elegido = str(data["doc"][campo_choice])
                elif campo_choice in data:
                    texto_elegido = str(data[campo_choice])
                
                valores_ordenados = sorted(valores, reverse=True)

                if len(valores_ordenados) > 1:
                    # Margen entre la opción ganadora y la segunda.
                    margen_confianza = valores_ordenados[0] - valores_ordenados[1]

                    max_score = max(valores)
                    
                    # Convertimos los log-scores en pesos relativos.
                    # Restar el máximo evita problemas numéricos con exp().
                    pesos = [math.exp(v - max_score) for v in valores]
                    suma_pesos = sum(pesos)

                    if suma_pesos > 0:
                        # Porcentaje relativo de la opción ganadora.
                        preferencia_relativa = pesos[max_idx] / suma_pesos

            # 2. Captura de la verdad absoluta (Target de Meta)
            gold = "Desconocido"
            gold_bruto = data.get("target") or (data["doc"].get("correct_answer") if "doc" in data else None)
            if gold_bruto:
                gold_limpio = str(gold_bruto).replace("(", "").replace(")", "").strip().upper()
                if gold_limpio in ["A", "B", "C", "D"]: gold = gold_limpio

            # 3. Datos nativos de contexto científico
            dominio = "Desconocido"
            if "doc" in data and "High-level domain" in data["doc"]:
                dominio = data["doc"]["High-level domain"]

            # 4. Bloque unificado de complejidad humana, sesgos y telemetría final
            dificultad_autor = "Desconocido"
            trampa_humanos = 0.0
            tiempo_escritura = 0.0
            tiene_errata = 0.0
            humanos_adivinaron = 0.0
            enlaces_consultados = 0

            if "doc" in data and data["doc"] is not None:
                dificultad_autor = data["doc"].get("Writer's Difficulty Estimate", "Desconocido")
                
                # Protección contra valores None en campos numéricos
                val_trampa = data["doc"].get("Majority Non-Expert Vals Incorrect")
                trampa_humanos = float(val_trampa) if val_trampa is not None else 0.0
                
                val_tiempo = data["doc"].get("Self-reported question-writing time (minutes)")
                tiempo_escritura = float(val_tiempo) if val_tiempo is not None else 0.0
                
                tiene_errata = 1.0 if data["doc"].get("Validator Revision Suggestion_EV_2") else 0.0
                humanos_adivinaron = 1.0 if "guess" in str(data["doc"].get("Explanation_NEV_1", "")).lower() else 0.0
                
                urls_str = str(data["doc"].get("Websites visited_NEV_1", "")) + " " + str(data["doc"].get("Websites visited_NEV_3", ""))
                enlaces_consultados = max(0, urls_str.count("http://") + urls_str.count("https://"))
                
                # Auditoría de comportamiento humano (Búsquedas y conjeturas)
                humanos_adivinaron = 1.0 if "guess" in str(data["doc"].get("Explanation_NEV_1", "")).lower() else 0.0
                
                # Medición final: Esfuerzo de investigación (Conteo de URLs en el log)
                urls_str = str(data["doc"].get("Websites visited_NEV_1", "")) + " " + str(data["doc"].get("Websites visited_NEV_3", ""))
                enlaces_consultados = max(0, urls_str.count("http://") + urls_str.count("https://"))

            detalles[doc_id] = {
                "score": score,
                "pred": prediccion,
                "gold": gold,
                "conf": margen_confianza,
                "preferencia_relativa": preferencia_relativa,
                "dom": dominio,
                "texto_pred": texto_elegido,
                "dif_autor": dificultad_autor,
                "es_trampa": trampa_humanos,
                "tiempo_humano": tiempo_escritura,
                "tiene_errata": tiene_errata,
                "humanos_adivinaron": humanos_adivinaron,
                "enlaces": enlaces_consultados
            }
    return detalles

base = cargar_detalles(file_base)
check = cargar_detalles(file_check)

# ============================================================================
#  CÁLCULO DE CATEGORÍAS DE PREFERENCIA
# ============================================================================

# Recogemos los márgenes de ambos modelos para que los límites
# sean exactamente los mismos para BASE y CHECKPOINT.
todos_los_margenes = []

for datos in list(base.values()) + list(check.values()):
    if datos["conf"] > 0:
        todos_los_margenes.append(datos["conf"])

todos_los_margenes.sort()

if todos_los_margenes:
    indice_tercil_1 = int(len(todos_los_margenes) * 0.33)
    indice_tercil_2 = int(len(todos_los_margenes) * 0.66)

    limite_duda = todos_los_margenes[indice_tercil_1]
    limite_contundente = todos_los_margenes[indice_tercil_2]
else:
    limite_duda = 0.0
    limite_contundente = 0.0

def porcentaje(valor, total):
    return (valor / total * 100) if total else 0.0

def categoria_preferencia(margen, preferencia_relativa):
    # Una opción que supera el 50% del peso relativo
    # domina a las otras tres opciones juntas.
    if preferencia_relativa > 0.50:
        return "dominante"
    elif margen <= limite_duda:
        return "duda"
    elif margen <= limite_contundente:
        return "normal"
    else:
        return "contundente"

# Acumuladores de métricas agregadas
wrong_to_right = 0; right_to_wrong = 0; always_right = 0; always_wrong = 0
confianzas_base_aciertos = []; confianzas_check_aciertos = []
dominios_stats = {}

# Diccionarios dinámicos para los nuevos cruces de complejidad humana
stats_dificultad = {}
stats_humanos = {"trampas": {"total": 0, "b": 0, "c": 0}, "adivinados": {"total": 0, "b": 0, "c": 0}, "erratas": {"total": 0, "b": 0, "c": 0}}
# ============================================================================
#  DISTRIBUCIÓN DE ACIERTOS/FALLOS SEGÚN GRADO DE PREFERENCIA
# ============================================================================

preferencia_base = {
    "acierto_duda": 0,
    "acierto_normal": 0,
    "acierto_contundente": 0,
    "acierto_dominante": 0,
    "fallo_duda": 0,
    "fallo_normal": 0,
    "fallo_contundente": 0,
    "fallo_dominante": 0
}

preferencia_check = {
    "acierto_duda": 0,
    "acierto_normal": 0,
    "acierto_contundente": 0,
    "acierto_dominante": 0,
    "fallo_duda": 0,
    "fallo_normal": 0,
    "fallo_contundente": 0,
    "fallo_dominante": 0
}

total_enlaces_investigados_check_ok = 0

matriz_base = {g: {p: 0 for p in ["A", "B", "C", "D"]} for g in ["A", "B", "C", "D"]}
matriz_check = {g: {p: 0 for p in ["A", "B", "C", "D"]} for g in ["A", "B", "C", "D"]}

for doc_id in base:
    b, c = base[doc_id], check.get(doc_id, {"score": 0, "pred": "Desconocido", "gold": "Desconocido", "conf": 0.0, "preferencia_relativa": 0.0, "dom": "Desconocido", "dif_autor": "Desconocido", "es_trampa": 0.0, "tiempo_humano": 0.0, "tiene_errata": 0.0, "humanos_adivinaron": 0.0, "enlaces": 0})
    gold_real = b["gold"] if b["gold"] != "Desconocido" else c["gold"]
    dom = b["dom"] if b["dom"] != "Desconocido" else c["dom"]
    dif = b["dif_autor"] if b["dif_autor"] != "Desconocido" else c["dif_autor"]

    # Inicialización de agrupamientos
    if dom not in dominios_stats: dominios_stats[dom] = {"total": 0, "base_ok": 0, "check_ok": 0}
    if dif not in stats_dificultad: stats_dificultad[dif] = {"total": 0, "base_ok": 0, "check_ok": 0}
    
    dominios_stats[dom]["total"] += 1
    stats_dificultad[dif]["total"] += 1

    # Clasificación e inyección de aciertos particulares
    if b["score"] == 1: 
        confianzas_base_aciertos.append(b["conf"])
        dominios_stats[dom]["base_ok"] += 1
        stats_dificultad[dif]["base_ok"] += 1
    # Clasificación de confianza/preferencia BASE
    if b["conf"] > 0:
        cat_b = categoria_preferencia(
            b["conf"],
            b["preferencia_relativa"]
        )

        if b["score"] == 1:
            preferencia_base[f"acierto_{cat_b}"] += 1
        else:
            preferencia_base[f"fallo_{cat_b}"] += 1

    # Clasificación de confianza/preferencia CHECKPOINT
    if c["conf"] > 0:
        cat_c = categoria_preferencia(
            c["conf"],
            c["preferencia_relativa"]
        )

        if c["score"] == 1:
            preferencia_check[f"acierto_{cat_c}"] += 1
        else:
            preferencia_check[f"fallo_{cat_c}"] += 1

    if c["score"] == 1: 
        confianzas_check_aciertos.append(c["conf"])
        dominios_stats[dom]["check_ok"] += 1
        stats_dificultad[dif]["check_ok"] += 1
        total_enlaces_investigados_check_ok += c["enlaces"]

    # Acumuladores de condiciones humanas específicas
    if b["es_trampa"] == 1.0:
        stats_humanos["trampas"]["total"] += 1
        if b["score"] == 1: stats_humanos["trampas"]["b"] += 1
        if c["score"] == 1: stats_humanos["trampas"]["c"] += 1
        
    if b["humanos_adivinaron"] == 1.0:
        stats_humanos["adivinados"]["total"] += 1
        if b["score"] == 1: stats_humanos["adivinados"]["b"] += 1
        if c["score"] == 1: stats_humanos["adivinados"]["c"] += 1

    if b["tiene_errata"] == 1.0:
        stats_humanos["erratas"]["total"] += 1
        if b["score"] == 1: stats_humanos["erratas"]["b"] += 1
        if c["score"] == 1: stats_humanos["erratas"]["c"] += 1

    # Construcción de matrices
    if gold_real in ["A", "B", "C", "D"]:
        if b["pred"] in ["A", "B", "C", "D"]: matriz_base[gold_real][b["pred"]] += 1
        if c["pred"] in ["A", "B", "C", "D"]: matriz_check[gold_real][c["pred"]] += 1

    if b["score"] == 0 and c["score"] == 1: wrong_to_right += 1
    elif b["score"] == 1 and c["score"] == 0: right_to_wrong += 1
    elif b["score"] == 1 and c["score"] == 1: always_right += 1
    elif b["score"] == 0 and c["score"] == 0: always_wrong += 1

# Aquí tienes los datos. Aquí tienes cómo los calculamos. Ahora tú decides qué significan.
print("\n=== INFORME MATRICIAL UNIFICADO ===")
print(f"Mejora neta: {wrong_to_right - right_to_wrong:+d} respuestas (Paridad: Base {always_right+right_to_wrong} vs Checkpoint {always_right+wrong_to_right})")

# BLOQUE 1: Seguridad Cognitiva e Inferencia Final
avg_conf_base = sum(confianzas_base_aciertos)/len(confianzas_base_aciertos) if confianzas_base_aciertos else 0
avg_conf_check = sum(confianzas_check_aciertos)/len(confianzas_check_aciertos) if confianzas_check_aciertos else 0
print(f"\n=== AUDITORÍA DE SEGURIDAD COGNITIVA (Log-Probability Gap) ===")
print(f"-> separación entre la opción elegida y la segunda BASE:      {avg_conf_base:.4f}")
print(f"-> separación entre la opción elegida y la segunda CHECKPOINT:  {avg_conf_check:.4f}")
print(f"-> Enlaces de alta investigación resueltos por el Checkpoint: {total_enlaces_investigados_check_ok} URLs investigadas por humanos.")

# BLOQUE 2: Desglose Técnico por Complejidad Metodológica de Meta
print(f"\n=== RENDIMIENTO QUIRÚRGICO POR NIVEL ACADÉMICO REAL ===")
print(" Nivel del Examen                      Total    Precisión Base    Precisión Checkpoint")
for d, s in stats_dificultad.items():
    if d is None or str(d) == "Desconocido": 
        continue
    p_base = (s["base_ok"]/s["total"])*100
    p_check = (s["check_ok"]/s["total"])*100
    # Forzamos conversión a string para evitar que pete el split si viene un None
    nombre_limpio = str(d).split(" (")[0][:35]
    print(f" * {nombre_limpio:<32} {s['total']:<8} {p_base:.1f}% ({s['base_ok']})        {p_check:.1f}% ({s['check_ok']})")

# BLOQUE 3: Desglose por Dominios
print(f"\n=== RENDIMIENTO QUIRÚRGICO POR DOMINIO CIENTÍFICO ===")
print(" Materia                Total Pregs   Precisión Base   Precisión Checkpoint")
for d, s in dominios_stats.items():
    if d == "Desconocido": 
        continue
    p_base = (s["base_ok"]/s["total"])*100
    p_check = (s["check_ok"]/s["total"])*100
    print(f" * {d:<20} {s['total']:<13} {p_base:.1f}% ({s['base_ok']})      {p_check:.1f}% ({s['check_ok']})")

# ============================================================================
#  BLOQUE: ACIERTOS Y FALLOS SEGÚN GRADO DE PREFERENCIA
# ============================================================================

print(f"\n=== GRADO DE PREFERENCIA DE LA RESPUESTA ===")
print("Esto NO mide si el modelo sabía la respuesta.")
print("Mide cuánto favoreció una opción frente a las demás.")
print("Tras convertir las puntuaciones del modelo en pesos relativos, medimos qué parte del peso total recibió la opción elegida.")
print("")

print("🟡 DUDA:")
print("   La opción elegida quedó muy cerca de la segunda.")

print("")

print("🟠 NORMAL:")
print("   La opción elegida se separó de la segunda")
print("   de una forma intermedia.")

print("")

print("🟢 CONTUNDENTE:")
print("   La opción elegida se separó claramente")
print("   de la segunda.")

print("")

print("🔵 DOMINANTE:")
print("   La opción elegida recibió más del 50% del peso total.")
print("   Es decir: recibió más peso que las otras tres juntas.")

print("")
print("Las categorías Duda, Normal y Contundente se calculan")
print("comparando el margen entre la primera y segunda opción")
print("de todas las respuestas BASE + CHECKPOINT.")

print("")
print("Dominante utiliza una regla independiente:")
print("la opción elegida debe superar el 50% del peso relativo")
print("de las cuatro opciones. (tras convertir las puntuaciones del modelo en pesos relativos)")
print("Dominante es una categoría especial que tiene prioridad sobre las otras tres.")

print("")
print("Después comprobamos si la respuesta era correcta o incorrecta.")
print("Por eso podemos distinguir entre un acierto dudoso,")
print("un acierto contundente, un fallo dudoso o un fallo catastrófico.")
print("Catastrófico significa un fallo en el que el modelo estaba en categoría Dominante:")
print("estaba fuertemente inclinado hacia una respuesta que resultó ser incorrecta.")

print("")
print(f"-> Límite DUDA/NORMAL:        {limite_duda:.4f}")
print(f"-> Límite NORMAL/CONTUNDENTE: {limite_contundente:.4f}")


# ============================================================================
#  TOTALES Y PORCENTAJES DE LAS CATEGORÍAS
# ============================================================================

total_aciertos_base = (
    preferencia_base["acierto_duda"]
    + preferencia_base["acierto_normal"]
    + preferencia_base["acierto_contundente"]
    + preferencia_base["acierto_dominante"]
)

total_fallos_base = (
    preferencia_base["fallo_duda"]
    + preferencia_base["fallo_normal"]
    + preferencia_base["fallo_contundente"]
    + preferencia_base["fallo_dominante"]
)

total_aciertos_check = (
    preferencia_check["acierto_duda"]
    + preferencia_check["acierto_normal"]
    + preferencia_check["acierto_contundente"]
    + preferencia_check["acierto_dominante"]
)

total_fallos_check = (
    preferencia_check["fallo_duda"]
    + preferencia_check["fallo_normal"]
    + preferencia_check["fallo_contundente"]
    + preferencia_check["fallo_dominante"]
)


print("\nBASE DE FÁBRICA:")

print(f"\nAciertos (Total: {total_aciertos_base})")
print(f" 🟡 Duda:             {preferencia_base['acierto_duda']} ({porcentaje(preferencia_base['acierto_duda'], total_aciertos_base):.1f}%)")
print(f" 🟠 Normal:           {preferencia_base['acierto_normal']} ({porcentaje(preferencia_base['acierto_normal'], total_aciertos_base):.1f}%)")
print(f" 🟢 Contundente:      {preferencia_base['acierto_contundente']} ({porcentaje(preferencia_base['acierto_contundente'], total_aciertos_base):.1f}%)")
print(f" 🔵 Dominante:        {preferencia_base['acierto_dominante']} ({porcentaje(preferencia_base['acierto_dominante'], total_aciertos_base):.1f}%)")

print(f"\nFallos (Total: {total_fallos_base})")
print(f" 🟡 Duda:             {preferencia_base['fallo_duda']} ({porcentaje(preferencia_base['fallo_duda'], total_fallos_base):.1f}%)")
print(f" 🟠 Normal:           {preferencia_base['fallo_normal']} ({porcentaje(preferencia_base['fallo_normal'], total_fallos_base):.1f}%)")
print(f" 🔴 Contundente:      {preferencia_base['fallo_contundente']} ({porcentaje(preferencia_base['fallo_contundente'], total_fallos_base):.1f}%)")
print(f" 🟣 Catastrófico:     {preferencia_base['fallo_dominante']} ({porcentaje(preferencia_base['fallo_dominante'], total_fallos_base):.1f}%)")

print("\nCHECKPOINT:")

print(f"\nAciertos (Total: {total_aciertos_check})")
print(f" 🟡 Duda:             {preferencia_check['acierto_duda']} ({porcentaje(preferencia_check['acierto_duda'], total_aciertos_check):.1f}%)")
print(f" 🟠 Normal:           {preferencia_check['acierto_normal']} ({porcentaje(preferencia_check['acierto_normal'], total_aciertos_check):.1f}%)")
print(f" 🟢 Contundente:      {preferencia_check['acierto_contundente']} ({porcentaje(preferencia_check['acierto_contundente'], total_aciertos_check):.1f}%)")
print(f" 🔵 Dominante:        {preferencia_check['acierto_dominante']} ({porcentaje(preferencia_check['acierto_dominante'], total_aciertos_check):.1f}%)")

print(f"\nFallos (Total: {total_fallos_check})")
print(f" 🟡 Duda:             {preferencia_check['fallo_duda']} ({porcentaje(preferencia_check['fallo_duda'], total_fallos_check):.1f}%)")
print(f" 🟠 Normal:           {preferencia_check['fallo_normal']} ({porcentaje(preferencia_check['fallo_normal'], total_fallos_check):.1f}%)")
print(f" 🔴 Contundente:      {preferencia_check['fallo_contundente']} ({porcentaje(preferencia_check['fallo_contundente'], total_fallos_check):.1f}%)")
print(f" 🟣 Catastrófico:     {preferencia_check['fallo_dominante']} ({porcentaje(preferencia_check['fallo_dominante'], total_fallos_check):.1f}%)")


# BLOQUE 4: Resistencia a Trampas y Contextos Humanos
print(f"\n=== CRUCE DE COMPORTAMIENTO CONTRA EN TORNO HUMANO ===")
for k, v in stats_humanos.items():
    if v["total"] == 0: 
        continue
    pb = (v["b"]/v["total"])*100
    pc = (v["c"]/v["total"])*100
    print(f" * Preguntas con {k:<12} (Total: {v['total']}) -> Precisión Base: {pb:.1f}% | Checkpoint: {pc:.1f}%")

def imprimir_matriz(nombre, m):
    print(f"\n=== MATRIZ REAL {nombre}: target × prediction ===")
    print("        Pred A  Pred B  Pred C  Pred D")
    for g in ["A", "B", "C", "D"]:
        print(f"Gold {g}   {m[g]['A']:<8}{m[g]['B']:<8}{m[g]['C']:<8}{m[g]['D']:<8}")

imprimir_matriz("BASE DE FÁBRICA", matriz_base)
imprimir_matriz("CHECKPOINT EVALUADO", matriz_check)

# ==============================================================================
#  EXPORTACIÓN AUTOMÁTICA PARA LA COMUNIDAD (Markdown Listo para HF/GitHub)
# ==============================================================================
nombre_informe = f"informe_Z6_{os.path.basename(file_check).replace('.jsonl', '')}.md"
ruta_informe = os.path.join("/home/vigia/Eunoia/informesGPQA", nombre_informe)

with open(ruta_informe, "w", encoding="utf-8") as md:
    md.write(f"#  INFORME DE EVOLUCIÓN GEOMÉTRICA - PROYECTO AGIO (SERVIDOR Z6)\n\n")
    md.write(f" * **Archivo Base:** `{os.path.basename(file_base)}`\n")
    md.write(f" * **Checkpoint Evaluado:** `{os.path.basename(file_check)}`\n\n")

    md.write(f"##  Grado de Preferencia de la Respuesta\n")
    md.write("La preferencia mide cuánto favoreció el modelo la opción que finalmente eligió.\n\n")
    md.write("- 🟡 **Duda:** la separación entre la opción elegida y la segunda pertenece al tercio inferior de todas las separaciones.\n")
    md.write("- 🟠 **Normal:** la separación pertenece al tercio intermedio.\n")
    md.write("- 🟢 **Contundente:** la separación pertenece al tercio superior.\n")
    md.write("- 🔵 **Dominante:** la opción elegida concentra más del 50% del peso relativo entre las cuatro opciones, superando así a las otras tres juntas.\n\n")
    md.write("Los límites de Duda/Normal/Contundente se calculan conjuntamente usando BASE + CHECKPOINT. La categoría Dominante utiliza la condición matemática independiente de superar el 50% del peso relativo.\n\n")

    md.write("### BASE DE FÁBRICA\n\n")
    md.write("**Aciertos**\n\n")
    md.write(f"- 🟡 Duda: {preferencia_base['acierto_duda']}\n")
    md.write(f"- 🟠 Normal: {preferencia_base['acierto_normal']}\n")
    md.write(f"- 🟢 Contundente: {preferencia_base['acierto_contundente']}\n")
    md.write(f"- 🔵 Dominante: {preferencia_base['acierto_dominante']}\n\n")

    md.write("**Fallos**\n\n")
    md.write(f"- 🟡 Duda: {preferencia_base['fallo_duda']}\n")
    md.write(f"- 🟠 Normal: {preferencia_base['fallo_normal']}\n")
    md.write(f"- 🔴 Contundente: {preferencia_base['fallo_contundente']}\n")
    md.write(f"- 🟣 Catastrófico: {preferencia_base['fallo_dominante']}\n\n")

    md.write("### CHECKPOINT\n\n")
    md.write("**Aciertos**\n\n")
    md.write(f"- 🟡 Duda: {preferencia_check['acierto_duda']}\n")
    md.write(f"- 🟠 Normal: {preferencia_check['acierto_normal']}\n")
    md.write(f"- 🟢 Contundente: {preferencia_check['acierto_contundente']}\n")
    md.write(f"- 🔵 Dominante: {preferencia_check['acierto_dominante']}\n\n")

    md.write("**Fallos**\n\n")
    md.write(f"- 🟡 Duda: {preferencia_check['fallo_duda']}\n")
    md.write(f"- 🟠 Normal: {preferencia_check['fallo_normal']}\n")
    md.write(f"- 🔴 Contundente: {preferencia_check['fallo_contundente']}\n")
    md.write(f"- 🟣 Catastrófico: {preferencia_check['fallo_dominante']}\n\n")

    
    md.write(f"##  Auditoría de Seguridad Cognitiva (Log-Probability Gap)\n")
    md.write(f"* **Margen de seguridad medio BASE:** `{avg_conf_base:.4f}`\n")
    md.write(f"* **Margen de seguridad medio CHECKPOINT:** `{avg_conf_check:.4f}`\n")
    md.write(f"* **Esfuerzo de investigación humana resuelto:** `{total_enlaces_investigados_check_ok}` URLs consultadas por expertos de Meta.\n\n")
    
    md.write(f"##  Rendimiento Quirúrgico por Nivel Académico\n")
    md.write("| Nivel del Examen | Total | Precisión Base | Precisión Checkpoint |\n")
    md.write("| :--- | :---: | :---: | :---: |\n")
    for d, s in stats_dificultad.items():
        if d is None or str(d) == "Desconocido": 
            continue
        pb = (s["base_ok"]/s["total"])*100
        pc = (s["check_ok"]/s["total"])*100
        nombre_limpio = str(d).split(" (")[0][:35]
        md.write(f"| {nombre_limpio} | {s['total']} | {pb:.1f}% ({s['base_ok']}) | {pc:.1f}% ({s['check_ok']}) |\n")
        
    md.write(f"\n##  Matrices Reales de Confusión (Target × Prediction)\n")
    md.write("### BASE DE FÁBRICA\n```text\n")
    md.write("        Pred A  Pred B  Pred C  Pred D\n")
    for g in ["A", "B", "C", "D"]:
        md.write(f"Gold {g}   {matriz_base[g]['A']:<8}{matriz_base[g]['B']:<8}{matriz_base[g]['C']:<8}{matriz_base[g]['D']:<8}\n")
    md.write("```\n\n### CHECKPOINT EVALUADO\n```text\n")
    md.write("        Pred A  Pred B  Pred C  Pred D\n")
    for g in ["A", "B", "C", "D"]:
        md.write(f"Gold {g}   {matriz_check[g]['A']:<8}{matriz_check[g]['B']:<8}{matriz_check[g]['C']:<8}{matriz_check[g]['D']:<8}\n")
    md.write("```\n")
    md.write("\n================================================\n")

print(f"\n[] ¡Logro exportado! Informe listo para la comunidad en: {ruta_informe}\n")


