import argparse
import json
import re
import sys
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


COLUMNS = [
    "sample_id",
    "image_path",
    "image_filename",
    "image_type_raw",
    "image_type",
    "nregistro",
    "nombre",
    "nombre_normalizado",
    "principio_activo_texto",
    "principio_activo_normalizado",
    "num_principios_activos",
    "codigo_atc_original",
    "nombre_atc_original",
    "nivel_atc_original",
    "atc1_codigo",
    "atc2_codigo",
    "atc3_codigo",
    "atc3_nombre",
    "atc4_codigo",
    "atc4_nombre",
    "atc5_codigo",
    "atc5_nombre",
    "macroclase_15_codigo",
    "macroclase_15_nombre",
    "dosis",
    "forma_farmaceutica",
    "forma_farmaceutica_simplificada",
    "via_administracion",
    "labtitular",
    "labcomercializador",
    "cpresc",
    "receta",
    "generico",
    "comerc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye un dataset tabular (una fila por imagen) a partir de un JSON "
            "de medicamentos CIMA/AEMPS y una carpeta de imágenes locales."
        )
    )
    parser.add_argument("--json_path", required=True, help="Ruta al JSON de medicamentos.")
    parser.add_argument("--images_dir", required=True, help="Carpeta con imágenes locales.")
    parser.add_argument("--output_csv", required=True, help="Ruta del CSV principal de salida.")
    parser.add_argument(
        "--min_samples_nombre",
        type=int,
        default=5,
        help="Mínimo de muestras por nombre_normalizado para dataset_nombre_frecuente.csv (default: 5).",
    )
    return parser.parse_args()


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_text(value: Any) -> Optional[str]:
    text = safe_str(value)
    if text is None:
        return None

    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def extract_named_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        return safe_str(value.get("nombre") or value.get("name"))
    return safe_str(value)


def map_image_type(image_type_raw: Optional[str]) -> Optional[str]:
    if image_type_raw is None:
        return None

    t = image_type_raw.strip().lower()
    mapping = {
        "materialas": "caja",
        "formafarmac": "pastilla",
    }
    return mapping.get(t, t)


def parse_json_records(json_path: Path) -> List[Dict[str, Any]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No existe el archivo JSON: {json_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido en {json_path}: {exc}") from exc

    if isinstance(data, list):
        records = [x for x in data if isinstance(x, dict)]
        return records
    if isinstance(data, dict):
        for key in ("results", "medicamentos", "items", "data"):
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]
        raise ValueError(
            "El JSON es un diccionario pero no contiene una lista reconocible de medicamentos."
        )

    raise ValueError("El JSON debe ser una lista de diccionarios o un diccionario con lista interna.")


def _extract_text_from_obj(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return safe_str(obj)
    if isinstance(obj, (int, float, bool)):
        return safe_str(obj)
    if isinstance(obj, dict):
        for key in (
            "nombre",
            "name",
            "denominacion",
            "descripcion",
            "texto",
            "principioActivo",
            "principio_activo",
            "valor",
            "value",
        ):
            if key in obj and safe_str(obj.get(key)) is not None:
                return safe_str(obj.get(key))
        return None
    return safe_str(obj)


def _split_possible_multi_text(text: str) -> List[str]:
    parts = re.split(r"\s*\+\s*|;|,|/|\|", text)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned if cleaned else [text.strip()]


def extract_principio_activo(med: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
    source = med.get("principiosActivos")
    if source is None or (isinstance(source, list) and len(source) == 0):
        source = med.get("pactivos")

    if source is None:
        return None, None, 0

    items: List[str] = []

    if isinstance(source, list):
        for elem in source:
            txt = _extract_text_from_obj(elem)
            if txt:
                items.extend(_split_possible_multi_text(txt))
    else:
        txt = _extract_text_from_obj(source)
        if txt:
            items.extend(_split_possible_multi_text(txt))

    seen = set()
    ordered: List[str] = []
    for item in items:
        s = safe_str(item)
        if not s:
            continue
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            ordered.append(s.strip())

    if not ordered:
        return None, None, 0

    normalized_items: List[str] = []
    for item in ordered:
        n = normalize_text(item)
        if n:
            normalized_items.append(n)

    texto = " + ".join(ordered) if ordered else None
    texto_norm = " + ".join(normalized_items) if normalized_items else None
    num = len(normalized_items) if normalized_items else len(ordered)

    return texto, texto_norm, num


def infer_atc_level(code: Optional[str]) -> Optional[int]:
    """
    Fallback cuando no viene 'nivel' en el JSON.
    Longitudes observadas:
    - len 3 -> nivel 2
    - len 4 -> nivel 3
    - len 5 -> nivel 4
    - len 7 -> nivel 5
    """
    if not code:
        return None
    code = code.strip().upper()
    if len(code) == 3:
        return 2
    if len(code) == 4:
        return 3
    if len(code) == 5:
        return 4
    if len(code) == 7:
        return 5
    return None


def parse_atcs(atcs_value: Any) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    result = {
        2: (None, None),
        3: (None, None),
        4: (None, None),
        5: (None, None),
    }

    atc_items: List[Any] = []
    if isinstance(atcs_value, list):
        atc_items = atcs_value
    elif isinstance(atcs_value, dict):
        atc_items = list(atcs_value.values())
    elif atcs_value is not None:
        atc_items = [atcs_value]

    for item in atc_items:
        code = None
        name = None
        level = None

        if isinstance(item, dict):
            code = safe_str(item.get("codigo") or item.get("code") or item.get("atc"))
            name = safe_str(item.get("nombre") or item.get("name") or item.get("descripcion"))
            raw_level = item.get("nivel") or item.get("level")
            if raw_level is not None:
                try:
                    level = int(raw_level)
                except (TypeError, ValueError):
                    level = None

            if level is None:
                level = infer_atc_level(code)

        else:
            code = safe_str(item)
            level = infer_atc_level(code)
            name = None

        if level in (2, 3, 4, 5):
            prev_code, prev_name = result[level]
            if prev_code is None and code is not None:
                prev_code = code
            if prev_name is None and name is not None:
                prev_name = name
            result[level] = (prev_code, prev_name)

    return result


def get_best_atc_entry(atcs_value: Any) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Devuelve el ATC más específico disponible.
    Preferencia: nivel 5 > 4 > 3 > 2.
    """
    atcs = ensure_list(atcs_value)

    best_code = None
    best_name = None
    best_level = -1

    for item in atcs:
        code = None
        name = None
        level = None

        if isinstance(item, dict):
            code = safe_str(item.get("codigo") or item.get("code") or item.get("atc"))
            name = safe_str(item.get("nombre") or item.get("name") or item.get("descripcion"))
            raw_level = item.get("nivel") or item.get("level")
            if raw_level is not None:
                try:
                    level = int(raw_level)
                except (TypeError, ValueError):
                    level = None
        else:
            code = safe_str(item)

        if level is None:
            level = infer_atc_level(code)

        if code is not None and level is not None and level > best_level:
            best_level = level
            best_code = code.upper()
            best_name = name

    if best_level == -1:
        return None, None, None

    return best_code, best_name, best_level


def derive_atc1_atc2(code: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not code:
        return None, None

    code = code.strip().upper()
    atc1 = code[0] if len(code) >= 1 else None
    atc2 = code[:3] if len(code) >= 3 else None
    return atc1, atc2


def map_macroclass_15(atc1: Optional[str], atc2: Optional[str]) -> Optional[str]:
    if atc2 in {"N01", "N02"}:
        return "dolor_anestesia"

    if atc2 in {"N03", "N04", "N05", "N06", "N07"}:
        return "neurologia_psiquiatria"

    if atc1 == "C":
        return "cardiovascular"

    if atc2 in {"L01", "L02"}:
        return "oncologia"

    if atc2 in {"L03", "L04"}:
        return "inmunologia_biologicos"

    if atc1 in {"J", "P"}:
        return "antiinfecciosos_sistemicos"

    if atc2 is not None and atc2.startswith("A"):
        try:
            number = int(atc2[1:3])
        except ValueError:
            number = None

        if number is not None:
            if 1 <= number <= 9:
                return "digestivo"
            if 10 <= number <= 16:
                return "metabolismo_diabetes_nutricion"

    if atc1 == "B":
        return "hematologia_sangre"

    if atc1 == "R":
        return "respiratorio"

    if atc1 == "M":
        return "musculoesqueletico"

    if atc1 == "G":
        return "genitourinario_hormonas_sexuales"

    if atc1 == "D":
        return "dermatologia"

    if atc1 == "S":
        return "organos_sentidos"

    if atc1 in {"H", "V"}:
        return "hormonal_sistemico_y_varios_diagnosticos"

    return None


def macroclass_description(macroclass: Optional[str]) -> Optional[str]:
    descriptions = {
        "dolor_anestesia": "Dolor y anestesia",
        "neurologia_psiquiatria": "Neurología y psiquiatría",
        "cardiovascular": "Cardiovascular",
        "oncologia": "Oncología",
        "inmunologia_biologicos": "Inmunología y biológicos",
        "antiinfecciosos_sistemicos": "Antiinfecciosos sistémicos",
        "digestivo": "Digestivo",
        "metabolismo_diabetes_nutricion": "Metabolismo, diabetes y nutrición",
        "hematologia_sangre": "Hematología y sangre",
        "respiratorio": "Respiratorio",
        "musculoesqueletico": "Musculoesquelético",
        "genitourinario_hormonas_sexuales": "Genitourinario y hormonas sexuales",
        "dermatologia": "Dermatología",
        "organos_sentidos": "Órganos de los sentidos",
        "hormonal_sistemico_y_varios_diagnosticos": "Hormonal sistémico y varios diagnósticos",
    }
    return descriptions.get(macroclass)


def extract_via_administracion(med: Dict[str, Any]) -> Optional[str]:
    vias = med.get("viasAdministracion")
    vias_list = ensure_list(vias)

    values: List[str] = []
    for v in vias_list:
        if isinstance(v, dict):
            txt = _extract_text_from_obj(v)
        else:
            txt = safe_str(v)
        if txt:
            values.append(txt)

    if not values:
        return None

    seen = set()
    ordered = []
    for value in values:
        key = value.strip().lower()
        if key not in seen:
            seen.add(key)
            ordered.append(value.strip())

    return " + ".join(ordered) if ordered else None


def extract_image_type_raw(photo_obj: Any) -> Optional[str]:
    if photo_obj is None:
        return None

    if isinstance(photo_obj, str):
        text = photo_obj.lower()
        if "materialas" in text:
            return "materialas"
        if "formafarmac" in text:
            return "formafarmac"
        return safe_str(photo_obj)

    if isinstance(photo_obj, dict):
        for key in ("tipo", "tipoFoto", "type", "categoria", "category", "clase"):
            value = safe_str(photo_obj.get(key))
            if value:
                return value.lower()

        joined = " ".join(
            str(v).lower() for v in photo_obj.values() if isinstance(v, (str, int, float))
        )
        if "materialas" in joined:
            return "materialas"
        if "formafarmac" in joined:
            return "formafarmac"

    return None


def index_local_images(images_dir: Path) -> Tuple[Dict[Tuple[str, str], List[Path]], Dict[str, List[Path]]]:
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    key_index: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    reg_index: Dict[str, List[Path]] = defaultdict(list)

    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_ext:
            continue

        name_low = path.name.lower()

        reg_match = re.search(r"__(\d{3,})__", name_low)
        reg = reg_match.group(1) if reg_match else None

        if reg:
            reg_index[reg].append(path)

            image_type = None
            if "__materialas__" in name_low:
                image_type = "materialas"
            elif "__formafarmac__" in name_low:
                image_type = "formafarmac"

            if image_type:
                key_index[(reg, image_type)].append(path)

    for k in key_index:
        key_index[k] = sorted(key_index[k], key=lambda p: p.name.lower())
    for r in reg_index:
        reg_index[r] = sorted(reg_index[r], key=lambda p: p.name.lower())

    return key_index, reg_index


def find_local_image(
    nregistro: Optional[str],
    image_type_raw: Optional[str],
    key_index: Dict[Tuple[str, str], List[Path]],
    reg_index: Dict[str, List[Path]],
) -> Optional[Path]:
    if not nregistro:
        return None

    reg = str(nregistro).strip()
    if not reg:
        return None

    type_key = image_type_raw.strip().lower() if isinstance(image_type_raw, str) else None

    if type_key:
        candidates = key_index.get((reg, type_key), [])
        if candidates:
            return candidates[0]

    reg_candidates = reg_index.get(reg, [])
    if type_key and reg_candidates:
        for p in reg_candidates:
            if type_key in p.name.lower():
                return p

    if reg_candidates:
        return reg_candidates[0]

    return None


def build_rows(
    records: List[Dict[str, Any]],
    key_index: Dict[Tuple[str, str], List[Path]],
    reg_index: Dict[str, List[Path]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_counter: Dict[Tuple[str, str], int] = defaultdict(int)

    for med in records:
        nregistro = safe_str(med.get("nregistro"))
        nombre = safe_str(med.get("nombre"))
        nombre_norm = normalize_text(nombre)

        principio_texto, principio_norm, num_principios = extract_principio_activo(med)

        atc_levels = parse_atcs(med.get("atcs"))
        atc3_codigo, atc3_nombre = atc_levels[3]
        atc4_codigo, atc4_nombre = atc_levels[4]
        atc5_codigo, atc5_nombre = atc_levels[5]

        codigo_atc_original, nombre_atc_original, nivel_atc_original = get_best_atc_entry(med.get("atcs"))
        atc1_codigo, atc2_codigo = derive_atc1_atc2(codigo_atc_original)
        macroclase_15_codigo = map_macroclass_15(atc1_codigo, atc2_codigo)
        macroclase_15_nombre = macroclass_description(macroclase_15_codigo)

        fotos = ensure_list(med.get("fotos"))
        if not fotos:
            continue

        used_paths_for_med = set()

        for foto in fotos:
            image_type_raw = extract_image_type_raw(foto)
            local_img = find_local_image(nregistro, image_type_raw, key_index, reg_index)

            if local_img is None:
                warnings.warn(
                    f"No se encontró imagen local para nregistro={nregistro}, tipo={image_type_raw}"
                )
                continue

            # Evitar duplicar exactamente la misma imagen en un mismo medicamento
            local_img_key = str(local_img.resolve())
            if local_img_key in used_paths_for_med:
                continue
            used_paths_for_med.add(local_img_key)

            image_type = map_image_type(image_type_raw)
            key = (nregistro or "sinregistro", image_type or "desconocido")
            sample_counter[key] += 1
            sample_id = f"{key[0]}_{key[1]}_{sample_counter[key]}"

            row = {
                "sample_id": sample_id,
                "image_path": str(local_img.resolve()),
                "image_filename": local_img.name,
                "image_type_raw": image_type_raw,
                "image_type": image_type,
                "nregistro": nregistro,
                "nombre": nombre,
                "nombre_normalizado": nombre_norm,
                "principio_activo_texto": principio_texto,
                "principio_activo_normalizado": principio_norm,
                "num_principios_activos": num_principios,
                "codigo_atc_original": codigo_atc_original,
                "nombre_atc_original": nombre_atc_original,
                "nivel_atc_original": nivel_atc_original,
                "atc1_codigo": atc1_codigo,
                "atc2_codigo": atc2_codigo,
                "atc3_codigo": atc3_codigo,
                "atc3_nombre": atc3_nombre,
                "atc4_codigo": atc4_codigo,
                "atc4_nombre": atc4_nombre,
                "atc5_codigo": atc5_codigo,
                "atc5_nombre": atc5_nombre,
                "macroclase_15_codigo": macroclase_15_codigo,
                "macroclase_15_nombre": macroclase_15_nombre,
                "dosis": safe_str(med.get("dosis")),
                "forma_farmaceutica": extract_named_field(med.get("formaFarmaceutica")),
                "forma_farmaceutica_simplificada": extract_named_field(med.get("formaFarmaceuticaSimplificada")),
                "via_administracion": extract_via_administracion(med),
                "labtitular": safe_str(med.get("labtitular")),
                "labcomercializador": safe_str(med.get("labcomercializador")),
                "cpresc": safe_str(med.get("cpresc")),
                "receta": med.get("receta"),
                "generico": med.get("generico"),
                "comerc": med.get("comerc"),
            }
            rows.append(row)

    return rows


def save_distribution(df: pd.DataFrame, column: str, output_path: Path) -> None:
    if column not in df.columns:
        out = pd.DataFrame(columns=[column, "num_imagenes"])
        out.to_csv(output_path, index=False, encoding="utf-8")
        return

    series = df[column].fillna("None").astype(str).value_counts(dropna=False)
    dist = series.rename_axis(column).reset_index(name="num_imagenes")
    dist.to_csv(output_path, index=False, encoding="utf-8")


def top_n_text(df: pd.DataFrame, column: str, n: int = 20) -> str:
    if column not in df.columns or df.empty:
        return "(sin datos)"
    counts = df[column].fillna("None").astype(str).value_counts().head(n)
    if counts.empty:
        return "(sin datos)"
    lines = [f"{idx}: {int(val)}" for idx, val in counts.items()]
    return "\n".join(lines)


def generate_additional_files(
    df: pd.DataFrame,
    output_csv: Path,
    total_records_read: int,
    min_samples_nombre: int,
) -> None:
    out_dir = output_csv.parent

    save_distribution(df, "macroclase_15_nombre", out_dir / "class_distribution_macroclase_15.csv")
    save_distribution(df, "atc1_codigo", out_dir / "class_distribution_atc1.csv")
    save_distribution(df, "atc3_nombre", out_dir / "class_distribution_atc3.csv")
    save_distribution(
        df,
        "principio_activo_normalizado",
        out_dir / "class_distribution_principio_activo.csv",
    )
    save_distribution(df, "nombre_normalizado", out_dir / "class_distribution_nombre.csv")

    if "nombre_normalizado" in df.columns and not df.empty:
        counts = df["nombre_normalizado"].dropna().astype(str).value_counts()
        valid_classes = set(counts[counts >= min_samples_nombre].index.tolist())
        dataset_nombre_frecuente = df[df["nombre_normalizado"].isin(valid_classes)].copy()
    else:
        dataset_nombre_frecuente = df.copy()

    dataset_nombre_frecuente.to_csv(
        out_dir / "dataset_nombre_frecuente.csv", index=False, encoding="utf-8"
    )

    dataset_caja = df[df["image_type"] == "caja"].copy()
    dataset_caja.to_csv(out_dir / "dataset_caja.csv", index=False, encoding="utf-8")

    dataset_pastilla = df[df["image_type"] == "pastilla"].copy()
    dataset_pastilla.to_csv(out_dir / "dataset_pastilla.csv", index=False, encoding="utf-8")

    num_total_images = int(len(df))
    num_caja = int((df["image_type"] == "caja").sum()) if "image_type" in df.columns else 0
    num_pastilla = int((df["image_type"] == "pastilla").sum()) if "image_type" in df.columns else 0

    n_macro = int(df["macroclase_15_nombre"].nunique(dropna=True)) if "macroclase_15_nombre" in df.columns else 0
    n_atc3 = int(df["atc3_nombre"].nunique(dropna=True)) if "atc3_nombre" in df.columns else 0
    n_principio = (
        int(df["principio_activo_normalizado"].nunique(dropna=True))
        if "principio_activo_normalizado" in df.columns
        else 0
    )
    n_nombre = int(df["nombre_normalizado"].nunique(dropna=True)) if "nombre_normalizado" in df.columns else 0

    resumen = [
        "Resumen del dataset",
        "===================",
        f"Número total de registros leídos: {total_records_read}",
        f"Número total de imágenes encontradas: {num_total_images}",
        f"Número de imágenes de caja: {num_caja}",
        f"Número de imágenes de pastilla: {num_pastilla}",
        f"Número de clases únicas de macroclase_15_nombre: {n_macro}",
        f"Número de clases únicas de atc3_nombre: {n_atc3}",
        f"Número de clases únicas de principio_activo_normalizado: {n_principio}",
        f"Número de clases únicas de nombre_normalizado: {n_nombre}",
        "",
        "Top 20 clases más frecuentes de macroclase_15_nombre",
        "--------------------------------------------------",
        top_n_text(df, "macroclase_15_nombre", n=20),
        "",
        "Top 20 clases más frecuentes de atc3_nombre",
        "------------------------------------------",
        top_n_text(df, "atc3_nombre", n=20),
        "",
        "Top 20 clases más frecuentes de principio_activo_normalizado",
        "-------------------------------------------------------------",
        top_n_text(df, "principio_activo_normalizado", n=20),
        "",
        "Top 20 clases más frecuentes de nombre_normalizado",
        "---------------------------------------------------",
        top_n_text(df, "nombre_normalizado", n=20),
        "",
    ]
    (out_dir / "dataset_resumen.txt").write_text("\n".join(resumen), encoding="utf-8")


def main() -> None:
    args = parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"No existe --json_path: {json_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"No existe --images_dir o no es carpeta: {images_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    records = parse_json_records(json_path)
    key_index, reg_index = index_local_images(images_dir)
    rows = build_rows(records, key_index, reg_index)

    df = pd.DataFrame(rows, columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]

    df.to_csv(output_csv, index=False, encoding="utf-8")

    generate_additional_files(
        df=df,
        output_csv=output_csv,
        total_records_read=len(records),
        min_samples_nombre=args.min_samples_nombre,
    )

    n_total = len(df)
    n_caja = int((df["image_type"] == "caja").sum()) if "image_type" in df.columns else 0
    n_pastilla = int((df["image_type"] == "pastilla").sum()) if "image_type" in df.columns else 0
    n_macro = int(df["macroclase_15_nombre"].nunique(dropna=True)) if "macroclase_15_nombre" in df.columns else 0
    n_atc3 = int(df["atc3_nombre"].nunique(dropna=True)) if "atc3_nombre" in df.columns else 0
    n_prin = (
        int(df["principio_activo_normalizado"].nunique(dropna=True))
        if "principio_activo_normalizado" in df.columns
        else 0
    )
    n_nom = int(df["nombre_normalizado"].nunique(dropna=True)) if "nombre_normalizado" in df.columns else 0

    print("Dataset generado correctamente.")
    print(f"- Registros leídos (JSON): {len(records)}")
    print(f"- Imágenes encontradas (filas CSV): {n_total}")
    print(f"- Imágenes caja: {n_caja}")
    print(f"- Imágenes pastilla: {n_pastilla}")
    print(f"- Clases únicas macroclase_15_nombre: {n_macro}")
    print(f"- Clases únicas atc3_nombre: {n_atc3}")
    print(f"- Clases únicas principio_activo_normalizado: {n_prin}")
    print(f"- Clases únicas nombre_normalizado: {n_nom}")
    print(f"- CSV principal: {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)