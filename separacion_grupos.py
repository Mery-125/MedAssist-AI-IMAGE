import argparse
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una clasificación personalizada de 15 macroclases a partir de códigos ATC."
    )
    parser.add_argument("--json_path", required=True, help="Ruta al JSON de medicamentos.")
    parser.add_argument(
        "--output_dataset_csv",
        default="dataset_macroclase_15.csv",
        help="CSV de salida con una fila por medicamento y su macroclase.",
    )
    parser.add_argument(
        "--output_summary_csv",
        default="resumen_macroclase_15.csv",
        help="CSV de salida con el número de medicamentos por macroclase.",
    )
    return parser.parse_args()


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("results", "medicamentos", "items", "data"):
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]

    raise ValueError("No se encontró una lista de medicamentos en el JSON.")


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def get_best_atc_entry(med: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Devuelve el ATC más específico disponible.
    Preferencia: nivel 5 > 4 > 3 > 2.
    """
    atcs = med.get("atcs", [])
    if not isinstance(atcs, list):
        return None, None, None

    best_code = None
    best_name = None
    best_level = -1

    for atc in atcs:
        if not isinstance(atc, dict):
            continue

        nivel = atc.get("nivel")
        codigo = safe_str(atc.get("codigo"))
        nombre = safe_str(atc.get("nombre"))

        if codigo is None or nivel is None:
            continue

        try:
            nivel = int(nivel)
        except (TypeError, ValueError):
            continue

        if nivel > best_level:
            best_level = nivel
            best_code = codigo.upper()
            best_name = nombre

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
    """
    Mapeo de 15 macroclases:

    1. dolor_anestesia -> N01, N02
    2. neurologia_psiquiatria -> N03, N04, N05, N06, N07
    3. cardiovascular -> C
    4. oncologia -> L01, L02
    5. inmunologia_biologicos -> L03, L04
    6. antiinfecciosos_sistemicos -> J + P
    7. digestivo -> A01-A09
    8. metabolismo_diabetes_nutricion -> A10-A16
    9. hematologia_sangre -> B
    10. respiratorio -> R
    11. musculoesqueletico -> M
    12. genitourinario_hormonas_sexuales -> G
    13. dermatologia -> D
    14. organos_sentidos -> S
    15. hormonal_sistemico_y_varios_diagnosticos -> H + V
    """
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


def build_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for med in records:
        nombre_medicamento = safe_str(med.get("nombre"))
        nregistro = safe_str(med.get("nregistro"))

        atc_code, atc_name, atc_level = get_best_atc_entry(med)
        atc1, atc2 = derive_atc1_atc2(atc_code)
        macroclass = map_macroclass_15(atc1, atc2)
        macroclass_name = macroclass_description(macroclass)

        rows.append(
            {
                "nregistro": nregistro,
                "nombre_medicamento": nombre_medicamento,
                "codigo_atc_original": atc_code,
                "nombre_atc_original": atc_name,
                "nivel_atc_original": atc_level,
                "atc1_codigo": atc1,
                "atc2_codigo": atc2,
                "macroclase_15_codigo": macroclass,
                "macroclase_15_nombre": macroclass_name,
            }
        )

    return rows


def save_dataset(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "nregistro",
        "nombre_medicamento",
        "codigo_atc_original",
        "nombre_atc_original",
        "nivel_atc_original",
        "atc1_codigo",
        "atc2_codigo",
        "macroclase_15_codigo",
        "macroclase_15_nombre",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    counter = Counter()
    examples = defaultdict(list)

    for row in rows:
        key = (
            row["macroclase_15_codigo"],
            row["macroclase_15_nombre"],
        )
        counter[key] += 1

        nombre = row.get("nombre_medicamento")
        if nombre and len(examples[key]) < 5:
            examples[key].append(nombre)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "macroclase_15_codigo",
                "macroclase_15_nombre",
                "num_medicamentos",
                "ejemplo_1",
                "ejemplo_2",
                "ejemplo_3",
                "ejemplo_4",
                "ejemplo_5",
            ]
        )

        for key, count in counter.most_common():
            codigo, nombre = key
            ex = examples[key] + [""] * (5 - len(examples[key]))
            writer.writerow([codigo, nombre, count] + ex[:5])


def print_summary(rows: List[Dict[str, Any]]) -> None:
    counter = Counter()

    for row in rows:
        key = (
            row["macroclase_15_codigo"],
            row["macroclase_15_nombre"],
        )
        counter[key] += 1

    print("=" * 70)
    print("RESUMEN DE MACROCLASES (15 GRUPOS)")
    print("=" * 70)
    print(f"Medicamentos procesados: {len(rows)}")
    print(f"Número de macroclases encontradas: {sum(1 for k in counter if k[0] is not None)}")
    print()

    for (codigo, nombre), count in counter.most_common():
        print(f"{codigo} -> {nombre}: {count}")


def main() -> None:
    args = parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    output_dataset_csv = Path(args.output_dataset_csv).expanduser().resolve()
    output_summary_csv = Path(args.output_summary_csv).expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"No existe el JSON: {json_path}")

    output_dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(json_path)
    rows = build_rows(records)

    save_dataset(rows, output_dataset_csv)
    save_summary(rows, output_summary_csv)
    print_summary(rows)

    print()
    print(f"CSV dataset guardado en: {output_dataset_csv}")
    print(f"CSV resumen guardado en: {output_summary_csv}")


if __name__ == "__main__":
    main()