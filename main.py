from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import requests
import json
import os
import io
import re
import math
import warnings
import unicodedata
from collections import defaultdict
from datetime import datetime
from typing import Optional, Any

import pikepdf
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

# -------------------------------------------------------------------
# Oracle: fallback seguro para que la app no explote si falta conexion.py
# -------------------------------------------------------------------
try:
    from conexion import OracleEnterpriseConnection
except Exception:
    class OracleEnterpriseConnection:
        def connect(self):
            return False

        def execute_query(self, query, params=None, fetch_size=1000):
            return None

        def close_connection(self):
            pass


load_dotenv()

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
app = FastAPI(title="LabAnalyzer v2.0")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# ENV helpers
# -------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = _env_int("OLLAMA_TIMEOUT", 180)
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

PDF_MIN_CHARS = 80

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------
class BuscarArchivosRequest(BaseModel):
    identificacion: str


class AnalyzeRequest(BaseModel):
    url: str
    id_usuario: int
    id_archivo: Optional[int] = None
    password: Optional[str] = None
    output_lang: Optional[str] = "es"
    force_reprocess: Optional[bool] = False


class SaveAnalysisRequest(BaseModel):
    id_archivo: int
    nombre_alternativo: str
    analisis: str


class EvolucionRequest(BaseModel):
    id_usuario: int


# -------------------------------------------------------------------
# SQL
# -------------------------------------------------------------------
SQL_ARCHIVOS = """
    SELECT
        ar.id,
        ar.fecha_cargue,
        ar.id_usuario,
        'http://tekerapp.maxapex.net/FILES_PROD_TEKER_NEW/' || ar.nombre_archivo_almacenado AS url,
        ar.nombre_archivo,
        us.identificacion,
        ar.nombre_alternativo,
        CASE WHEN ar.analisis IS NOT NULL THEN 1 ELSE 0 END AS ya_analizado,
        ar.analisis
    FROM tkr_archivos ar, tkr_usuarios us
    WHERE ar.id_descripcion IN (1, 2, 4)
      AND ar.id_usuario = us.id
      AND us.identificacion = :identificacion
    ORDER BY ar.fecha_cargue DESC
"""

SQL_SAVE_ANALYSIS = """
    UPDATE tkr_archivos
    SET nombre_alternativo = :nombre_alternativo,
        analisis           = :analisis
    WHERE id = :id_archivo
"""

SQL_CHECK_ANALIZADO = """
    SELECT analisis
    FROM tkr_archivos
    WHERE id = :id_archivo
      AND analisis IS NOT NULL
"""

SQL_ANALISIS_USUARIO = """
    SELECT analisis, nombre_alternativo, id, fecha_cargue
    FROM tkr_archivos
    WHERE id_usuario = :id_usuario
      AND analisis IS NOT NULL
    ORDER BY fecha_cargue ASC
"""

# -------------------------------------------------------------------
# Utilidades generales
# -------------------------------------------------------------------
def _safe_clob_to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.read() if hasattr(value, "read") else str(value)
    except Exception:
        return str(value)


def _strip_accents(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _norm_text(s: str) -> str:
    s = _strip_accents(str(s or "")).upper().strip()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _slug_text(s: str) -> str:
    s = _strip_accents(str(s or "")).strip()
    s = re.sub(r"[^a-zA-Z0-9\s_-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s or "Examen"


def _json_default(obj):
    if isinstance(obj, (datetime,)):
        return obj.strftime("%Y%m%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return str(obj)


def _parse_fecha(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    s = str(value).strip()
    if not s:
        return None

    # Solo fecha desde datetime string
    s = s.replace("T", " ").strip()
    s_date = s.split(" ")[0]

    patterns = [
        "%Y%m%d",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
    ]

    for p in patterns:
        try:
            return datetime.strptime(s_date, p)
        except Exception:
            pass

    # Oracle-like or ISO more flexible
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s_date)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None

    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", s_date)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None

    m = re.match(r"^(\d{2})[-/](\d{2})[-/](\d{4})$", s_date)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except Exception:
            return None

    return None


def _fecha_to_yyyymmdd(value: Any) -> Optional[str]:
    dt = _parse_fecha(value)
    return dt.strftime("%Y%m%d") if dt else None


def _fecha_display(value: Any) -> str:
    dt = _parse_fecha(value)
    return dt.strftime("%d/%m/%Y") if dt else ""


def _extraer_valor_numerico(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except Exception:
            return None

    s = str(value).strip().lower()
    if not s or s in {"negativo", "positivo", "normal", "sin dato", "n/a", "na"}:
        return None

    # Reemplazar coma decimal
    s = s.replace(",", ".")

    # Casos tipo <5 o >10
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def build_nombre_alternativo(tipo_examen: str, fecha: Any) -> str:
    nombre = _slug_text(tipo_examen or "Examen")
    fecha_part = _fecha_to_yyyymmdd(fecha) or datetime.now().strftime("%Y%m%d")
    return f"{nombre}_{fecha_part}"


# -------------------------------------------------------------------
# Clasificación simple
# -------------------------------------------------------------------
_CLASIF_KEYWORDS = {
    "LAB": ["HEMOGRAMA", "HEMATOLOGIA", "QUIMICA", "GLUCOSA", "CREATININA", "TSH", "PERFIL", "ORINA"],
    "RAD": ["RX", "RADIOGRAFIA", "TAC", "TOMOGRAFIA", "ECO", "ECOGRAFIA", "RESONANCIA", "MAMOGRAFIA"],
    "PAT": ["BIOPSIA", "CITOLOGIA", "PAPANICOLAU", "PATOLOGIA"],
    "CARD": ["ECG", "ELECTROCARDIOGRAMA", "HOLTER", "ECOCARDIOGRAMA", "PRUEBA DE ESFUERZO"],
}

_DISCIPLINA = {
    "LAB": "Laboratorio Clínico",
    "RAD": "Imagenología",
    "PAT": "Patología",
    "CARD": "Cardiología Diagnóstica",
    "OT": "Otros",
}


def clasificar_examen(tipo_examen: str) -> dict:
    txt = _norm_text(tipo_examen)
    for cat, words in _CLASIF_KEYWORDS.items():
        if any(w in txt for w in words):
            return {
                "disciplina": _DISCIPLINA[cat],
                "subcategoria": tipo_examen,
                "fhir_category": cat,
            }
    return {
        "disciplina": "Laboratorio Clínico",
        "subcategoria": tipo_examen or "General",
        "fhir_category": "LAB",
    }


# -------------------------------------------------------------------
# CUPS catalog
# -------------------------------------------------------------------
def _load_cups_catalog() -> pd.DataFrame:
    base = os.path.dirname(os.path.abspath(__file__))
    files = [
        os.path.join(base, "BIOPSIAS.XLSX"),
        os.path.join(base, "parear_procedimientos.xlsx"),
    ]
    frames = []

    for path in files:
        if os.path.exists(path):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    df = pd.read_excel(path)
                if "CODIGO_CUPS" in df.columns and "NOMBRE_PROCEDIMIENTO" in df.columns:
                    frames.append(df[["CODIGO_CUPS", "NOMBRE_PROCEDIMIENTO"]].copy())
            except Exception as e:
                print(f"[CUPS] Error leyendo {path}: {e}")

    if not frames:
        return pd.DataFrame(columns=["CODIGO_CUPS", "NOMBRE_PROCEDIMIENTO", "_norm"])

    cat = pd.concat(frames, ignore_index=True).drop_duplicates()
    cat["CODIGO_CUPS"] = cat["CODIGO_CUPS"].astype(str).str.strip()
    cat["NOMBRE_PROCEDIMIENTO"] = cat["NOMBRE_PROCEDIMIENTO"].astype(str).str.strip()
    cat["_norm"] = cat["NOMBRE_PROCEDIMIENTO"].apply(_norm_text)
    return cat


_CUPS_CATALOG = _load_cups_catalog()
_CUPS_VEC = None
_CUPS_MAT = None

if not _CUPS_CATALOG.empty:
    _CUPS_VEC = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", min_df=1)
    _CUPS_MAT = _CUPS_VEC.fit_transform(_CUPS_CATALOG["_norm"])


def asignar_cups(tipo_examen: str, codigo_cups_llm: Optional[str] = None, score_min: float = 0.20) -> dict:
    empty = {
        "codigo_cups": None,
        "nombre_cups": None,
        "cups_score": 0.0,
        "cups_fuente": "sin_match",
    }

    if _CUPS_CATALOG.empty or _CUPS_VEC is None:
        return empty

    if codigo_cups_llm:
        hit = _CUPS_CATALOG[_CUPS_CATALOG["CODIGO_CUPS"] == str(codigo_cups_llm).strip()]
        if not hit.empty:
            r = hit.iloc[0]
            return {
                "codigo_cups": str(r["CODIGO_CUPS"]),
                "nombre_cups": str(r["NOMBRE_PROCEDIMIENTO"]),
                "cups_score": 1.0,
                "cups_fuente": "llm",
            }

    q = _norm_text(tipo_examen)
    if not q:
        return empty

    sims = cosine_similarity(_CUPS_VEC.transform([q]), _CUPS_MAT).flatten()
    idx = int(np.argmax(sims))
    score = float(sims[idx])

    if score < score_min:
        return {**empty, "cups_score": round(score, 3)}

    r = _CUPS_CATALOG.iloc[idx]
    return {
        "codigo_cups": str(r["CODIGO_CUPS"]),
        "nombre_cups": str(r["NOMBRE_PROCEDIMIENTO"]),
        "cups_score": round(score, 3),
        "cups_fuente": "catalogo",
    }


# -------------------------------------------------------------------
# Archivo / OCR / PDF
# -------------------------------------------------------------------
def download_file(url: str) -> tuple[bytes, str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LabAnalyzer/2.0)"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error descargando archivo: {e}")


def get_file_extension(content_type: str, url: str = "") -> str:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return "pdf"
    if "png" in ct:
        return "png"
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"

    u = (url or "").lower().split("?")[0]
    if u.endswith(".pdf"):
        return "pdf"
    if u.endswith(".png"):
        return "png"
    if u.endswith(".jpg") or u.endswith(".jpeg"):
        return "jpg"
    return "jpg"


def is_pdf_encrypted(file_bytes: bytes) -> bool:
    try:
        with pikepdf.open(io.BytesIO(file_bytes)):
            return False
    except pikepdf.PasswordError:
        return True
    except Exception:
        return False


def unlock_pdf(file_bytes: bytes, password: str) -> bytes:
    try:
        out = io.BytesIO()
        with pikepdf.open(io.BytesIO(file_bytes), password=password) as pdf:
            pdf.save(out)
        return out.getvalue()
    except pikepdf.PasswordError:
        raise HTTPException(status_code=400, detail="Contraseña incorrecta para el PDF")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error desbloqueando PDF: {e}")


def extract_text_native_pdf(file_bytes: bytes) -> str:
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = (page.extract_text() or "").strip()
                if txt:
                    parts.append(f"[PAGINA_{i}]\n{txt}")
        return "\n\n".join(parts).strip()
    except Exception as e:
        print(f"[pdfplumber] Error: {e}")
        return ""


def extract_text_ocr(file_bytes: bytes, content_type: str, url_file: str) -> str:
    if not OCR_SPACE_API_KEY:
        raise HTTPException(status_code=500, detail="Falta OCR_SPACE_API_KEY en .env")

    ext = get_file_extension(content_type, url_file)
    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (f"document.{ext}", file_bytes)},
            data={
                "apikey": OCR_SPACE_API_KEY,
                "language": "spa",
                "isOverlayRequired": False,
                "OCREngine": 2,
                "scale": True,
                "isTable": True,
                "filetype": ext,
            },
            timeout=60,
        )
        result = response.json()
        if result.get("IsErroredOnProcessing"):
            raise HTTPException(status_code=400, detail=f"OCR error: {result.get('ErrorMessage')}")
        parsed = result.get("ParsedResults") or []
        return "\n".join(p.get("ParsedText", "") for p in parsed).strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error OCR.Space: {e}")


def extract_text(file_bytes: bytes, content_type: str, url_file: str) -> str:
    ext = get_file_extension(content_type, url_file)
    if ext == "pdf":
        native = extract_text_native_pdf(file_bytes)
        if len(native) >= PDF_MIN_CHARS:
            return native
    return extract_text_ocr(file_bytes, content_type, url_file)


# -------------------------------------------------------------------
# Ollama
# -------------------------------------------------------------------
def call_ollama_json(prompt: str, system: str = "") -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("response") or "").strip()
        return json.loads(clean_json(raw))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Ollama: {e}")


def clean_json(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return raw
    start_obj = raw.find("{")
    end_obj = raw.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return raw[start_obj:end_obj + 1]
    return raw


PROMPT_ANALISIS = """
Devuelve SOLO un JSON válido con esta estructura:
{
  "profesional_documento": {"nombre": null, "cargo": null, "registro": null},
  "institucion_documento": null,
  "examenes": [
    {
      "fecha": "YYYYMMDD",
      "tipo_examen": "string",
      "disciplina": "string",
      "subcategoria": "string",
      "fhir_category": "LAB|RAD|PAT|CARD|OT",
      "region_anatomica": null,
      "codigo_cups": null,
      "fhir_imaging_study": null,
      "profesional": null,
      "cargo_profesional": null,
      "registro_profesional": null,
      "institucion": null,
      "resultados": [
        {
          "parametro": "string",
          "valor": "string",
          "unidad": null,
          "referencia": null,
          "estado": "normal|alto|bajo|sin_dato",
          "metodo": null,
          "codigo_cups": null,
          "nota": null
        }
      ],
      "notas_clinicas": [],
      "notas_medico": null,
      "notas_paciente": null
    }
  ]
}

Reglas:
- Si no hay fecha clara, usa null.
- La fecha DEBE ir en YYYYMMDD.
- Si hay varios exámenes, incluye varios objetos.
- Si no sabes disciplina/fhir_category, infiérela.
- Devuelve solamente JSON.
"""


def normalize_to_multi(raw_parsed, id_usuario: int, id_archivo: Optional[int]) -> dict:
    if isinstance(raw_parsed, list):
        raw_parsed = {
            "profesional_documento": None,
            "institucion_documento": None,
            "examenes": raw_parsed,
        }

    if not isinstance(raw_parsed, dict):
        raw_parsed = {"profesional_documento": None, "institucion_documento": None, "examenes": []}

    prof_doc = raw_parsed.get("profesional_documento") or {}
    inst_doc = raw_parsed.get("institucion_documento")
    examenes = raw_parsed.get("examenes") or []

    fixed = []
    for exam in examenes:
        if not isinstance(exam, dict):
            continue

        if not exam.get("disciplina") or not exam.get("fhir_category"):
            clasif = clasificar_examen(exam.get("tipo_examen", ""))
            exam["disciplina"] = exam.get("disciplina") or clasif["disciplina"]
            exam["subcategoria"] = exam.get("subcategoria") or clasif["subcategoria"]
            exam["fhir_category"] = exam.get("fhir_category") or clasif["fhir_category"]

        exam["fecha"] = _fecha_to_yyyymmdd(exam.get("fecha"))

        if not exam.get("profesional") and isinstance(prof_doc, dict):
            exam["profesional"] = prof_doc.get("nombre")
            exam["cargo_profesional"] = exam.get("cargo_profesional") or prof_doc.get("cargo")
            exam["registro_profesional"] = exam.get("registro_profesional") or prof_doc.get("registro")

        if not exam.get("institucion"):
            exam["institucion"] = inst_doc

        cups_exam = asignar_cups(exam.get("tipo_examen", ""), exam.get("codigo_cups"))
        exam["codigo_cups"] = cups_exam["codigo_cups"]
        exam["nombre_cups"] = cups_exam["nombre_cups"]
        exam["cups_score"] = cups_exam["cups_score"]
        exam["cups_fuente"] = cups_exam["cups_fuente"]

        resultados = exam.get("resultados") or []
        fixed_results = []
        for r in resultados:
            if not isinstance(r, dict):
                continue
            if exam.get("fhir_category") == "LAB":
                cups_param = asignar_cups(r.get("parametro", ""), r.get("codigo_cups"), score_min=0.25)
                r["codigo_cups"] = cups_param["codigo_cups"]
                r["nombre_cups"] = cups_param["nombre_cups"]
                r["cups_score"] = cups_param["cups_score"]
                r["cups_fuente"] = cups_param["cups_fuente"]
            else:
                r["codigo_cups"] = None
                r["nombre_cups"] = None
                r["cups_score"] = 0.0
                r["cups_fuente"] = "n/a"
            fixed_results.append(r)

        exam["resultados"] = fixed_results

        if exam.get("fhir_category") == "RAD" and not exam.get("fhir_imaging_study"):
            tipo = _norm_text(exam.get("tipo_examen", ""))
            if "RX" in tipo or "RADIOGRAFIA" in tipo:
                mod = "DX"
            elif "TAC" in tipo or "TOMOGRAFIA" in tipo:
                mod = "CT"
            elif "RESONANCIA" in tipo:
                mod = "MR"
            elif "ECO" in tipo or "ECOGRAFIA" in tipo:
                mod = "US"
            elif "MAMOGRAFIA" in tipo:
                mod = "MG"
            else:
                mod = "OT"
            exam["fhir_imaging_study"] = {
                "modalidad_dicom": mod,
                "region_anatomica": exam.get("region_anatomica"),
            }
        else:
            exam.setdefault("fhir_imaging_study", None)

        exam.setdefault("notas_clinicas", [])
        fixed.append(exam)

    return {
        "profesional_documento": prof_doc if prof_doc else None,
        "institucion_documento": inst_doc,
        "examenes": fixed,
        "id_usuario": id_usuario,
        "id_archivo": id_archivo,
        "multi": len(fixed) > 1,
    }


# -------------------------------------------------------------------
# Evolución clínica
# -------------------------------------------------------------------
def _normalizar_grupo(nombre: str) -> str:
    txt = _norm_text(nombre or "EXAMEN")
    return txt[:60] if txt else "EXAMEN"


def _calcular_tendencia(puntos: list[dict]) -> dict:
    if len(puntos) < 2:
        return {
            "tipo": "sin_datos",
            "direccion": "sin_dato",
            "delta_absoluto": None,
            "delta_porcentual": None,
            "pendiente": None,
            "r2": None,
            "interpretacion": "No hay suficientes puntos.",
        }

    puntos_ordenados = sorted(
        [p for p in puntos if p.get("fecha_dt") and p.get("valor") is not None],
        key=lambda x: x["fecha_dt"],
    )

    if len(puntos_ordenados) < 2:
        return {
            "tipo": "sin_datos",
            "direccion": "sin_dato",
            "delta_absoluto": None,
            "delta_porcentual": None,
            "pendiente": None,
            "r2": None,
            "interpretacion": "No hay suficientes puntos válidos.",
        }

    y = [float(p["valor"]) for p in puntos_ordenados]
    x = list(range(len(y)))

    delta_abs = y[-1] - y[0]
    delta_pct = None
    if y[0] not in (0, None):
        try:
            delta_pct = (delta_abs / y[0]) * 100.0
        except Exception:
            delta_pct = None

    if len(y) >= 3 and _scipy_stats is not None:
        try:
            lr = _scipy_stats.linregress(x, y)
            pendiente = float(lr.slope)
            r2 = float(lr.rvalue ** 2)
        except Exception:
            pendiente = None
            r2 = None
    else:
        pendiente = None
        r2 = None

    # Reglas simples
    base_abs = abs(y[0]) if y[0] not in (None, 0) else 1.0
    ratio = abs(delta_abs) / base_abs

    if delta_pct is not None and abs(delta_pct) < 5:
        direccion = "estable"
    elif ratio < 0.05:
        direccion = "estable"
    elif delta_abs > 0:
        direccion = "ascendente"
    elif delta_abs < 0:
        direccion = "descendente"
    else:
        direccion = "estable"

    tipo = "regresion_lineal" if len(y) >= 3 else "comparacion_simple"

    return {
        "tipo": tipo,
        "direccion": direccion,
        "delta_absoluto": round(delta_abs, 4),
        "delta_porcentual": round(delta_pct, 2) if delta_pct is not None else None,
        "pendiente": round(pendiente, 6) if pendiente is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
        "n": len(y),
    }


def _generar_narrativa_tendencia(parametro: str, tendencia: dict, unidad: Optional[str], referencia: Optional[str]) -> str:
    direccion = tendencia.get("direccion", "sin_dato")
    delta = tendencia.get("delta_porcentual")
    unidad_txt = f" {unidad}" if unidad else ""
    ref_txt = f" Referencia reportada: {referencia}." if referencia else ""

    if direccion == "ascendente":
        base = f"{parametro}: tendencia ascendente"
    elif direccion == "descendente":
        base = f"{parametro}: tendencia descendente"
    elif direccion == "estable":
        base = f"{parametro}: comportamiento globalmente estable"
    else:
        base = f"{parametro}: sin suficientes datos"

    if delta is not None:
        base += f" (variación {delta:+.2f}%{unidad_txt})."

    return base + ref_txt


def _resumen_global_narrativa(grupos: dict) -> str:
    frases = []
    for grupo, params in grupos.items():
        for parametro, info in params.items():
            frases.append(info.get("narrativa_parametro"))
    frases = [f for f in frases if f]
    if not frases:
        return "No se encontraron suficientes datos longitudinales para generar una narrativa clínica."
    return " ".join(frases[:12])


def construir_evolucion(rows, id_usuario: int) -> dict:
    grupos = defaultdict(dict)
    total_archivos = len(rows or [])

    for row in rows or []:
        analisis_raw, nombre_alt, id_archivo, fecha_cargue = row
        analisis_str = _safe_clob_to_str(analisis_raw)
        if not analisis_str:
            continue

        try:
            data = json.loads(analisis_str)
        except Exception:
            continue

        examenes = []
        if isinstance(data, dict) and "examenes" in data:
            examenes = data.get("examenes") or []
        elif isinstance(data, dict):
            examenes = [data]
        elif isinstance(data, list):
            examenes = data

        for exam in examenes:
            if not isinstance(exam, dict):
                continue

            grupo = _normalizar_grupo(
                exam.get("tipo_examen")
                or exam.get("subcategoria")
                or exam.get("disciplina")
                or nombre_alt
                or "EXAMEN"
            )

            fecha_exam = _fecha_to_yyyymmdd(exam.get("fecha")) or _fecha_to_yyyymmdd(fecha_cargue)
            fecha_dt = _parse_fecha(fecha_exam)
            if not fecha_exam or not fecha_dt:
                continue

            for res in exam.get("resultados", []) or []:
                if not isinstance(res, dict):
                    continue

                parametro = (res.get("parametro") or "").strip()
                if not parametro:
                    continue

                valor_num = _extraer_valor_numerico(res.get("valor"))
                if valor_num is None:
                    continue

                if parametro not in grupos[grupo]:
                    grupos[grupo][parametro] = {
                        "puntos": [],
                        "unidad": res.get("unidad"),
                        "referencia": res.get("referencia"),
                    }

                grupos[grupo][parametro]["puntos"].append({
                    "fecha": fecha_exam,
                    "fecha_dt": fecha_dt,
                    "valor": float(valor_num),
                    "valor_raw": res.get("valor"),
                    "id_archivo": int(id_archivo) if id_archivo is not None else None,
                    "estado": res.get("estado"),
                })

                if not grupos[grupo][parametro].get("unidad") and res.get("unidad"):
                    grupos[grupo][parametro]["unidad"] = res.get("unidad")
                if not grupos[grupo][parametro].get("referencia") and res.get("referencia"):
                    grupos[grupo][parametro]["referencia"] = res.get("referencia")

    # Filtrar solo parámetros con múltiples mediciones
    grupos_filtrados = {}
    for grupo, parametros in grupos.items():
        params_ok = {}
        for parametro, info in parametros.items():
            puntos = sorted(info["puntos"], key=lambda x: x["fecha_dt"])
            if len(puntos) < 2:
                continue

            tendencia = _calcular_tendencia(puntos)
            narrativa_parametro = _generar_narrativa_tendencia(
                parametro=parametro,
                tendencia=tendencia,
                unidad=info.get("unidad"),
                referencia=info.get("referencia"),
            )

            params_ok[parametro] = {
                "puntos": [
                    {
                        "fecha": p["fecha"],
                        "valor": p["valor"],
                        "valor_raw": p["valor_raw"],
                        "id_archivo": p["id_archivo"],
                        "estado": p["estado"],
                    }
                    for p in puntos
                ],
                "tendencia": tendencia,
                "unidad": info.get("unidad"),
                "referencia": info.get("referencia"),
                "narrativa_parametro": narrativa_parametro,
            }

        if params_ok:
            grupos_filtrados[grupo] = params_ok

    narrativa = _resumen_global_narrativa(grupos_filtrados)

    return {
        "grupos": grupos_filtrados,
        "narrativa": narrativa,
        "total_archivos": total_archivos,
        "id_usuario": id_usuario,
    }


# -------------------------------------------------------------------
# Rutas
# -------------------------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/proxy-file")
def proxy_file(url: str):
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LabAnalyzer/2.0)"},
            timeout=30,
            stream=True,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "application/octet-stream")
        if url.lower().split("?")[0].endswith(".pdf"):
            content_type = "application/pdf"

        def iter_content():
            for chunk in resp.iter_content(chunk_size=8192):
                yield chunk

        return StreamingResponse(
            iter_content(),
            media_type=content_type,
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "public, max-age=300",
                "X-Frame-Options": "",
                "Content-Security-Policy": "",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"No se pudo obtener el archivo: {e}")


@app.post("/archivos")
def buscar_archivos(req: BuscarArchivosRequest):
    db = OracleEnterpriseConnection()
    if not db.connect():
        raise HTTPException(
            status_code=503,
            detail="No se pudo conectar a Oracle. Verifica que exista conexion.py y sus credenciales.",
        )
    try:
        rows = db.execute_query(SQL_ARCHIVOS, {"identificacion": req.identificacion})
        if rows is None:
            raise HTTPException(status_code=500, detail="Error ejecutando consulta Oracle")

        archivos = []
        for row in rows:
            (id_archivo, fecha_cargue, id_usuario, url,
             nombre_archivo, identificacion,
             nombre_alternativo, ya_analizado, analisis_clob) = row

            fecha_yyyymmdd = _fecha_to_yyyymmdd(fecha_cargue)
            analisis_str = _safe_clob_to_str(analisis_clob)

            archivos.append({
                "id_archivo": int(id_archivo) if id_archivo is not None else None,
                "fecha_cargue": fecha_yyyymmdd,        # unificado
                "fecha_cargue_display": _fecha_display(fecha_cargue),
                "id_usuario": int(id_usuario) if id_usuario is not None else None,
                "url": str(url) if url else None,
                "nombre_archivo": str(nombre_archivo) if nombre_archivo else "Archivo sin nombre",
                "identificacion": str(identificacion) if identificacion else None,
                "nombre_alternativo": str(nombre_alternativo) if nombre_alternativo else None,
                "ya_analizado": bool(ya_analizado),
                "analisis": analisis_str,
            })

        return JSONResponse(content={
            "archivos": archivos,
            "identificacion": req.identificacion,
            "total": len(archivos),
        })
    finally:
        db.close_connection()


@app.post("/save-analysis")
def save_analysis(req: SaveAnalysisRequest):
    db = OracleEnterpriseConnection()
    if not db.connect():
        raise HTTPException(status_code=503, detail="No se pudo conectar a Oracle.")
    try:
        result = db.execute_query(
            SQL_SAVE_ANALYSIS,
            {
                "nombre_alternativo": req.nombre_alternativo,
                "analisis": req.analisis,
                "id_archivo": req.id_archivo,
            },
        )
        if result is None:
            raise HTTPException(status_code=500, detail="Error guardando análisis en Oracle")

        return JSONResponse(content={
            "ok": True,
            "id_archivo": req.id_archivo,
            "nombre_alternativo": req.nombre_alternativo,
        })
    finally:
        db.close_connection()


@app.post("/check-pdf")
def check_pdf(req: AnalyzeRequest):
    file_bytes, content_type = download_file(req.url)
    ext = get_file_extension(content_type, req.url)
    if ext != "pdf":
        return JSONResponse(content={"encrypted": False})
    return JSONResponse(content={"encrypted": is_pdf_encrypted(file_bytes)})


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # cache BD
    if req.id_archivo and not req.force_reprocess:
        db = OracleEnterpriseConnection()
        if db.connect():
            try:
                rows = db.execute_query(SQL_CHECK_ANALIZADO, {"id_archivo": req.id_archivo})
                if rows:
                    stored_str = _safe_clob_to_str(rows[0][0])
                    if stored_str and stored_str.strip():
                        stored_result = json.loads(stored_str)
                        stored_result["cached"] = True
                        return JSONResponse(content=stored_result)
            except Exception as e:
                print(f"[analyze] cache warning: {e}")
            finally:
                db.close_connection()

    file_bytes, content_type = download_file(req.url)
    ext = get_file_extension(content_type, req.url)

    if ext == "pdf" and is_pdf_encrypted(file_bytes):
        if not req.password:
            raise HTTPException(status_code=423, detail="PDF_ENCRYPTED")
        file_bytes = unlock_pdf(file_bytes, req.password)

    text = extract_text(file_bytes, content_type, req.url)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No se pudo leer el documento")

    raw_parsed = call_ollama_json(prompt=text, system=PROMPT_ANALISIS)
    result = normalize_to_multi(raw_parsed, req.id_usuario, req.id_archivo)

    examenes = result["examenes"]
    if len(examenes) == 1:
        nombre_alt = build_nombre_alternativo(
            examenes[0].get("tipo_examen", "Examen"),
            examenes[0].get("fecha"),
        )
    else:
        primera_fecha = next((e.get("fecha") for e in examenes if e.get("fecha")), datetime.now().strftime("%Y%m%d"))
        nombre_alt = build_nombre_alternativo(f"MultiExamen_{len(examenes)}", primera_fecha)

    result["nombre_alternativo"] = nombre_alt
    result["cached"] = False

    return JSONResponse(content=result)


@app.post("/evolucion")
def evolucion(req: EvolucionRequest):
    db = OracleEnterpriseConnection()
    if not db.connect():
        raise HTTPException(status_code=503, detail="No se pudo conectar a Oracle.")

    try:
        rows = db.execute_query(SQL_ANALISIS_USUARIO, {"id_usuario": req.id_usuario})
        if rows is None:
            raise HTTPException(status_code=500, detail="Error ejecutando consulta Oracle.")

        data = construir_evolucion(rows, req.id_usuario)
        return JSONResponse(content=data)
    finally:
        db.close_connection()