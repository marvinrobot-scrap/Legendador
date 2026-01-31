import os
import sys
import re
import time
import subprocess
import shutil
import json
import requests
from langdetect import detect
from dotenv import load_dotenv

# ================== CONFIGURAÇÃO DLLs NVIDIA (Windows) ==================

def configurar_dlls_nvidia():
    """
    Garante que o Windows encontre as DLLs da NVIDIA (cublas, cudnn, etc.)
    instaladas via pip dentro de site-packages.
    """
    print("[INIT] Configurando ambiente GPU e DLLs (NVIDIA)...")

    base_python = sys.prefix
    caminhos_possiveis = [
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    ]

    dl_encontradas = 0
    for path in caminhos_possiveis:
        if os.path.exists(path):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                    dl_encontradas += 1
                except Exception:
                    pass

    if dl_encontradas > 0:
        print(f"   [OK] {dl_encontradas} diretórios de DLLs da NVIDIA registrados.")
    else:
        print("   [AVISO] Nenhuma pasta NVIDIA encontrada automaticamente. "
              "Se continuar dando erro de cublas/cudnn, verifique a instalação das libs NVIDIA.")

# Executa a configuração ANTES de carregar o Whisper
configurar_dlls_nvidia()

# ================== IMPORT DO WHISPER (GPU) ==================

from faster_whisper import WhisperModel  # [web:16][web:19]

# ================== CONFIG GERAL ==================

load_dotenv()

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"      # ajuste se mudar a porta[web:2]
LMSTUDIO_MODEL = "qwen2.5-7b-instruct-1m"           # nome do modelo no LM Studio[web:31]
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # [web:19]

INPUT_DIR = "input"
OUTPUT_DIR = "output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== FFmpeg: legendas e áudio ==================

def run_ffmpeg_extract_subtitles(video_path, srt_out_path):
    """
    Tenta extrair o primeiro stream de legenda embutido em formato SRT.
    """
    cmd_probe = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_type",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return False

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        return False

    subtitle_index = streams[0]["index"]

    cmd_extract = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-map", f"0:s:{subtitle_index}",
        "-c:s", "srt",
        srt_out_path,
    ]
    r = subprocess.run(cmd_extract, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(srt_out_path)


def run_ffmpeg_extract_audio(video_path, audio_out_path):
    """
    Extrai áudio em formato ideal para Whisper: WAV 16kHz mono PCM.[web:21]
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        audio_out_path,
    ]
    subprocess.run(cmd, check=True)

# ================== Detecção de idioma ==================

def detect_language_from_srt(srt_path):
    text_samples = []
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "-->" in line:
                continue
            if line.isdigit():
                continue
            text_samples.append(line)
            if len(text_samples) > 50:
                break
    if not text_samples:
        return "unknown"
    text = " ".join(text_samples)
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    return lang

# ================== OMDb (busca por título/ano) ==================

def parse_title_and_year_from_filename(filename_base):
    """
    Tenta extrair 'Título' e 'Ano' de um nome no formato:
    'Titulo do Filme (1979)'.
    """
    m = re.match(r"^(.*)\((\d{4})\)$", filename_base.strip())
    if m:
        title = m.group(1).strip()
        year = m.group(2)
        return title, year

    cleaned = re.sub(r"[._]+", " ", filename_base)
    cleaned = cleaned.strip()
    return cleaned, None


def call_omdb_by_search(raw_title_guess):
    """
    Usa OMDb com 's=' (search) e tenta escolher o melhor resultado.[web:60][web:63][web:71]
    """
    if not OMDB_API_KEY:
        return None

    title, year = parse_title_and_year_from_filename(raw_title_guess)

    params_search = {
        "s": title,
        "type": "movie",
        "apikey": OMDB_API_KEY,
    }
    if year:
        params_search["y"] = year

    try:
        resp = requests.get("http://www.omdbapi.com/", params=params_search, timeout=10)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    if data.get("Response") != "True" or "Search" not in data:
        return None

    candidates = data["Search"]

    best = None
    if year:
        for c in candidates:
            if c.get("Year") == year:
                best = c
                break
    if best is None and candidates:
        best = candidates[0]

    if not best:
        return None

    imdb_id = best.get("imdbID")
    if not imdb_id:
        return None

    params_detail = {
        "i": imdb_id,
        "apikey": OMDB_API_KEY,
        "plot": "short",
    }
    try:
        resp2 = requests.get("http://www.omdbapi.com/", params=params_detail, timeout=10)
    except Exception:
        return None

    if resp2.status_code != 200:
        return None

    d = resp2.json()
    if d.get("Response") != "True":
        return None

    return {
        "Title": d.get("Title"),
        "Year": d.get("Year"),
        "Country": d.get("Country"),
        "Language": d.get("Language"),
        "imdbID": d.get("imdbID"),
    }

# ================== LM Studio (Qwen2.5) ==================

def lmstudio_chat(system_prompt, user_prompt, temperature=0.15, max_tokens=2048):
    """
    Chamada à API de chat do LM Studio (compatível com OpenAI).[web:2][web:4]
    """
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ================== Whisper local (GPU) ==================

WHISPER_MODEL = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cuda",
    compute_type="int8_float16",  # [web:19]
)

def transcribe_with_whisper_to_srt(audio_path, srt_out_path, language=None):
    """
    Usa faster-whisper para transcrever audio_path e gerar um SRT básico.[web:16][web:19]
    """
    segments, info = WHISPER_MODEL.transcribe(
        audio_path,
        beam_size=5,
        best_of=5,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    def format_timestamp(seconds):
        millis = int(round(seconds * 1000))
        hours = millis // 3_600_000
        millis %= 3_600_000
        minutes = millis // 60_000
        millis %= 60_000
        secs = millis // 1000
        millis %= 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    lines = []
    idx = 1
    for seg in segments:
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        idx += 1

    with open(srt_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================== Utilitários SRT (entradas) ==================

def parse_srt_entries(srt_path):
    """
    Lê um SRT e retorna lista de dicts:
    {"index": "1", "time": "...", "text_lines": ["..."]}.
    """
    entries = []
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        block_lines = []
        for line in f:
            if line.strip() == "":
                if block_lines:
                    entry = _parse_single_srt_block(block_lines)
                    if entry:
                        entries.append(entry)
                    block_lines = []
            else:
                block_lines.append(line.rstrip("\n"))
        if block_lines:
            entry = _parse_single_srt_block(block_lines)
            if entry:
                entries.append(entry)
    return entries


def _parse_single_srt_block(lines):
    if len(lines) < 2:
        return None
    index = lines[0].strip()
    time_line = lines[1].strip()
    text_lines = [l for l in lines[2:] if l.strip() != ""]
    return {"index": index, "time": time_line, "text_lines": text_lines}


def serialize_srt_entries(entries):
    out_lines = []
    for e in entries:
        out_lines.append(str(e["index"]))
        out_lines.append(e["time"])
        if e["text_lines"]:
            out_lines.extend(e["text_lines"])
        out_lines.append("")
    return "\n".join(out_lines).strip() + "\n"


def _split_text_into_lines(text, max_len=42):
    """
    Quebra o texto em linhas de no máximo max_len caracteres.
    """
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for w in words:
        if current and current_len + 1 + len(w) > max_len:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            if current:
                current_len += 1 + len(w)
            else:
                current_len = len(w)
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines

# ================== Tradução/Revisão entrada a entrada ==================

def _parse_lote_resposta(resposta):
    """
    Lê resposta no formato:
    ID: 12
    TEXTO_TRADUZIDO: ...
    """
    traduzidos = {}
    current_id = None
    current_text_lines = []

    for line in resposta.splitlines():
        line = line.strip()
        if line.startswith("ID:"):
            if current_id is not None and current_text_lines:
                traduzidos[current_id] = " ".join(current_text_lines).strip()
            current_id = line[3:].strip()
            current_text_lines = []
        elif line.startswith("TEXTO_TRADUZIDO:"):
            txt = line[len("TEXTO_TRADUZIDO:"):].strip()
            current_text_lines.append(txt)
        elif current_id is not None and line:
            current_text_lines.append(line)

    if current_id is not None and current_text_lines:
        traduzidos[current_id] = " ".join(current_text_lines).strip()

    return traduzidos


def translate_and_review_srt(original_srt_path, movie_info=None):
    """
    Percorre as entradas do SRT e, em blocos, pede ao Qwen para
    traduzir/revisar apenas o texto, mantendo índices e tempos.
    """
    entries = parse_srt_entries(original_srt_path)

    movie_context = ""
    if movie_info:
        movie_context = (
            f"Título oficial: {movie_info.get('Title')}\n"
            f"Ano: {movie_info.get('Year')}\n"
            f"País: {movie_info.get('Country')}\n"
            f"Idiomas originais: {movie_info.get('Language')}\n"
            f"IMDb ID: {movie_info.get('imdbID')}\n"
        )

    system_prompt = (
        "Você é um tradutor e revisor profissional de legendas de filmes.\n"
        "Regras:\n"
        "- Receberá lotes de falas numeradas (ID) com texto original.\n"
        "- Para cada fala, traduza o texto para português do Brasil, corrigindo gramática e sintaxe.\n"
        "- Mantenha o sentido e o tom; evite gírias modernas e internetês.\n"
        "- NÃO traduza nomes próprios de pessoas e lugares; mantenha no idioma original.\n"
        "- NÃO adicione nem remova falas; devolva exatamente um texto traduzido por ID.\n"
        "- Não inclua comentários extras.\n"
    )

    BATCH_SIZE = 100  # entradas de legenda por chamada

    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start:start + BATCH_SIZE]
        print(f"Revendo/Traduzindo entradas {start + 1} a {start + len(batch)}...")

        linhas_lote = []
        for e in batch:
            original_text = " / ".join(e["text_lines"])
            linhas_lote.append(f"ID: {e['index']}")
            linhas_lote.append(f"TEXTO_ORIGINAL: {original_text}")
        lote_str = "\n".join(linhas_lote)

        user_prompt = (
            "Informações sobre o filme:\n"
            f"{movie_context}\n\n"
            "A seguir está um lote de falas de legenda, cada uma com um ID e um TEXTO_ORIGINAL.\n"
            "Para cada fala, devolva APENAS o texto traduzido e revisado, linha por linha, no formato:\n"
            "ID: <id>\n"
            "TEXTO_TRADUZIDO: <texto em português do Brasil>\n\n"
            "Lote de falas:\n"
            f"{lote_str}\n"
        )

        resposta = lmstudio_chat(
            system_prompt,
            user_prompt,
            temperature=0.15,
            max_tokens=2048
        )

        traduzidos = _parse_lote_resposta(resposta)

        for e in batch:
            if e["index"] in traduzidos:
                novo_texto = traduzidos[e["index"]]
                e["text_lines"] = _split_text_into_lines(novo_texto, max_len=42)

    return serialize_srt_entries(entries)

# ================== Pipeline principal ==================

def process_video_file(video_path, movie_title_guess=None):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_srt = os.path.join(INPUT_DIR, base_name + ".temp.srt")
    final_srt = os.path.join(OUTPUT_DIR, base_name + ".srt")

    print(f"Processando: {video_path}")

    has_subs = run_ffmpeg_extract_subtitles(video_path, temp_srt)
    if not has_subs:
        print("Nenhuma legenda embutida encontrada. Extraindo áudio e gerando legenda com Whisper (GPU)...")
        audio_path = os.path.join(INPUT_DIR, base_name + ".wav")
        run_ffmpeg_extract_audio(video_path, audio_path)
        transcribe_with_whisper_to_srt(audio_path, temp_srt, language=None)
        os.remove(audio_path)

    lang = detect_language_from_srt(temp_srt)
    print(f"Idioma detectado da legenda: {lang}")

    movie_info = call_omdb_by_search(base_name)
    if movie_info:
        print("Informações do filme encontradas:", movie_info)
    else:
        print(f"Não foi possível obter informações do filme pela OMDb a partir de '{base_name}'.")

    translated_srt_text = translate_and_review_srt(temp_srt, movie_info=movie_info)

    with open(final_srt, "w", encoding="utf-8") as f:
        f.write(translated_srt_text)

    new_video_path = os.path.join(OUTPUT_DIR, os.path.basename(video_path))
    shutil.move(video_path, new_video_path)

    if os.path.exists(temp_srt):
        os.remove(temp_srt)

    print(f"Arquivo SRT final salvo em: {final_srt}")
    print(f"Vídeo movido para: {new_video_path}")


def main():
    video_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".flv"))
    ]
    if not video_files:
        print(f"Nenhum arquivo de vídeo encontrado em '{INPUT_DIR}'.")
        return

    for i, vf in enumerate(video_files, start=1):
        print(f"\n===== Iniciando vídeo {i}/{len(video_files)}: {vf} =====\n")
        video_path = os.path.join(INPUT_DIR, vf)
        process_video_file(video_path)
        time.sleep(2)  # pequena pausa opcional entre vídeos


if __name__ == "__main__":
    main()
