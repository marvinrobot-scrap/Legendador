import os
import sys
import re
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

from faster_whisper import WhisperModel  # usa CUDA se disponível[web:16][web:19]

# ================== CONFIG GERAL ==================

load_dotenv()

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # ajuste se for outra porta[web:2]
LMSTUDIO_MODEL = "qwen2.5-7b-instruct-1m"       # nome lógico do modelo no LM Studio[web:31]
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # base/small/medium/large-v2[web:19]

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
    Se não encontrar ano, devolve apenas o título limpo.
    """
    m = re.match(r"^(.*)\((\d{4})\)$", filename_base.strip())
    if m:
        title = m.group(1).strip()
        year = m.group(2)
        return title, year

    # Se não estiver no formato Título (Ano), só limpa pontos/_ e retorna sem ano
    cleaned = re.sub(r"[._]+", " ", filename_base)
    cleaned = cleaned.strip()
    return cleaned, None


def call_omdb_by_search(raw_title_guess):
    """
    Usa OMDb com 's=' (search) e tenta escolher o melhor resultado,
    de preferência usando o ano se existir no nome do arquivo.[web:60][web:63][web:71]
    """
    if not OMDB_API_KEY:
        return None

    title, year = parse_title_and_year_from_filename(raw_title_guess)

    # 1) Busca lista de resultados
    params_search = {
        "s": title,
        "type": "movie",   # foca só em filmes[web:60]
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

    candidates = data["Search"]  # lista de {Title, Year, imdbID, Type}[web:60][web:63]

    # 2) Escolhe melhor candidato
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

    # 3) Busca detalhes completos pelo imdbID
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

def lmstudio_chat(system_prompt, user_prompt, temperature=0.15, max_tokens=4096):
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

# Carregamos o modelo uma vez para reaproveitar
WHISPER_MODEL = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cuda",
    compute_type="int8_float16",  # bom compromisso entre qualidade e VRAM[web:19]
)

def transcribe_with_whisper_to_srt(audio_path, srt_out_path, language=None):
    """
    Usa faster-whisper para transcrever audio_path e gerar um SRT básico.[web:16][web:19]
    language: código ISO (ex: 'en', 'de', 'ru'); se None, auto-detecção.
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
        lines.append("")  # linha em branco
        idx += 1

    with open(srt_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ================== Utilitários SRT (blocos) ==================

def load_srt_entries(srt_path):
    """
    Lê um SRT e retorna uma lista de entradas, cada uma como string completa.
    """
    entries = []
    current = []

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if current:
                    entries.append("".join(current).rstrip() + "\n\n")
                    current = []
            else:
                current.append(line)
        if current:
            entries.append("".join(current).rstrip() + "\n\n")

    return entries


def chunk_srt_entries_by_chars(entries, max_chars=16000):
    """
    Agrupa entradas SRT em blocos cujo texto total não passa de max_chars.
    max_chars ≈ 4000 tokens (usando ~4 caracteres por token).[web:48][web:56]
    """
    blocks = []
    current_block = []
    current_len = 0

    for entry in entries:
        L = len(entry)
        if current_len + L > max_chars and current_block:
            blocks.append("".join(current_block))
            current_block = [entry]
            current_len = L
        else:
            current_block.append(entry)
            current_len += L

    if current_block:
        blocks.append("".join(current_block))

    return blocks

def translate_and_review_srt(original_srt_path, movie_info=None):
    """
    Divide o SRT em blocos (~4000 tokens), envia cada bloco para o Qwen
    e concatena as saídas em um único SRT traduzido.
    """
    entries = load_srt_entries(original_srt_path)
    blocks = chunk_srt_entries_by_chars(entries, max_chars=16000)

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
        "Você é um tradutor profissional especializado em legendas de filmes.\n"
        "Regras importantes:\n"
        "- Mantenha todos os nomes próprios de pessoas e lugares no idioma original.\n"
        "- Traduza para português do Brasil com foco em clareza, gramática e sintaxe corretas.\n"
        "- Evite gírias atuais, internetês e expressões muito modernas; use um português neutro.\n"
        "- Preserve a numeração e os timestamps das legendas no formato SRT.\n"
        "- Cada bloco deve continuar sincronizado com o respectivo tempo; apenas traduza o texto.\n"
        "- Não inclua comentários ou explicações extras; devolva APENAS o conteúdo SRT traduzido."
    )

    translated_blocks = []

    for i, block in enumerate(blocks, start=1):
        print(f"Traduzindo bloco {i}/{len(blocks)}...")

        user_prompt = (
            "A seguir estão as informações conhecidas sobre o filme, seguidas de um TRECHO do arquivo SRT original.\n\n"
            "=== INFORMAÇÕES DO FILME ===\n"
            f"{movie_context}\n"
            "=== TRECHO DE LEGENDA ORIGINAL (SRT) ===\n"
            f"{block}\n"
            "Por favor, devolva apenas o arquivo SRT traduzido e revisado deste trecho, "
            "no mesmo formato, sem comentários adicionais."
        )

        translated_block = lmstudio_chat(
            system_prompt,
            user_prompt,
            temperature=0.15,   # mais determinístico[web:31]
            max_tokens=4096     # suficiente para blocos dessa ordem[web:32]
        )

        translated_blocks.append(translated_block.rstrip() + "\n\n")

    return "".join(translated_blocks)

# ================== Pipeline principal ==================

def process_video_file(video_path, movie_title_guess=None):
    """
    Processa um único arquivo de vídeo:
    - Tenta extrair legenda embutida.
    - Se não tiver legenda, extrai áudio e gera SRT com Whisper (GPU).
    - Detecta idioma da legenda.
    - Obtém informações do filme pela OMDb via busca ('s=') usando o nome do arquivo.
    - Traduz e revisa a legenda em blocos com Qwen (LM Studio).
    - Salva SRT final em output/ e move o vídeo para output/.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_srt = os.path.join(INPUT_DIR, base_name + ".temp.srt")
    final_srt = os.path.join(OUTPUT_DIR, base_name + ".srt")

    print(f"Processando: {video_path}")

    # 1) Tenta extrair legenda embutida
    has_subs = run_ffmpeg_extract_subtitles(video_path, temp_srt)
    if not has_subs:
        print("Nenhuma legenda embutida encontrada. Extraindo áudio e gerando legenda com Whisper (GPU)...")
        audio_path = os.path.join(INPUT_DIR, base_name + ".wav")
        run_ffmpeg_extract_audio(video_path, audio_path)
        transcribe_with_whisper_to_srt(audio_path, temp_srt, language=None)
        os.remove(audio_path)

    # 2) Detecta idioma da legenda
    lang = detect_language_from_srt(temp_srt)
    print(f"Idioma detectado da legenda: {lang}")

    # 3) Busca informações do filme pela OMDb usando o nome do arquivo (base_name)
    movie_info = call_omdb_by_search(base_name)
    if movie_info:
        print("Informações do filme encontradas:", movie_info)
    else:
        print(f"Não foi possível obter informações do filme pela OMDb a partir de '{base_name}'.")

    # 4) Tradução e revisão em blocos
    translated_srt_text = translate_and_review_srt(temp_srt, movie_info=movie_info)

    # 5) Grava SRT final em output/
    with open(final_srt, "w", encoding="utf-8") as f:
        f.write(translated_srt_text)

    # 6) Move o vídeo original para output/
    new_video_path = os.path.join(OUTPUT_DIR, os.path.basename(video_path))
    shutil.move(video_path, new_video_path)

    # 7) Remove temporários
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

    for vf in video_files:
        video_path = os.path.join(INPUT_DIR, vf)
        process_video_file(video_path)


if __name__ == "__main__":
    main()
