# Pipeline de Transcrição e Tradução de Legendas com FFmpeg, Whisper e LM Studio (Qwen2.5)

Este projeto automatiza a extração, transcrição, tradução e revisão de legendas de vídeos, com foco em filmes antigos e estrangeiros.

## Funcionalidades

- Criação automática de pastas:
  - `input/` para entrada de vídeos.
  - `output/` para saída de vídeos processados e legendas `.srt`.
- Extração de legendas embutidas com `ffmpeg`.
- Transcrição de áudio com Whisper local (`faster-whisper`) em GPU, quando o vídeo não tem legenda.
- Detecção automática de idioma da legenda (`langdetect`).
- Consulta de dados do filme na OMDb API (título, ano, país, idiomas, IMDb ID).[file:1]
- Tradução e revisão de legendas para português do Brasil usando Qwen2.5‑7B‑Instruct‑1M via LM Studio:
  - Mantém nomes de pessoas e lugares no idioma original.
  - Evita gírias atuais, usando português neutro.
  - Preserva numeração e timestamps no formato `.srt`.
- Processamento de transcrições longas em blocos (~4000 tokens) para melhor desempenho e estabilidade.
- Move o vídeo original processado para a pasta `output/` junto com a legenda final.

## Arquitetura Resumida

1. O script procura arquivos de vídeo em `input/`.
2. Para cada vídeo:
   - Tenta extrair legenda embutida (`ffmpeg`).
   - Se não houver legenda:
     - Extrai áudio em WAV 16 kHz mono (`ffmpeg`).
     - Transcreve com `faster-whisper` (GPU) gerando um `.srt` básico.[web:16][web:21]
   - Detecta o idioma da legenda (`langdetect`).
   - Tenta obter informações do filme pela OMDb API usando o título inferido do nome do arquivo.[file:1]
   - Divide o `.srt` em blocos de até ~16000 caracteres (≈ 4000 tokens) e envia cada bloco ao modelo Qwen2.5 via LM Studio para tradução e revisão.
   - Concatena os blocos traduzidos em um `.srt` final em português.
   - Move o vídeo original para `output/` e grava a legenda final na mesma pasta, com o mesmo nome base do vídeo.

## Dependências

### Sistema

- `ffmpeg` e `ffprobe` instalados e acessíveis no `PATH`.[web:6]
- Placa de vídeo NVIDIA com drivers e CUDA/cuDNN instalados (para Whisper e LM Studio em GPU).[web:16][web:19]

### Python

Bibliotecas principais:

- `requests` – chamadas HTTP (LM Studio e OMDb).
- `langdetect` – detecção de idioma da legenda.
- `python-dotenv` – carregamento de variáveis de ambiente.
- `faster-whisper` – transcrição de áudio local em GPU.[web:16][web:19]

Instalação recomendada:

```bash
pip install requests langdetect python-dotenv faster-whisper

LM Studio e Modelo Qwen2.5

    Baixe e instale o LM Studio.

    Baixe o modelo qwen2.5-7b-instruct-1m@q8_0 (ou outro que preferir com contexto longo).[web:31][web:32]

    Nas configurações do LM Studio:

        Ative o uso de GPU (CUDA).

        Configure o contexto do modelo para 16384 tokens.

        Ative o servidor local de API (menu “API”/“Server”).

    Anote o nome exato do modelo no LM Studio (por exemplo qwen2.5-7b-instruct-1m) e coloque na constante LMSTUDIO_MODEL do script.

O script se conecta ao LM Studio usando a API de chat compatível com OpenAI, em http://localhost:1234/v1/chat/completions (endereço padrão do servidor LM Studio).[web:2][web:4]
OMDb API (dados do filme)

    Crie uma conta em https://www.omdbapi.com/.

    Obtenha sua chave de API (por exemplo 3754e60e).[file:1]

    Crie um arquivo .env na raiz do projeto com:

text
OMDB_API_KEY=3754e60e
WHISPER_MODEL_SIZE=small

A OMDb API será usada via chamadas do tipo:

text
http://www.omdbapi.com/?apikey=YOUR_KEY&t=TITULO_AQUI

para obter título oficial, ano, país, idiomas e IMDb ID.[file:1]
Uso

    Garanta que:

        LM Studio esteja aberto, com o modelo Qwen2.5 carregado e o servidor local habilitado.

        Sua GPU esteja disponível.

    Coloque um ou mais arquivos de vídeo em input/.

    Execute o script principal:

bash
python main.py

    Para cada vídeo, o script:

        Gera (ou extrai) uma legenda .srt original.

        Traduz e revisa a legenda para português do Brasil em blocos.

        Move o vídeo para output/ e salva a legenda final em output/NOME_DO_VIDEO.srt.

Notas e Limitações

    OMDb/IMDb não oferecem busca por hash de vídeo; se desejar usar hash, será para controle interno seu (por exemplo, mapear hash → título/IMDb ID numa base própria).[web:28][file:1]

    O tempo de processamento depende:

        Duração do filme.

        Modelo Whisper escolhido.

        Modelo Qwen2.5 e quantização.

    Para filmes muito longos, você pode reduzir o tamanho máximo de caracteres por bloco (por exemplo, 12000) para garantir ainda mais folga em contexto.

Licença

Use e modifique o projeto conforme sua necessidade pessoal. Verifique licenças de modelos (Qwen2.5) e da API OMDb antes de uso comercial.