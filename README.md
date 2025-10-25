# Serviço de Análise de Sentimentos Facial

Microserviço em Python que expõe uma API HTTP para detectar emoções faciais em imagens estáticas
utilizando o modelo pré-treinado disponibilizado pela biblioteca [`fer`](https://github.com/justinshenk/fer).

## Funcionalidades
- Endpoint `/predict` recebe uma imagem (multipart/form-data) e retorna a emoção predominante para cada rosto encontrado.
- Endpoint `/health` permite monitorar o estado do serviço.
- Pronto para execução em GPU por meio de container baseado em CUDA.

## Requisitos locais
- Python 3.10+
- Pip 21+

## Instalação local
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## Execução local
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Exemplo de requisição
```bash
curl -X POST \
  -F "file=@/caminho/para/imagem.jpg" \
  http://localhost:8000/predict
```

Resposta esperada:
```json
{
  "predictions": [
    {
      "label": "happy",
      "confidence": 0.98,
      "box": {"x": 123, "y": 45, "width": 88, "height": 88}
    }
  ]
}
```

## Docker com GPU

1. Compile a imagem (necessário [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
   ```bash
   docker build -t analise-sentimentos-gpu .
   ```
2. Execute liberando a GPU:
   ```bash
   docker run --rm -p 8000:8000 --gpus all analise-sentimentos-gpu
   ```

## Estrutura do projeto
```
.
├── app.py           # Aplicação FastAPI
├── Dockerfile       # Container CUDA com GPU
├── README.md        # Este arquivo
└── requirements.txt # Dependências Python
```

## Considerações
- O detector FER utiliza PyTorch/TensorFlow por baixo dos panos e faz uso automático de GPU quando disponível.
- Imagens com múltiplos rostos retornam uma lista de previsões, uma por rosto detectado.
- Verifique as implicações éticas e legais do reconhecimento de emoções antes de usar em produção.
