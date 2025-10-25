# Análise de Sentimentos em Tempo Real

Esta aplicação em Python abre a câmera do computador e utiliza um modelo pré-treinado de reconhecimento de emoções faciais para indicar, em tempo real, o sentimento predominante da pessoa enquadrada.

## Pré-requisitos
- Python 3.9 ou superior
- Webcam conectada ao computador

## Instalação
1. Crie e ative um ambiente virtual (opcional, mas recomendado).
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como executar
1. Conecte a webcam ao computador.
2. Execute o script principal:
   ```bash
   python app.py
   ```
3. Uma janela será aberta exibindo a imagem da câmera com a emoção predominante sobreposta ao rosto detectado.
4. Pressione a tecla `q` para encerrar a aplicação.

## Estrutura do projeto
```
.
├── app.py            # Código principal da aplicação de reconhecimento de emoções
├── README.md         # Este arquivo
└── requirements.txt  # Dependências necessárias
```

## Observações
- O modelo utilizado é fornecido pela biblioteca `fer`, que disponibiliza pesos treinados no dataset FER2013.
- A aplicação é voltada para demonstrações educacionais. Para uso profissional, avalie as implicações éticas e de privacidade, e considere calibrações adicionais.
