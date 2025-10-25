"""Aplicação de análise de sentimentos em tempo real usando webcam.

O script utiliza a biblioteca `fer`, que disponibiliza um modelo pré-treinado
no dataset FER2013, para estimar a emoção predominante no rosto detectado.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
from fer import FER


@dataclass
class EmotionPrediction:
    """Representa a emoção predominante detectada em um rosto."""

    label: str
    confidence: float
    box: Tuple[int, int, int, int]


def setup_camera(index: int = 0) -> cv2.VideoCapture:
    """Inicializa a captura da webcam."""

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(
            "Não foi possível acessar a câmera. Verifique a conexão e as permissões."
        )
    return capture


def predict_emotions(detector: FER, frame) -> list[EmotionPrediction]:
    """Retorna as emoções predominantes para cada rosto no quadro fornecido."""

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(rgb_frame) or []

    predictions: list[EmotionPrediction] = []
    for result in results:
        x, y, w, h = result["box"]
        emotions: Dict[str, float] = result["emotions"]
        label, confidence = max(emotions.items(), key=lambda item: item[1])
        predictions.append(
            EmotionPrediction(
                label=label,
                confidence=confidence,
                box=(x, y, w, h),
            )
        )
    return predictions


def draw_predictions(frame, predictions: list[EmotionPrediction]) -> None:
    """Sobrepõe as informações de emoção no quadro de vídeo."""

    for prediction in predictions:
        x, y, w, h = prediction.box
        label = f"{prediction.label.capitalize()}: {prediction.confidence * 100:.1f}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 10 if y - 10 > 20 else y + h + 20
        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    print("Inicializando detector de emoções...")
    detector = FER()

    try:
        capture = setup_camera()
    except RuntimeError as exc:  # cobertura mínima de erro para feedback rápido
        print(exc)
        sys.exit(1)

    print("Pressione 'q' para encerrar a aplicação.")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Não foi possível ler o quadro da câmera. Encerrando aplicação.")
            break

        predictions = predict_emotions(detector, frame)

        display_frame = frame.copy()
        if predictions:
            draw_predictions(display_frame, predictions)
        else:
            cv2.putText(
                display_frame,
                "Nenhum rosto detectado",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Análise de Sentimentos", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
