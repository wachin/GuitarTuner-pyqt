import sys
import numpy as np
import pyaudio
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtCore import QTimer
from scipy.fftpack import fft

# Constantes para el análisis de audio
CHUNK = 4096  # Tamaño del buffer de audio
RATE = 44100  # Frecuencia de muestreo
TUNING_FREQUENCY = 440.0  # Frecuencia estándar (La4 a 440 Hz)

# Mapa de notas y frecuencias en una octava (se puede expandir según las necesidades)
NOTE_FREQUENCIES = {
    'E2': 82.41, 'A2': 110.0, 'D3': 146.83, 'G3': 196.0, 'B3': 246.94, 'E4': 329.63
}
NOTES = list(NOTE_FREQUENCIES.keys())
FREQUENCIES = list(NOTE_FREQUENCIES.values())

class GuitarTuner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initAudio()

    def initUI(self):
        self.setWindowTitle("Guitar Tuner")
        self.setGeometry(100, 100, 300, 200)

        # Etiqueta para mostrar la nota detectada
        self.note_label = QLabel("Note: ", self)
        self.note_label.move(50, 50)

        # Etiqueta para mostrar la desviación en centavos
        self.cents_label = QLabel("Cents: ", self)
        self.cents_label.move(50, 100)

        # Timer para actualizar el afinador cada cierto tiempo
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTuner)
        self.timer.start(100)  # Actualiza cada 100 ms

    def initAudio(self):
        # Inicializa la captura de audio con pyaudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paFloat32,
                                      channels=1,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)

    def get_frequency(self, data):
        # Realiza la FFT sobre los datos de audio y encuentra la frecuencia dominante
        fft_data = np.abs(fft(data))[:CHUNK // 2]
        freq_bins = np.fft.fftfreq(len(data), 1.0 / RATE)[:CHUNK // 2]
        dominant_freq = freq_bins[np.argmax(fft_data)]
        return dominant_freq

    def get_closest_note_and_cents(self, frequency):
        # Encuentra la nota más cercana y calcula la desviación en centavos
        min_diff = float('inf')
        closest_note = None
        closest_freq = None

        for note, note_freq in NOTE_FREQUENCIES.items():
            diff = abs(frequency - note_freq)
            if diff < min_diff:
                min_diff = diff
                closest_note = note
                closest_freq = note_freq

        # Calcula la desviación en centavos
        cents = 1200 * np.log2(frequency / closest_freq) if closest_freq else 0
        return closest_note, int(cents)

    def updateTuner(self):
        # Lee datos del micrófono
        data = np.frombuffer(self.stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)

        # Obtiene la frecuencia dominante
        frequency = self.get_frequency(data)

        # Encuentra la nota más cercana y la desviación en centavos
        note, cents = self.get_closest_note_and_cents(frequency)

        # Actualiza las etiquetas con la nota y los centavos
        self.note_label.setText(f"Note: {note} ({frequency:.2f} Hz)")
        self.cents_label.setText(f"Cents: {cents:+d}")

    def closeEvent(self, event):
        # Cierra el flujo de audio al salir del programa
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GuitarTuner()
    window.show()
    sys.exit(app.exec())
