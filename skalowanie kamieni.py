import cv2
import numpy as np
import serial
import time
import sys
import threading
from ultralytics import YOLO

# --- KONFIGURACJA PORTU SZEREGOWEGO ---
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# --- KONFIGURACJA KAMERY ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- KONFIGURACJA WYŚWIETLANIA TEKSTU ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_COLOR_INFO = (0, 255, 0)  # Zielony
FONT_COLOR_RATIO = (0, 255, 255)  # Żółty
FONT_COLOR_MEASURE = (0, 0, 255)  # Czerwony
FONT_THICKNESS = 2

# --- Ścieżki do plików kalibracyjnych ---
CAL_MATRIX_FILE = 'camera_matrix.npy'
DIST_COEFFS_FILE = 'dist_coeffs.npy'

# --- 🎯 STAŁE DO SKALOWANIA (TWOJE WARTOŚCI) ---
BASE_PX_PER_MM = 1.5422
BASE_DISTANCE_MM = 690.0

# --- 🎯 Ścieżka do modelu YOLO ---
YOLO_MODEL_PATH = 'best2.pt'


# --- 🎯 ZMIANA: Funkcja dla wątku czytającego port szeregowy ---
# Ta funkcja będzie działać w tle
def serial_reader_task(ser, shared_data, lock):
    """
    Czyta dane z portu szeregowego w osobnym wątku i aktualizuje
    słownik `shared_data` najnowszymi pomiarami.
    """
    print("Wątek szeregowy rozpoczęty.")
    while shared_data['running']:
        try:
            # Używamy wbudowanego timeoutu w ser.readline()
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:  # Jeśli linia jest pusta (timeout)
                continue

            # Mamy linię, przetwarzamy ją
            distance = 0.0
            px_per_mm = 0.0
            status = 0  # 0 = OK, -1 = Błąd

            if "Distance:" in line:
                parts = line.split(" ")
                if len(parts) >= 2:
                    distance = float(parts[1])
                    if distance > 0:
                        px_per_mm = BASE_PX_PER_MM * (BASE_DISTANCE_MM / distance)
                    status = 0
            elif "Couldn't" in line or "Error" in line:
                status = -1

            # --- Bezpieczna aktualizacja danych ---
            # Zdobądź blokadę, aby główna pętla nie czytała danych w połowie ich zapisu
            with lock:
                if status == -1:
                    shared_data['distance'] = -1.0
                    shared_data['px_per_mm'] = 0.0
                elif distance > 0:
                    shared_data['distance'] = distance
                    shared_data['px_per_mm'] = px_per_mm
            # Zwalniamy blokadę automatycznie (dzięki 'with')

        except serial.SerialException as e:
            print(f"Błąd portu szeregowego: {e}")
            with lock:
                shared_data['running'] = False
        except Exception as e:
            # Ignoruj błędy parsowania itp., aby wątek działał dalej
            # print(f"Błąd w wątku (ignorowany): {e}")
            pass

    print("Wątek szeregowy zakończony.")


def main():
    # --- 0. Ładowanie plików kalibracyjnych ---
    print("Ładowanie plików kalibracyjnych...")
    try:
        mtx = np.load(CAL_MATRIX_FILE)
        dist = np.load(DIST_COEFFS_FILE)
        print("Pliki kalibracyjne załadowane.")
    except Exception as e:
        print(f"\nBŁĄD: Nie można załadować plików .npy. Błąd: {e}")
        sys.exit(1)

    # --- 1. Inicjalizacja Portu Szeregowego ---
    print(f"Próba połączenia z {SERIAL_PORT}...")
    try:
        # WAŻNE: Ustawiamy timeout, aby .readline() nie blokowało wątku na wieczność
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print("Połączono z portem COM!")
    except Exception as e:
        print(f"\nBŁĄD: Nie można otworzyć portu {SERIAL_PORT}. Błąd: {e}")
        sys.exit(1)

    # --- 2. Inicjalizacja Kamery ---
    print(f"Próba otwarcia kamery (indeks: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\nBŁĄD: Nie można otworzyć kamery.")
        ser.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Kamera otwarta! Rozdzielczość: {int(width)}x{int(height)}")

    # --- 2.5 Obliczanie zoptymalizowanej macierzy kamery ---
    h, w = int(height), int(width)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x_roi, y_roi, w_roi, h_roi = roi

    # --- 2.6 Ładowanie modelu YOLO ---
    print("Ładowanie modelu YOLOv8...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("Model YOLO załadowany.")
    except Exception as e:
        print(f"\nBŁĄD: Nie można załadować modelu YOLO '{YOLO_MODEL_PATH}'. Błąd: {e}")
        ser.close()
        cap.release()
        sys.exit(1)

    # --- 🎯 ZMIANA: Konfiguracja wielowątkowości ---
    data_lock = threading.Lock()
    shared_data = {
        'distance': 0.0,
        'px_per_mm': 0.0,
        'running': True  # Flaga do zatrzymania wątku
    }

    # Uruchomienie wątku czytającego port szeregowy
    serial_thread = threading.Thread(
        target=serial_reader_task,
        args=(ser, shared_data, data_lock),
        daemon=True  # Wątek zakończy się automatycznie, gdy główny program się zamknie
    )
    serial_thread.start()

    print("\nNaciśnij 'q', aby zakończyć.\n")

    # --- Zmienne pętli (lokalne kopie) ---
    current_distance_mm = 0.0
    dynamic_px_per_mm = 0.0

    try:
        while True:
            # --- 3. Odczyt i kalibracja klatki ---
            ret, frame = cap.read()
            if not ret:
                break

            undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
            frame_roi = undistorted_frame[y_roi: y_roi + h_roi, x_roi: x_roi + w_roi]
            frame_to_show = cv2.flip(frame_roi, 1)

            # --- 🎯 ZMIANA: 4. Pobranie najnowszych danych z wątku ---
            # Zamiast czytać z portu, pobieramy najnowsze dane ze 'shared_data'
            # Robimy to w bloku 'with lock', aby mieć pewność, że dane są spójne
            with data_lock:
                current_distance_mm = shared_data['distance']
                dynamic_px_per_mm = shared_data['px_per_mm']

                if not shared_data['running']:  # Sprawdź, czy wątek nie zgłosił błędu
                    print("Wątek szeregowy zatrzymany, zamykanie pętli głównej.")
                    break

            # --- CAŁA SEKCJA `if ser.in_waiting > 0:` ZOSTAŁA USUNIĘTA ---

            # --- 5. Detekcja i Pomiar YOLO ---
            results = model(frame_to_show, verbose=False)
            annotated_frame = results[0].plot()

            # Pobierz wysokość klatki, aby wykryć dolną krawędź ekranu
            # annotated_frame.shape[0] to wysokość obrazu w pikselach
            screen_height = annotated_frame.shape[0]

            for box in results[0].boxes:
                # Współrzędne ramki: (lewy-górny x, lewy-górny y, prawy-dolny x, prawy-dolny y)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pixel_width = x2 - x1
                pixel_height = y2 - y1

                real_width_mm = 0.0
                real_height_mm = 0.0

                # Użyj danych `dynamic_px_per_mm` pobranych na początku pętli
                if dynamic_px_per_mm > 0:
                    real_width_mm = pixel_width / dynamic_px_per_mm
                    real_height_mm = pixel_height / dynamic_px_per_mm

                text_width = f"Szer: {real_width_mm:.1f} mm"
                text_height = f"Wys: {real_height_mm:.1f} mm"

                # --- 🎯 Nowa logika pozycjonowania tekstu ---

                # Ustaw pozycje tekstu domyślnie POD ramką
                # (bazując na dolnej krawędzi 'y2')
                # Możesz dostosować wartości '30' i '60' jeśli tekst jest za blisko/za daleko
                text_y_width = y2 + 30
                text_y_height = y2 + 60

                # ZABEZPIECZENIE: Sprawdź, czy druga linia tekstu nie wychodzi poza ekran
                if text_y_height > screen_height:
                    # Jeśli tak, przełącz się na rysowanie NAD ramką
                    # (bazując na górnej krawędzi 'y1')
                    text_y_width = y1 - 40
                    text_y_height = y1 - 10

                # Narysuj tekst na obliczonych pozycjach
                cv2.putText(annotated_frame, text_width, (x1, text_y_width), FONT, 1.0, FONT_COLOR_MEASURE, 2)
                cv2.putText(annotated_frame, text_height, (x1, text_y_height), FONT, 1.0, FONT_COLOR_MEASURE, 2)

            # --- 6. Rysowanie tekstu informacyjnego na obrazie ---
            # Użyj danych `current_distance_mm` pobranych na początku pętli
            if current_distance_mm == -1.0:
                text_distance = "Dystans: BŁĄD CZUJNIKA"
            elif current_distance_mm > 0:
                text_distance = f"Dystans: {current_distance_mm:.0f} mm"
            else:
                text_distance = "Dystans: --- mm"

            text_ratio = f"Stosunek: {dynamic_px_per_mm:.2f} px/mm"

            cv2.putText(annotated_frame, text_distance, (30, 70), FONT, FONT_SCALE, FONT_COLOR_INFO, FONT_THICKNESS)
            cv2.putText(annotated_frame, text_ratio, (30, 120), FONT, FONT_SCALE, FONT_COLOR_RATIO, FONT_THICKNESS)

            # --- 7. Wyświetlanie obrazu ---
            cv2.imshow("Detekcja YOLO z dynamicznym skalowaniem", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nZatrzymywanie skryptu...")
    finally:
        # --- 8. Sprzątanie ---
        print("Rozpoczynanie zamykania...")
        # Sygnalizuj wątkowi, że ma się zakończyć
        with data_lock:
            shared_data['running'] = False

        # Poczekaj, aż wątek się zakończy
        serial_thread.join(timeout=1.0)
        print("Wątek szeregowy zamknięty.")

        ser.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Zasoby zwolnione. Zakończono.")


if __name__ == "__main__":
    main()