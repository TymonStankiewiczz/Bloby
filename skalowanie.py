import serial
import time
import sys
import cv2
import numpy as np

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
BASE_PX_PER_MM = 1.5422  # Twój zmierzony stosunek
BASE_DISTANCE_MM = 690.0  # 69 cm

# --- 🎯 --- ZMIANA: STAŁE DO DETEKCJI BIAŁEGO ---
# Zamiast HSV, użyjemy progu jasności na obrazie w skali szarości.
# Wszystko jaśniejsze niż ta wartość (0-255) będzie "białe".
WHITE_THRESHOLD = 200  # Możesz to regulować (np. 180-220)
# Minimalny obszar kwadratu (w pikselach), aby odfiltrować szum
MIN_SQUARE_AREA = 2000


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
    print("Naciśnij 'q', aby zakończyć.\n")

    # --- 2.5 Obliczanie zoptymalizowanej macierzy kamery ---
    h, w = int(height), int(width)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x_roi, y_roi, w_roi, h_roi = roi

    # --- Zmienne pętli ---
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

            # --- 4. Sprawdzenie danych z portu szeregowego ---
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()

                    if "Distance:" in line:
                        parts = line.split(" ")
                        if len(parts) >= 2:
                            current_distance_mm = float(parts[1])
                            # Aktualizuj dynamiczny stosunek TYLKO jeśli mamy poprawny odczyt
                            if current_distance_mm > 0:
                                dynamic_px_per_mm = BASE_PX_PER_MM * (BASE_DISTANCE_MM / current_distance_mm)

                    elif "Couldn't" in line or "Error" in line:
                        current_distance_mm = -1.0
                        dynamic_px_per_mm = 0.0
                except (ValueError, UnicodeDecodeError, serial.SerialException):
                    pass

            # --- 🎯 --- ZMIANA: 5. Detekcja białego prostokąta ---
            # Konwersja do skali szarości
            gray = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2GRAY)

            # Zastosuj próg - wszystko jaśniejsze niż WHITE_THRESHOLD stanie się białe (255)
            ret, mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # (Opcjonalnie) Wygładzenie maski, aby usunąć szum
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            # Znalezienie konturów
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            real_width_mm = 0.0
            real_height_mm = 0.0

            if contours:
                # Znajdź największy kontur
                largest_contour = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest_contour) > MIN_SQUARE_AREA:
                    # Sprawdź, czy kontur jest prostokątem (ma 4 rogi)
                    peri = cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, 0.04 * peri, True)

                    if len(approx) == 4:
                        # Weź prostą ramkę (bounding box)
                        x, y, w, h = cv2.boundingRect(approx)

                        pixel_width = w
                        pixel_height = h  # <-- Nowa zmienna

                        # Oblicz prawdziwą wielkość, jeśli mamy stosunek
                        if dynamic_px_per_mm > 0:
                            real_width_mm = pixel_width / dynamic_px_per_mm
                            real_height_mm = pixel_height / dynamic_px_per_mm  # <-- Obliczenie wysokości

                        # Rysuj ramkę i wymiary
                        cv2.rectangle(frame_to_show, (x, y), (x + w, y + h), FONT_COLOR_MEASURE, 3)

                        text_width = f"Szer: {real_width_mm:.1f} mm"
                        text_height = f"Wys: {real_height_mm:.1f} mm"

                        cv2.putText(frame_to_show, text_width, (x, y - 40), FONT, 1.0, FONT_COLOR_MEASURE, 2)
                        cv2.putText(frame_to_show, text_height, (x, y - 10), FONT, 1.0, FONT_COLOR_MEASURE, 2)

            # --- 6. Rysowanie tekstu informacyjnego na obrazie ---
            if current_distance_mm == -1.0:
                text_distance = "Dystans: BŁĄD CZUJNIKA"
            elif current_distance_mm > 0:
                text_distance = f"Dystans: {current_distance_mm:.0f} mm"
            else:
                text_distance = "Dystans: --- mm"

            text_ratio = f"Stosunek: {dynamic_px_per_mm:.2f} px/mm"

            cv2.putText(frame_to_show, text_distance, (30, 70), FONT, FONT_SCALE, FONT_COLOR_INFO, FONT_THICKNESS)
            cv2.putText(frame_to_show, text_ratio, (30, 120), FONT, FONT_SCALE, FONT_COLOR_RATIO, FONT_THICKNESS)

            # --- 7. Wyświetlanie obrazu ---
            cv2.imshow("Detekcja z dynamicznym skalowaniem", frame_to_show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nZatrzymywanie skryptu...")
    finally:
        # --- 8. Sprzątanie ---
        ser.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Zasoby zwolnione. Zakończono.")


if __name__ == "__main__":
    main()