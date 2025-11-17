import cv2
import os
import time
import datetime

# --- Konfiguracja ---

# Nazwa folderu, w którym będą zapisywane zdjęcia
FOLDER_NAZWA = "zdjecia"

# Źródło wideo (0 to zazwyczaj domyślna kamera internetowa)
source = 0

# Interwał zapisu w sekundach
INTERWAL_ZAPISU = 1.0

# --- Przygotowanie ---

# Utwórz folder 'zdjecia', jeśli jeszcze nie istnieje
# 'exist_ok=True' sprawia, że skrypt nie wyrzuci błędu, jeśli folder już jest
os.makedirs(FOLDER_NAZWA, exist_ok=True)

# Uruchom przechwytywanie wideo
cap = cv2.VideoCapture(source)

# Sprawdź, czy kamera została otwarta poprawnie
if not cap.isOpened():
    print(f"Błąd: Nie można otworzyć źródła wideo: {source}")
    exit()

print("Kamera uruchomiona. Naciśnij 'q', aby zakończyć.")
print(f"Zdjęcia będą zapisywane co {INTERWAL_ZAPISU} sek. do folderu '{FOLDER_NAZWA}'.")

# Zapisz czas ostatniego wykonania zdjęcia
# Używamy time.time() do śledzenia czasu
ostatni_zapis = time.time()

# --- Główna pętla ---

try:
    while True:
        # Odczytaj klatkę z kamery
        ret, frame = cap.read()

        if not ret:
            print("Koniec strumienia wideo lub błąd odczytu.")
            break

        # Wyświetl klatkę na żywo
        cv2.imshow("Podgląd z kamery (naciśnij 'q' aby wyjść)", frame)

        # Sprawdź, czy minęła już 1 sekunda od ostatniego zapisu
        teraz = time.time()
        if teraz - ostatni_zapis >= INTERWAL_ZAPISU:
            # Pobierz aktualny czas do nazwy pliku
            teraz_dt = datetime.datetime.now()

            # Formatuj nazwę pliku: Godzina_Minuta_Sekunda.jpg
            # Używamy podkreśleń (_) zamiast spacji lub dwukropków,
            # ponieważ są bezpieczniejsze dla nazw plików.
            nazwa_pliku = teraz_dt.strftime("%H_%M_%S") + ".jpg"

            # Stwórz pełną ścieżkę zapisu (folder + nazwa pliku)
            pelna_sciezka = os.path.join(FOLDER_NAZWA, nazwa_pliku)

            # Zapisz klatkę do pliku
            cv2.imwrite(pelna_sciezka, frame)

            print(f"Zapisano zdjęcie: {pelna_sciezka}")

            # Zaktualizuj czas ostatniego zapisu
            ostatni_zapis = teraz

        # Czekaj na naciśnięcie klawisza 'q', aby zakończyć
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Zamykanie programu...")
            break

except KeyboardInterrupt:
    # Obsługa przerwania przez użytkownika (np. Ctrl+C w terminalu)
    print("Program przerwany przez użytkownika.")

finally:
    # --- Zakończenie ---
    # Niezależnie od tego, jak pętla się zakończy, zwolnij zasoby
    cap.release()
    cv2.destroyAllWindows()
    print("Zwolniono zasoby kamery i zamknięto okna.")