from ultralytics import YOLO
import torch

# Sprawdź, czy dostępne jest GPU i wyczyść pamięć (dobra praktyka)
if torch.cuda.is_available():
    print("Dostępne GPU: ", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    print("GPU nie jest dostępne, trening będzie bardzo wolny.")


def trenuj_model():
    # 1. Załaduj model: 'yolov8x.pt' to największy, najdokładniejszy,
    #    ale też najwolniejszy i najbardziej wymagający model.
    model = YOLO('yolov8x.pt')

    # 2. Rozpocznij trening
    print("Rozpoczynam trening modelu yolov8x...")
    try:
        results = model.train(
            # --- Podstawowe parametry ---
            data='data.yaml',  # Ścieżka do Twojego pliku .yaml
            epochs=75,  # Liczba epok (dobry punkt startowy)
            batch=4,  # Zobacz uwagę o VRAM poniżej!
            imgsz=640,  # Standardowy rozmiar obrazu
            patience=20,  # Zatrzymaj trening, jeśli przez 20 epok nie ma poprawy
            name='yolov8x_kamienie_bez_augmentacji',  # Nazwa folderu z wynikami

            # --- Optymalizator i planista ---
            # Dobre, zbalansowane ustawienia
            optimizer='AdamW',  # Często daje stabilniejsze wyniki niż SGD
            lr0=0.001,  # Początkowy learning rate (0.001 dla AdamW jest OK)

            # --- WYŁĄCZENIE AUGMENTACJI (zgodnie z prośbą) ---
            # Ustawienie wszystkich wartości na 0.0 lub False wyłącza je.
            augment=False,  # Główny przełącznik (chociaż lepiej być pewnym poniżej)
            hsv_h=0.0,  # Augmentacja odcienia (Hue)
            hsv_s=0.0,  # Augmentacja nasycenia (Saturation)
            hsv_v=0.0,  # Augmentacja wartości (Value)
            degrees=0.0,  # Rotacja
            translate=0.0,  # Przesunięcie
            scale=0.0,  # Skalowanie (zoom)
            shear=0.0,  # Pochylenie
            perspective=0.0,  # Perspektywa
            flipud=0.0,  # Odwrócenie góra-dół
            fliplr=0.0,  # Odwrócenie lewo-prawo
            mosaic=0.0,  # Augmentacja Mosaic (bardzo ważna, ale ją wyłączamy)
            mixup=0.0,  # Augmentacja MixUp
            copy_paste=0.0  # Augmentacja Copy-Paste
        )
        print("Trening zakończony pomyślnie.")

    except Exception as e:
        print(f"Wystąpił błąd podczas treningu: {e}")
        print("Spróbuj zmniejszyć parametr 'batch' (np. do 2 lub 1), jeśli to błąd pamięci CUDA.")


if __name__ == '__main__':
    trenuj_model()