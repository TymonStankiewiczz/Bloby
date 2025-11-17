import cv2
import numpy as np
import serial
import time
import sys
import threading
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import os.path

# --- KONFIGURACJA PORTU SZEREGOWEGO ---
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# --- KONFIGURACJA KAMERY ---
# USUNIĘTO - teraz klatki przychodzą z ROS

# --- KONFIGURACJA WYŚWIETLANIA TEKSTU (OpenCV) ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_INFO = 1.0
FONT_COLOR_INFO = (0, 255, 0)  # Zielony
FONT_COLOR_RATIO = (0, 255, 255)  # Żółty
FONT_COLOR_MEASURE = (0, 0, 255)  # Czerwony
FONT_THICKNESS = 2

# --- Ścieżki do plików kalibracyjnych ---
CAL_MATRIX_FILE = 'camera_matrix.npy'
DIST_COEFFS_FILE = 'dist_coeffs.npy'

# --- 🎯 STAŁE DO SKALOWANIA ---
BASE_PX_PER_MM = 1.5422
BASE_DISTANCE_MM = 690.0

# --- 🎯 Ścieżka do modelu YOLO ---
YOLO_MODEL_PATH = 'best2.pt'

# --- Plik ustawień dla blobów ---
SETTINGS_FILE = "color_filter_settings.json"


# --- Funkcja dla wątku czytającego port szeregowy ---
# (Bez zmian)
def serial_reader_task(ser, shared_data, lock):
    print("Wątek szeregowy rozpoczęty.")
    while shared_data['running']:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            distance = 0.0
            px_per_mm = 0.0
            status = 0
            if "Distance:" in line:
                parts = line.split(" ")
                if len(parts) >= 2:
                    distance = float(parts[1])
                    if distance > 0:
                        px_per_mm = BASE_PX_PER_MM * (BASE_DISTANCE_MM / distance)
                    status = 0
            elif "Couldn't" in line or "Error" in line:
                status = -1
            with lock:
                if status == -1:
                    shared_data['distance'] = -1.0
                    shared_data['px_per_mm'] = 0.0
                elif distance > 0:
                    shared_data['distance'] = distance
                    shared_data['px_per_mm'] = px_per_mm
        except serial.SerialException as e:
            print(f"Błąd portu szeregowego: {e}")
            with lock:
                shared_data['running'] = False
        except Exception as e:
            pass
    print("Wątek szeregowy zakończony.")


# --- Klasa dla okna debugowania blobów ---
# (Bez zmian)
class BlobDebugWindow(tk.Toplevel):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Blobbin - Widok Debugowania")
        self.geometry("1450x400")
        video_frame = tk.Frame(self)
        video_frame.pack(pady=10)
        self.label_original = tk.Label(video_frame)
        self.label_original.pack(side=tk.LEFT, padx=5)
        self.label_modified = tk.Label(video_frame)
        self.label_modified.pack(side=tk.LEFT, padx=5)
        self.label_annotated = tk.Label(video_frame)
        self.label_annotated.pack(side=tk.LEFT, padx=5)
        self.photo_orig = None
        self.photo_mod = None
        self.photo_ann = None

    def update_feeds(self, orig_img, mod_img, ann_img):
        if not self.winfo_exists():
            return
        try:
            self.photo_orig = self.convert_cv_to_photo(orig_img)
            self.photo_mod = self.convert_cv_to_photo(mod_img)
            self.photo_ann = self.convert_cv_to_photo(ann_img)
            self.label_original.config(image=self.photo_orig)
            self.label_modified.config(image=self.photo_mod)
            self.label_annotated.config(image=self.photo_ann)
        except Exception as e:
            print(f"Błąd aktualizacji okna debug: {e}")

    def convert_cv_to_photo(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(image=img_pil)


# --- Główna klasa aplikacji (WERSJA ROS) ---
class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System Detekcji i Skalowania v1.0 (ROS)")

        # --- ZMIANY: Zmienne klatek dla ROS ---
        self.ros_frame = None  # Surowa klatka z ROS
        self.processed_frame = None  # Klatka po kalibracji i flipie
        self.current_hsv = None  # Klatka HSV
        self.frame_width = 0
        self.frame_height = 0

        # --- Zmienne kalibracji ---
        self.mtx = None
        self.dist = None
        self.new_camera_mtx = None
        self.roi = None

        # --- Zmienne portu szeregowego ---
        self.yolo_model = None
        self.serial_connection = None
        self.serial_thread = None
        self.data_lock = threading.Lock()
        self.shared_data = {
            'distance': 0.0,
            'px_per_mm': 0.0,
            'running': True
        }
        self.current_distance_mm = 0.0
        self.dynamic_px_per_mm = 0.0
        self.text_distance = "Dystans: --- mm"
        self.text_ratio = "Stosunek: --- px/mm"

        # --- Zmienne z 'blobytest.py' ---
        self.DISPLAY_WIDTH = 960
        self.display_height = 0
        self.blob_filters = []
        self.is_picking_color = False
        self.blob_id_counter = 0
        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.settings_window = None
        self.debug_window = None

        # Zmienne ustawień
        self.use_blur = tk.BooleanVar(value=True)
        self.blur_kernel_level = tk.IntVar(value=2)
        self.use_morph_open = tk.BooleanVar(value=True)
        self.morph_open_level = tk.IntVar(value=2)
        self.use_morph_close = tk.BooleanVar(value=True)
        self.morph_close_level = tk.IntVar(value=2)
        self.contour_min_area = tk.IntVar(value=200)

        # Zmienne do okna debugowania
        self.debug_original_frame = None
        self.debug_masked_frame = None

        # --- Inicjalizacja sprzętu i modeli ---
        # ZMIANA: Usunięto initialize_camera
        if not self.initialize_calibration():
            self.root.destroy();
            return
        if not self.initialize_serial():
            self.root.destroy();
            return
        if not self.initialize_yolo():
            self.root.destroy();
            return

        # --- Budowanie GUI ---
        self.create_gui()

        # --- Uruchomienie pętli ---
        self.load_settings()
        self.start_serial_thread()

        # ZMIANA: Czekaj na klatkę z ROS zamiast uruchamiać pętlę
        self.wait_for_first_ros_frame()

    # --- METODY INICJALIZACYJNE ---

    def initialize_calibration(self):
        print("Ładowanie plików kalibracyjnych...")
        try:
            self.mtx = np.load(CAL_MATRIX_FILE)
            self.dist = np.load(DIST_COEFFS_FILE)
            print("Pliki kalibracyjne załadowane.")
            return True
        except Exception as e:
            print(f"BŁĄD: Nie można załadować plików .npy. Błąd: {e}")
            return False

    # USUNIĘTO: initialize_camera()

    def initialize_serial(self):
        print(f"Próba połączenia z {SERIAL_PORT}...")
        try:
            self.serial_connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2)
            print("Połączono z portem COM!")
            return True
        except Exception as e:
            print(f"BŁĄD: Nie można otworzyć portu {SERIAL_PORT}. Błąd: {e}")
            return False

    def initialize_yolo(self):
        print("Ładowanie modelu YOLOv8...")
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print("Model YOLO załadowany.")
            return True
        except Exception as e:
            print(f"BŁĄD: Nie można załadować modelu YOLO '{YOLO_MODEL_PATH}'. Błąd: {e}")
            return False

    def start_serial_thread(self):
        self.serial_thread = threading.Thread(
            target=serial_reader_task,
            args=(self.serial_connection, self.shared_data, self.data_lock),
            daemon=True
        )
        self.serial_thread.start()

    # --- BUDOWANIE GUI ---
    # (Bez zmian)
    def create_gui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_column = tk.Frame(main_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_frame = tk.Frame(left_column, relief=tk.SUNKEN, borderwidth=1)
        video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack()
        self.video_label.bind("<Button-1>", self.on_color_pick)
        info_frame = tk.Frame(left_column)
        info_frame.pack(fill=tk.X, pady=5)
        self.distance_label = tk.Label(info_frame, text="Dystans: --- mm", font=("Arial", 14), fg="green")
        self.distance_label.pack(side=tk.LEFT, padx=10)
        self.ratio_label = tk.Label(info_frame, text="Stosunek: --- px/mm", font=("Arial", 14), fg="blue")
        self.ratio_label.pack(side=tk.LEFT, padx=10)
        controls_frame = tk.Frame(left_column, borderwidth=1, relief=tk.GROOVE)
        controls_frame.pack(fill=tk.X, pady=5)
        self.btn_pick_color = tk.Button(controls_frame, text="Pobierz Kolor", command=self.toggle_picking_mode)
        self.btn_pick_color.pack(side=tk.LEFT, padx=10, pady=5)
        self.btn_settings = tk.Button(controls_frame, text="Ustawienia Przetwarzania",
                                      command=self.open_settings_window)
        self.btn_settings.pack(side=tk.LEFT, padx=10, pady=5)
        self.btn_debug_blobs = tk.Button(controls_frame, text="Debuguj Bloby", command=self.open_blob_debug_window)
        self.btn_debug_blobs.pack(side=tk.LEFT, padx=10, pady=5)
        self.status_label = tk.Label(controls_frame, text="Gotowy.")
        self.status_label.pack(side=tk.LEFT, padx=10)
        right_column = tk.Frame(main_frame, borderwidth=2, relief=tk.RIDGE)
        right_column.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(right_column, text="Filtry Kolorów", font=("Arial", 12, "bold")).pack(pady=5)
        canvas = tk.Canvas(right_column, width=320)
        v_scrollbar = ttk.Scrollbar(right_column, orient="vertical", command=canvas.yview)
        self.blob_controls_frame = tk.Frame(canvas)
        self.blob_controls_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.blob_controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- NOWE METODY DO OBSŁUGI ROS ---

    def wait_for_first_ros_frame(self):
        """
        Czeka na dostarczenie pierwszej klatki przez ROS, aby
        obliczyć macierze kalibracyjne.
        """
        if self.ros_frame is None:
            self.status_label.config(text="Czekam na pierwszą klatkę z ROS...")
            self.root.after(100, self.wait_for_first_ros_frame)
        else:
            print("Pierwsza klatka ROS odebrana.")
            # Oblicz macierze kalibracyjne na podstawie rozmiaru klatki
            self.frame_height, self.frame_width = self.ros_frame.shape[:2]
            print(f"Rozdzielczość klatki ROS: {self.frame_width}x{self.frame_height}")
            print("Obliczanie zoptymalizowanej macierzy kamery...")

            h, w = self.frame_height, self.frame_width
            self.new_camera_mtx, roi_tuple = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
            self.roi = {'x': roi_tuple[0], 'y': roi_tuple[1], 'w': roi_tuple[2], 'h': roi_tuple[3]}

            print("Macierz obliczona. Uruchamianie pętli głównej.")
            self.status_label.config(text="Gotowy.")

            # Uruchom główną pętlę przetwarzania
            self.update_frame()

    def set_frame_from_ros(self, frame):
        """
        Publiczna metoda wywoływana przez węzeł ROS do
        dostarczania nowych klatek wideo.
        """
        self.ros_frame = frame

    # --- PĘTLA GŁÓWNA (`update_frame`) ---

    def update_frame(self):
        try:
            # --- 1. Pobranie i kalibracja klatki z ROS ---
            if self.ros_frame is None:
                # Czekaj, jeśli ROS jeszcze nie dostarczył klatki
                self.root.after(15, self.update_frame)
                return

            # Stwórz lokalną kopię do przetworzenia
            frame = self.ros_frame.copy()

            # --- 1.5 Kalibracja i przygotowanie klatki ---
            # Użyj macierzy obliczonych w wait_for_first_ros_frame
            undistorted_frame = cv2.undistort(frame, self.mtx, self.dist, None, self.new_camera_mtx)
            frame_roi = undistorted_frame[self.roi['y']: self.roi['y'] + self.roi['h'],
                        self.roi['x']: self.roi['x'] + self.roi['w']]

            # `self.processed_frame` to teraz podstawa do wszystkich operacji
            self.processed_frame = cv2.flip(frame_roi, 1)

            # Zapisz kopię dla okna debugowania
            self.debug_original_frame = self.processed_frame.copy()

            # --- 2. Pobranie danych z wątku serial ---
            # (Bez zmian)
            with self.data_lock:
                self.current_distance_mm = self.shared_data['distance']
                self.dynamic_px_per_mm = self.shared_data['px_per_mm']
                if not self.shared_data['running']:
                    print("Wątek szeregowy zatrzymany, zamykanie.")
                    self.on_close()
                    return

            if self.current_distance_mm == -1.0:
                self.text_distance = "Dystans: BŁĄD CZUJNIKA"
            elif self.current_distance_mm > 0:
                self.text_distance = f"Dystans: {self.current_distance_mm:.0f} mm"
            else:
                self.text_distance = "Dystans: --- mm"
            self.text_ratio = f"Stosunek: {self.dynamic_px_per_mm:.2f} px/mm"

            # --- 3. Przetwarzanie dla Blobów ---
            # ZMIANA: Bazuje na self.processed_frame
            if self.use_blur.get():
                k_level = self.blur_kernel_level.get()
                k_size = (k_level * 2) + 1
                blurred_frame = cv2.GaussianBlur(self.processed_frame, (k_size, k_size), 0)
            else:
                blurred_frame = self.processed_frame

            self.current_hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            filtered_image, annotated_image = self.process_blobs()
            self.debug_masked_frame = filtered_image

            # --- 4. Detekcja i Pomiar YOLO ---
            # (Bez zmian, operuje na annotated_image)
            results = self.yolo_model(annotated_image, verbose=False)
            screen_height = annotated_image.shape[0]

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.yolo_model.names[cls]} {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), FONT_COLOR_MEASURE, FONT_THICKNESS)
                cv2.putText(annotated_image, label, (x1, y1 - 10), FONT, 0.7, FONT_COLOR_MEASURE, 2)
                pixel_width = x2 - x1
                pixel_height = y2 - y1
                real_width_mm = 0.0
                real_height_mm = 0.0
                if self.dynamic_px_per_mm > 0:
                    real_width_mm = pixel_width / self.dynamic_px_per_mm
                    real_height_mm = pixel_height / self.dynamic_px_per_mm
                text_width = f"Szer: {real_width_mm:.1f} mm"
                text_height = f"Wys: {real_height_mm:.1f} mm"
                text_y_width = y2 + 25
                text_y_height = y2 + 50
                if text_y_height > screen_height:
                    text_y_width = y1 - 30
                    text_y_height = y1 - 5
                cv2.putText(annotated_image, text_width, (x1, text_y_width), FONT, 0.8, FONT_COLOR_MEASURE, 2)
                cv2.putText(annotated_image, text_height, (x1, text_y_height), FONT, 0.8, FONT_COLOR_MEASURE, 2)

            # --- 5. Rysowanie tekstu informacyjnego ---
            # (Usunięte, zgodnie z poprzednią prośbą)

            # --- 6. Aktualizacja GUI (Etykiety i Obraz) ---
            # (Bez zmian)
            self.distance_label.config(text=self.text_distance)
            self.ratio_label.config(text=self.text_ratio)
            orig_h, orig_w, _ = annotated_image.shape
            self.display_height = int(orig_h * (self.DISPLAY_WIDTH / orig_w))
            frame_display = cv2.resize(annotated_image, (self.DISPLAY_WIDTH, self.display_height))
            img_orig = Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            photo_orig = ImageTk.PhotoImage(image=img_orig)
            self.video_label.config(image=photo_orig)
            self.video_label.image = photo_orig

            # --- 7. Aktualizacja okna DEBUG (jeśli otwarte) ---
            if self.debug_window and self.debug_window.winfo_exists():
                debug_w = 480
                debug_h = int(orig_h * (debug_w / orig_w))
                debug_orig_resized = cv2.resize(self.debug_original_frame, (debug_w, debug_h))
                debug_mask_resized = cv2.resize(self.debug_masked_frame, (debug_w, debug_h))
                debug_ann_resized = cv2.resize(annotated_image, (debug_w, debug_h))
                self.debug_window.update_feeds(debug_orig_resized, debug_mask_resized, debug_ann_resized)

        except Exception as e:
            print(f"Błąd w pętli update_frame: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Pętla odświeża się co 15ms
            self.root.after(15, self.update_frame)

    # --- METODY Z 'blobytest.py' (dostosowane) ---

    def process_blobs(self):
        # ZMIANA: Bazuje na self.processed_frame i self.current_hsv
        if not self.blob_filters or self.current_hsv is None:
            zeros = np.zeros_like(self.processed_frame[:, :, 0])  # Maska 1-kanałowa
            filtered_img = cv2.bitwise_and(self.processed_frame, self.processed_frame, mask=zeros)
            return filtered_img, self.processed_frame.copy()

        h_dim, w_dim = self.current_hsv.shape[:2]
        total_mask = np.zeros((h_dim, w_dim), dtype=np.uint8)
        annotated_image = self.processed_frame.copy()

        k_open_level = self.morph_open_level.get()
        k_close_level = self.morph_close_level.get()
        min_area = self.contour_min_area.get()
        k_open_size = (k_open_level * 2) + 1
        k_close_size = (k_close_level * 2) + 1
        morph_open_kernel = np.ones((k_open_size, k_open_size), np.uint8)
        morph_close_kernel = np.ones((k_close_size, k_close_size), np.uint8)

        for blob in self.blob_filters:
            base_h, base_s, base_v = blob['base_h'], blob['base_s'], blob['base_v']
            h_m, h_p = blob['vars']['h_minus'].get(), blob['vars']['h_plus'].get()
            s_m, s_p = blob['vars']['s_minus'].get(), blob['vars']['s_plus'].get()
            v_m, v_p = blob['vars']['v_minus'].get(), blob['vars']['v_plus'].get()
            h_l, h_h = (base_h - h_m) % 180, (base_h + h_p) % 180
            s_l, s_h = max(0, base_s - s_m), min(255, base_s + s_p)
            v_l, v_h = max(0, base_v - v_m), min(255, base_v + v_p)
            if h_l > h_h:
                lower1, upper1 = np.array([h_l, s_l, v_l]), np.array([179, s_h, v_h])
                lower2, upper2 = np.array([0, s_l, v_l]), np.array([h_h, s_h, v_h])
                mask1 = cv2.inRange(self.current_hsv, lower1, upper1)
                mask2 = cv2.inRange(self.current_hsv, lower2, upper2)
                individual_mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower, upper = np.array([h_l, s_l, v_l]), np.array([h_h, s_h, v_h])
                individual_mask = cv2.inRange(self.current_hsv, lower, upper)
            mask_cleaned = individual_mask
            if self.use_morph_open.get():
                mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, morph_open_kernel)
            if self.use_morph_close.get():
                mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, morph_close_kernel)
            total_mask = cv2.bitwise_or(total_mask, mask_cleaned)
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blob_id = blob['id']
            blob_color = self.color_list[blob_id % len(self.color_list)]
            object_counter = 1
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), blob_color, 2)
                    label = f"Blob {blob_id} Obj {object_counter}"
                    cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blob_color, 2)
                    object_counter += 1

        filtered_image = cv2.bitwise_and(self.processed_frame, self.processed_frame, mask=total_mask)
        return filtered_image, annotated_image

    def toggle_picking_mode(self):
        self.is_picking_color = True
        self.status_label.config(text="TRYB WYBIERANIA: Kliknij wybrany kolor z wideo.")
        self.root.config(cursor="crosshair")

    def on_color_pick(self, event):
        # ZMIANA: Bazuje na self.processed_frame
        if not self.is_picking_color or self.processed_frame is None: return

        orig_h, orig_w, _ = self.processed_frame.shape
        ratio = orig_w / self.DISPLAY_WIDTH

        orig_x, orig_y = int(event.x * ratio), int(event.y * ratio)

        if 0 <= orig_y < orig_h and 0 <= orig_x < orig_w:
            hsv_color = self.current_hsv[orig_y, orig_x]
            h, s, v = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])
            self.add_new_blob_filter(h, s, v)
            self.is_picking_color = False
            self.status_label.config(text=f"Dodano Blob dla H:{h} S:{s} V:{v}.")
            self.root.config(cursor="")
        else:
            self.status_label.config(text="Kliknięcie poza zakresem obrazu.")

    def add_new_blob_filter(self, h, s, v):
        # (Bez zmian)
        blob_id = self.blob_id_counter;
        self.blob_id_counter += 1
        blob_frame = tk.LabelFrame(self.blob_controls_frame, text=f"Blob {blob_id} (Baza: H={h}, S={s}, V={v})",
                                   padx=10, pady=10)
        blob_frame.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=5)
        vars_dict = {
            'h_minus': tk.IntVar(value=10), 'h_plus': tk.IntVar(value=10),
            's_minus': tk.IntVar(value=50), 's_plus': tk.IntVar(value=50),
            'v_minus': tk.IntVar(value=50), 'v_plus': tk.IntVar(value=50),
        }
        self.create_slider(blob_frame, "H -", vars_dict['h_minus'], 0, 45)
        self.create_slider(blob_frame, "H +", vars_dict['h_plus'], 0, 45)
        self.create_slider(blob_frame, "S -", vars_dict['s_minus'], 0, 128)
        self.create_slider(blob_frame, "S +", vars_dict['s_plus'], 0, 128)
        self.create_slider(blob_frame, "V -", vars_dict['v_minus'], 0, 128)
        self.create_slider(blob_frame, "V +", vars_dict['v_plus'], 0, 128)
        btn_delete = tk.Button(blob_frame, text="Usuń", command=lambda b_id=blob_id: self.delete_blob(b_id))
        btn_delete.pack(pady=5, side=tk.BOTTOM)
        self.blob_filters.append({
            'id': blob_id, 'frame': blob_frame, 'vars': vars_dict,
            'base_h': h, 'base_s': s, 'base_v': v,
        })

    def create_slider(self, parent_frame, text, var, from_, to_):
        # (Bez zmian)
        frame = tk.Frame(parent_frame)
        frame.pack(fill=tk.X)
        tk.Label(frame, text=text, width=7).pack(side=tk.LEFT)
        scale = tk.Scale(frame, variable=var, from_=from_, to=to_, orient=tk.HORIZONTAL, length=200)
        scale.pack(fill=tk.X, expand=True)

    def delete_blob(self, blob_id):
        # (Bez zmian)
        blob_to_remove = None
        for blob in self.blob_filters:
            if blob['id'] == blob_id: blob_to_remove = blob; break
        if blob_to_remove:
            blob_to_remove['frame'].destroy()
            self.blob_filters.remove(blob_to_remove)
            self.status_label.config(text=f"Usunięto Blob {blob_id}")

    # --- OKNA I USTAWIENIA ---
    # (Bez zmian)
    def save_settings(self, show_status=True):
        settings_data = {
            'use_blur': self.use_blur.get(),
            'blur_kernel_level': self.blur_kernel_level.get(),
            'use_morph_open': self.use_morph_open.get(),
            'morph_open_level': self.morph_open_level.get(),
            'use_morph_close': self.use_morph_close.get(),
            'morph_close_level': self.morph_close_level.get(),
            'contour_min_area': self.contour_min_area.get()
        }
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings_data, f, indent=4)
            if show_status:
                self.status_label.config(text="Ustawienia pomyślnie zapisano!")
        except Exception as e:
            if show_status:
                self.status_label.config(text=f"Błąd zapisu ustawień: {e}")

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            print(f"Plik '{SETTINGS_FILE}' nie istnieje. Tworzenie domyślnego pliku...")
            self.save_settings(show_status=False)
            self.status_label.config(text="Utworzono domyślny plik.")
            return
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings_data = json.load(f)
            self.use_blur.set(settings_data.get('use_blur', True))
            self.blur_kernel_level.set(settings_data.get('blur_kernel_level', 2))
            self.use_morph_open.set(settings_data.get('use_morph_open', True))
            self.morph_open_level.set(settings_data.get('morph_open_level', 2))
            self.use_morph_close.set(settings_data.get('use_morph_close', True))
            self.morph_close_level.set(settings_data.get('morph_close_level', 2))
            self.contour_min_area.set(settings_data.get('contour_min_area', 200))
            self.status_label.config(text="Ustawienia wczytane z pliku.")
        except Exception as e:
            self.status_label.config(text=f"Błąd wczytywania ustawień: {e}. Używam domyślnych.")

    def open_settings_window(self):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Ustawienia Przetwarzania")
        self.settings_window.protocol("WM_DELETE_WINDOW", self.on_settings_close)
        main_settings_frame = tk.Frame(self.settings_window, padx=10, pady=10)
        main_settings_frame.pack(fill=tk.BOTH, expand=True)
        blur_frame = tk.LabelFrame(main_settings_frame, text="Rozmycie Gaussowskie", padx=10, pady=10)
        blur_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(blur_frame, text="Włącz rozmycie", variable=self.use_blur).pack(anchor=tk.W)
        tk.Label(blur_frame, text="Rozmiar jądra (Kernel 2*n + 1):").pack(anchor=tk.W)
        tk.Scale(blur_frame, variable=self.blur_kernel_level, from_=0, to=10, orient=tk.HORIZONTAL, length=300).pack(
            fill=tk.X)
        morph_frame = tk.LabelFrame(main_settings_frame, text="Morfologia", padx=10, pady=10)
        morph_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(morph_frame, text="Włącz Open", variable=self.use_morph_open).pack(anchor=tk.W)
        tk.Label(morph_frame, text="Rozmiar jądra Open (2*n + 1):").pack(anchor=tk.W)
        tk.Scale(morph_frame, variable=self.morph_open_level, from_=0, to=10, orient=tk.HORIZONTAL, length=300).pack(
            fill=tk.X)
        tk.Checkbutton(morph_frame, text="Włącz Close", variable=self.use_morph_close).pack(anchor=tk.W)
        tk.Label(morph_frame, text="Rozmiar jądra Close (2*n + 1):").pack(anchor=tk.W)
        tk.Scale(morph_frame, variable=self.morph_close_level, from_=0, to=10, orient=tk.HORIZONTAL, length=300).pack(
            fill=tk.X)
        detect_frame = tk.LabelFrame(main_settings_frame, text="Detekcja Konturów", padx=10, pady=10)
        detect_frame.pack(fill=tk.X, pady=5)
        tk.Label(detect_frame, text="Minimalna powierzchnia obiektu (px):").pack(anchor=tk.W)
        tk.Scale(detect_frame, variable=self.contour_min_area, from_=0, to=2000, orient=tk.HORIZONTAL, length=300).pack(
            fill=tk.X)
        button_frame = tk.Frame(main_settings_frame)
        button_frame.pack(fill=tk.X, pady=10)
        btn_revert = tk.Button(button_frame, text="Anuluj i wróć", command=self.load_settings)
        btn_revert.pack(side=tk.RIGHT, padx=5)
        btn_save = tk.Button(button_frame, text="Zapisz", command=self.save_settings)
        btn_save.pack(side=tk.RIGHT, padx=5)

    def on_settings_close(self):
        if self.settings_window:
            self.settings_window.destroy()
        self.settings_window = None

    def open_blob_debug_window(self):
        if self.debug_window is not None and self.debug_window.winfo_exists():
            self.debug_window.lift()
            return
        self.debug_window = BlobDebugWindow(self.root)
        self.debug_window.protocol("WM_DELETE_WINDOW", self.on_blob_debug_close)

    def on_blob_debug_close(self):
        if self.debug_window:
            self.debug_window.destroy()
        self.debug_window = None

    # --- ZAMYKANIE APLIKACJI ---

    def on_close(self):
        print("Rozpoczynanie zamykania...")

        # 1. Zatrzymaj wątek szeregowy
        if self.serial_thread:
            with self.data_lock:
                self.shared_data['running'] = False
            self.serial_thread.join(timeout=1.0)
            print("Wątek szeregowy zamknięty.")

        # 2. Zamknij port szeregowy
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Port szeregowy zamknięty.")

        # 3. Zwolnij kamerę - USUNIĘTE
        # if self.cap:
        #     self.cap.release()
        #     print("Kamera zwolniona.")

        # 4. Zamknij okna
        self.on_settings_close()
        self.on_blob_debug_close()

        # 5. Zniszcz główny wątek Tkinter
        self.root.destroy()
        print("Zasoby zwolnione. Zakończono.")

# USUNIĘTO: if __name__ == "__main__":
#
# Teraz musisz zaimportować tę klasę w swoim skrypcie ROS
# i uruchomić ją w ten sposób:
#
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import tkinter as tk
# import threading
# from nazwa_tego_pliku import CombinedApp
#
# class RosTkinterNode(Node):
#     def __init__(self, app):
#         super().__init__('ros_tkinter_node')
#         self.app = app
#         self.bridge = CvBridge()
#         self.subscription = self.create_subscription(
#             Image,
#             'twoj/temat/obrazu',  # ZMIEŃ NA WŁAŚCIWY TEMAT
#             self.image_callback,
#             10)
#
#     def image_callback(self, msg):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             self.app.set_frame_from_ros(cv_image)
#         except Exception as e:
#             self.get_logger().error(f'Błąd CvBridge: {e}')
#
# def main(args=None):
#     rclpy.init(args=args)
#
#     # Uruchom Tkinter w osobnym wątku
#     root = tk.Tk()
#     app = CombinedApp(root)
#
#     tkinter_thread = threading.Thread(target=root.mainloop, daemon=True)
#     tkinter_thread.start()
#
#     # Uruchom węzeł ROS
#     node = RosTkinterNode(app)
#
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         # Tkinter (jako daemon) zamknie się automatycznie
#
# if __name__ == '__main__':
#     main()