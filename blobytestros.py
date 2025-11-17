import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import os.path


class Blobbin:
    def __init__(self, root):
        self.root = root
        self.root.title("Blobbin v0.6 baj Tymon")

        # Kamera lokalna nie jest używana, obraz przychodzi z ROS2
        self.DISPLAY_WIDTH = 480
        self.display_height = 0

        self.blob_filters = []
        self.is_picking_color = False
        self.current_frame = None
        self.current_hsv = None
        self.blob_id_counter = 0
        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        self.settings_window = None
        self.SETTINGS_FILE = "color_filter_settings.json"

        self.use_blur = tk.BooleanVar(value=True)
        self.blur_kernel_level = tk.IntVar(value=2)
        self.use_morph_open = tk.BooleanVar(value=True)
        self.morph_open_level = tk.IntVar(value=2)
        self.use_morph_close = tk.BooleanVar(value=True)
        self.morph_close_level = tk.IntVar(value=2)
        self.contour_min_area = tk.IntVar(value=200)

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        video_frame = tk.Frame(main_frame)
        video_frame.pack(pady=10)
        self.label_original = tk.Label(video_frame)
        self.label_original.pack(side=tk.LEFT, padx=5)
        self.label_modified = tk.Label(video_frame)
        self.label_modified.pack(side=tk.LEFT, padx=5)
        self.label_annotated = tk.Label(video_frame)
        self.label_annotated.pack(side=tk.LEFT, padx=5)
        self.label_original.bind("<Button-1>", self.on_color_pick)
        controls_frame = tk.Frame(main_frame, borderwidth=2, relief=tk.RIDGE)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        top_controls_frame = tk.Frame(controls_frame)
        top_controls_frame.pack(fill=tk.X, pady=5)
        self.btn_pick_color = tk.Button(top_controls_frame, text="Pobierz Kolor", command=self.toggle_picking_mode)
        self.btn_pick_color.pack(side=tk.LEFT, padx=10)
        self.btn_settings = tk.Button(top_controls_frame, text="Ustawienia", command=self.open_settings_window)
        self.btn_settings.pack(side=tk.LEFT, padx=10)
        self.status_label = tk.Label(top_controls_frame, text="Kliknij przycisk i wybierz kolor z lewego feeda.")
        self.status_label.pack(side=tk.LEFT)
        scroll_container = tk.Frame(controls_frame)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(scroll_container, height=310)
        h_scrollbar = ttk.Scrollbar(scroll_container, orient="horizontal", command=canvas.xview)
        self.blob_controls_frame = tk.Frame(canvas)
        self.blob_controls_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.blob_controls_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_settings()
        self.wait_for_first_ros_frame()

    def wait_for_first_ros_frame(self):
        if self.current_frame is None:
            self.status_label.config(text="Czekam na pierwszy obraz z ROS2...")
            self.root.after(100, self.wait_for_first_ros_frame)
        else:
            self.status_label.config(text="Obraz z ROS2 odebrany. Możesz działać!")
            self.update_frame()

    def set_frame_from_ros(self, frame):
        self.current_frame = cv2.flip(frame, 1)
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
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(settings_data, f, indent=4)
            if show_status:
                self.status_label.config(text="Ustawienia pomyślnie zapisano!")
        except Exception as e:
            if show_status:
                self.status_label.config(text=f"Błąd zapisu ustawień: {e}")

    def load_settings(self):
        if not os.path.exists(self.SETTINGS_FILE):
            print(f"Plik '{self.SETTINGS_FILE}' nie istnieje. Tworzenie domyślnego pliku...")
            self.save_settings(show_status=False)
            self.status_label.config(text="Utworzono domyślny plik.")
            return
        try:
            with open(self.SETTINGS_FILE, 'r') as f:
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
        tk.Checkbutton(morph_frame, text="Włącz Open", variable=self.use_morph_open).pack(
            anchor=tk.W)
        tk.Label(morph_frame, text="Rozmiar jądra Open (2*n + 1):").pack(anchor=tk.W)
        tk.Scale(morph_frame, variable=self.morph_open_level, from_=0, to=10, orient=tk.HORIZONTAL, length=300).pack(
            fill=tk.X)
        tk.Checkbutton(morph_frame, text="Włącz Close", variable=self.use_morph_close).pack(
            anchor=tk.W)
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

    def toggle_picking_mode(self):
        self.is_picking_color = True
        self.status_label.config(text="TRYB WYBIERANIA: Kliknij wybrany kolor z feedu po lewej")
        self.root.config(cursor="crosshair")

    def on_color_pick(self, event):
        if not self.is_picking_color or self.current_frame is None: return
        orig_h, orig_w, _ = self.current_frame.shape
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
        blob_id = self.blob_id_counter;
        self.blob_id_counter += 1
        blob_frame = tk.LabelFrame(self.blob_controls_frame, text=f"Blob {blob_id} (Baza: H={h}, S={s}, V={v})",
                                   padx=10, pady=10)
        blob_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
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
        frame = tk.Frame(parent_frame)
        frame.pack(fill=tk.X)
        tk.Label(frame, text=text, width=7).pack(side=tk.LEFT)
        scale = tk.Scale(frame, variable=var, from_=from_, to=to_, orient=tk.HORIZONTAL, length=250)
        scale.pack(fill=tk.X, expand=True)

    def delete_blob(self, blob_id):
        blob_to_remove = None
        for blob in self.blob_filters:
            if blob['id'] == blob_id: blob_to_remove = blob; break
        if blob_to_remove:
            blob_to_remove['frame'].destroy()
            self.blob_filters.remove(blob_to_remove)
            self.status_label.config(text=f"Usunięto Blob {blob_id}")

    def process_blobs(self):
        if not self.blob_filters or self.current_hsv is None:
            zeros = np.zeros_like(self.current_frame)
            return zeros, self.current_frame.copy()

        h_dim, w_dim = self.current_hsv.shape[:2]
        total_mask = np.zeros((h_dim, w_dim), dtype=np.uint8)
        annotated_image = self.current_frame.copy()

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

        filtered_image = cv2.bitwise_and(self.current_frame, self.current_frame, mask=total_mask)
        return filtered_image, annotated_image

    def update_frame(self):
        try:
            if self.current_frame is None:
                self.status_label.config(text="Brak obrazu do wyświetlenia.")
                self.root.after(100, self.update_frame)
                return
            #self.current_frame = cv2.flip(self.current_frame, 1)
            if self.use_blur.get():
                k_level = self.blur_kernel_level.get()
                k_size = (k_level * 2) + 1
                blurred_frame = cv2.GaussianBlur(self.current_frame, (k_size, k_size), 0)
            else:
                blurred_frame = self.current_frame

            self.current_hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            filtered_image, annotated_image = self.process_blobs()

            orig_h, orig_w, _ = self.current_frame.shape
            self.display_height = int(orig_h * (self.DISPLAY_WIDTH / orig_w))

            frame_display = cv2.resize(self.current_frame, (self.DISPLAY_WIDTH, self.display_height))
            filtered_display = cv2.resize(filtered_image, (self.DISPLAY_WIDTH, self.display_height))
            annotated_display = cv2.resize(annotated_image, (self.DISPLAY_WIDTH, self.display_height))

            img_orig = Image.fromarray(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            photo_orig = ImageTk.PhotoImage(image=img_orig)
            img_mod = Image.fromarray(cv2.cvtColor(filtered_display, cv2.COLOR_BGR2RGB))
            photo_mod = ImageTk.PhotoImage(image=img_mod)
            img_ann = Image.fromarray(cv2.cvtColor(annotated_display, cv2.COLOR_BGR2RGB))
            photo_ann = ImageTk.PhotoImage(image=img_ann)

            self.label_original.config(image=photo_orig)
            self.label_original.image = photo_orig
            self.label_modified.config(image=photo_mod)
            self.label_modified.image = photo_mod
            self.label_annotated.config(image=photo_ann)
            self.label_annotated.image = photo_ann

        except Exception as e:
            print(f"Błąd w pętli update_frame: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.root.after(15, self.update_frame)

    def on_close(self):
        self.on_settings_close()
        #self.cap.release()
        self.root.destroy()


#if __name__ == "__main__":
#    root = tk.Tk()
#    app = Blobbin(root)
#    root.mainloop()