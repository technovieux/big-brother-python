import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import csv
from datetime import datetime

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Big Brother")
        self.iconbitmap("./logo.ico")
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(f"{self.screen_width}x{self.screen_height}")
        self.after(100, lambda: self.state("zoomed"))  # Maximize window after initialization

        # Loading screen
        self.center_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.center_frame.pack(expand=True)
        try:
            self.loading_img = Image.open("./loading_screen.png")
            # Resize to fit window without cropping
            def resize_image(img, max_width, max_height):
                width, height = img.size
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.loading_img = resize_image(self.loading_img, self.screen_width, self.screen_height)
            self.loading_imgtk = ImageTk.PhotoImage(self.loading_img)
            self.loading_label = customtkinter.CTkLabel(self.center_frame, image=self.loading_imgtk, text="")
            self.loading_label.pack(pady=20)
            # Add progress bar
            self.progress = customtkinter.CTkProgressBar(self.center_frame, width=400, height=20)
            self.progress.pack(pady=20)
            self.progress_value = 0
            self.after(0, self.animate_progress)
        except Exception as e:
            self.loading_label = customtkinter.CTkLabel(self.center_frame, text="Chargement de Big Brother...", font=customtkinter.CTkFont(size=40))
            self.loading_label.pack(pady=20)
            self.progress = customtkinter.CTkProgressBar(self.center_frame, width=400, height=20)
            self.progress.pack(pady=20)
            self.progress_value = 0
            self.after(0, self.animate_progress)

        # configure grid layout (4x4)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)



        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        # self.tabview.grid(row=0, column=0, rowspan=2, columnspan=2, padx=10, pady=0, sticky="nsew")
        self.tabview.add("Real-Time Data")
        self.tabview.add("Video")
        self.tabview.add("Image")
        self.tabview.tab("Real-Time Data").grid_columnconfigure((0, 1), weight=1)  # configure grid of individual tabs
        self.tabview.tab("Real-Time Data").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Video").grid_columnconfigure((0, 1, 2), weight=1)
        self.tabview.tab("Video").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Image").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Image").grid_rowconfigure(0, weight=1)

        # Real-Time Data tab
        self.realtime_label = customtkinter.CTkLabel(self.tabview.tab("Real-Time Data"), text="")
        self.realtime_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        self.start_realtime_button = customtkinter.CTkButton(self.tabview.tab("Real-Time Data"), text="Start Real-Time", command=self.start_realtime)
        self.start_realtime_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.stop_realtime_button = customtkinter.CTkButton(self.tabview.tab("Real-Time Data"), state="disabled", text="Stop Real-Time", command=self.stop_realtime)
        self.stop_realtime_button.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        self.cap = None
        self.realtime_running = False

        # Load Haar cascades for detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load YOLO model for person detection
        self.yolo_model = YOLO('yolov8n.pt')

        # Video tab
        self.video_label = customtkinter.CTkLabel(self.tabview.tab("Video"), text="select  a video file", font=customtkinter.CTkFont(size=20))
        self.video_label.grid(row=0, column=0, columnspan=3, padx=20, pady=20, sticky="nsew")
        self.select_video_button = customtkinter.CTkButton(self.tabview.tab("Video"), text="Select Video", command=self.select_video)
        self.select_video_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.play_video_button = customtkinter.CTkButton(self.tabview.tab("Video"), state="disabled", text="Play Video", command=self.play_video)
        self.play_video_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.stop_video_button = customtkinter.CTkButton(self.tabview.tab("Video"), state="disabled", text="Stop Video", command=self.stop_video)
        self.stop_video_button.grid(row=1, column=2, padx=10, pady=10, sticky="ew")
        self.video_cap = None
        self.video_running = False
        self.video_path = None
        self.video_loading = False
        self.video_fps = 25.0
        self.video_frame_index = 0
        self.video_times = []
        self.video_person_counts = []
        self.video_face_counts = []
        self.video_detection_skip = 2
        self.video_last_faces = []
        self.video_last_bodies = []
        self.prev_person_count = 0
        self.prev_face_count = 0
        self.total_person_count = 0
        self.total_face_count = 0
        self.realtime_times = []
        self.realtime_person_counts = []
        self.realtime_face_counts = []
        self.realtime_start_time = 0
        self.image_times = [0]
        self.image_person_counts = [0]
        self.image_face_counts = [0]
        self.current_face_areas = []
        self.current_body_areas = []

        # Image tab
        self.image_label = customtkinter.CTkLabel(self.tabview.tab("Image"), text="Select an image file", font=customtkinter.CTkFont(size=20))
        self.image_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.select_image_button = customtkinter.CTkButton(self.tabview.tab("Image"), text="Select Image", command=self.select_image)
        self.select_image_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.image_path = None
        self.image_loading = False


        self.graph1_frame = customtkinter.CTkFrame(self, fg_color="#2b2b2b")
        # self.graph1_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.graph1_frame.grid_columnconfigure(0, weight=1)
        self.graph1_frame.grid_rowconfigure(0, weight=1)

        self.graph1_fig, self.graph1_ax = plt.subplots(figsize=(6, 4.5), dpi=100)
        self.graph1_ax.set_title("Nombre de personnes et de visages")
        self.graph1_ax.set_xlabel("Temps (s)")
        self.graph1_ax.set_ylabel("Nombre")
        self.graph1_person_line, = self.graph1_ax.plot([], [], color="green", label="Personnes")
        self.graph1_face_line, = self.graph1_ax.plot([], [], color="blue", label="Visages")
        self.graph1_ax.legend(loc="upper right")
        self.graph1_canvas = FigureCanvasTkAgg(self.graph1_fig, master=self.graph1_frame)
        self.graph1_canvas.draw()
        self.graph1_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.graph2_frame = customtkinter.CTkFrame(self, fg_color="#2b2b2b")
        # self.graph2_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        self.graph2_frame.grid_columnconfigure(0, weight=1)
        self.graph2_frame.grid_rowconfigure(0, weight=1)
        self.graph2_fig, self.graph2_ax = plt.subplots(figsize=(6, 4.5), dpi=100)
        self.graph2_ax.set_title("Histogramme des aires de détection")
        self.graph2_ax.set_xlabel("Aire (pixels)")
        self.graph2_ax.set_ylabel("Fréquence")
        self.graph2_canvas = FigureCanvasTkAgg(self.graph2_fig, master=self.graph2_frame)
        self.graph2_canvas.draw()
        self.graph2_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.counter1_frame = customtkinter.CTkFrame(self, fg_color="#2b2b2b")
        # self.counter1_frame.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")
        self.counter1_frame.grid_columnconfigure((0, 1), weight=1)
        self.counter1_frame.grid_rowconfigure((0, 1, 2), weight=1)

        self.counter1_label = customtkinter.CTkLabel(self.counter1_frame, text="actually on screen")
        self.counter1_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.counter1_value1 = customtkinter.CTkLabel(self.counter1_frame, text="0", text_color="#029907", font=customtkinter.CTkFont(size=40, weight="bold"), bg_color="transparent")
        self.counter1_value1.grid(row=1, column=0, padx=0, pady=0)
        self.counter1_value1_label = customtkinter.CTkLabel(self.counter1_frame, text="persons", text_color="#029907", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color="transparent")
        self.counter1_value1_label.grid(row=2, column=0, padx=0, pady=(0, 10))
        self.counter1_value2 = customtkinter.CTkLabel(self.counter1_frame, text="0", text_color="#0000ff", font=customtkinter.CTkFont(size=40, weight="bold"), bg_color="transparent")
        self.counter1_value2.grid(row=1, column=1, padx=20, pady=20)
        self.counter1_value2_label = customtkinter.CTkLabel(self.counter1_frame, text="faces", text_color="#0000ff", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color="transparent")
        self.counter1_value2_label.grid(row=2, column=1, padx=0, pady=(0, 10))




        self.counter2_frame = customtkinter.CTkFrame(self, fg_color="#2b2b2b")
        # self.counter2_frame.grid(row=1, column=2, padx=20, pady=20, sticky="nsew")
        self.counter2_frame.grid_columnconfigure((0, 1), weight=1)
        self.counter2_frame.grid_rowconfigure((0, 1, 2), weight=1)

        self.counter2_label = customtkinter.CTkLabel(self.counter2_frame, text="since the beginning")
        self.counter2_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.counter2_value1 = customtkinter.CTkLabel(self.counter2_frame, text="0", text_color="#029907", font=customtkinter.CTkFont(size=40, weight="bold"), bg_color="transparent")
        self.counter2_value1.grid(row=1, column=0, padx=0, pady=0)
        self.counter2_value1_label = customtkinter.CTkLabel(self.counter2_frame, text="persons", text_color="#029907", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color="transparent")
        self.counter2_value1_label.grid(row=2, column=0, padx=0, pady=(0, 10))
        self.counter2_value2 = customtkinter.CTkLabel(self.counter2_frame, text="0", text_color="#0000ff", font=customtkinter.CTkFont(size=40, weight="bold"), bg_color="transparent")
        self.counter2_value2.grid(row=1, column=1, padx=20, pady=20)
        self.counter2_value2_label = customtkinter.CTkLabel(self.counter2_frame, text="faces", text_color="#0000ff", font=customtkinter.CTkFont(size=20, weight="bold"), bg_color="transparent")
        self.counter2_value2_label.grid(row=2, column=1, padx=0, pady=(0, 10))
        self.counter2_reset = customtkinter.CTkButton(self.counter2_frame, text="reset values", command=self.reset_counters)
        self.counter2_reset.grid(row=3, column=0, padx=10, pady=(0, 10), columnspan=2, sticky="nsew")

        self.export_frame = customtkinter.CTkFrame(self, fg_color="#2b2b2b")
        self.export_frame.grid_columnconfigure(0, weight=1)
        # self.export_frame.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")  # Will be gridded in show_main()
        self.export_label = customtkinter.CTkLabel(self.export_frame, text="Export data")
        self.export_label.grid(row=0, column=0, padx=10, pady=10)
        self.export_choice = customtkinter.CTkOptionMenu(self.export_frame, values=["JSON", "CSV"])
        self.export_choice.grid(row=1, column=0, padx=10, pady=10)
        self.export_button = customtkinter.CTkButton(self.export_frame, text="Export", command=self.export)
        self.export_button.grid(row=2, column=0, padx=10, pady=10)

        self.after(3000, self.show_main)

        #self.attributes("-fullscreen", True)


    def show_main(self):
        self.center_frame.pack_forget()
        self.tabview.grid(row=0, column=0, rowspan=2, columnspan=2, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.graph1_frame.grid(row=2, column=0, padx=(10, 0), pady=(10, 10), sticky="nsew")
        self.graph2_frame.grid(row=2, column=1, padx=(10, 0), pady=(10, 10), sticky="nsew")
        self.counter1_frame.grid(row=0, column=2, padx=(10, 10), pady=(10, 0), sticky="nsew")
        self.counter2_frame.grid(row=1, column=2, padx=(10, 10), pady=(10, 0), sticky="nsew")
        self.export_frame.grid(row=2, column=2, padx=(10, 10), pady=(10, 10), sticky="nsew")

    def animate_progress(self):
        self.progress_value += 0.1
        if self.progress_value > 1:
            self.progress_value = 1
        self.progress.set(self.progress_value)
        self.update()  # Force update to show animation
        if self.progress_value < 1:
            self.after(300, self.animate_progress)

    def start_realtime(self):
        if not self.realtime_running:
            self.cap = cv2.VideoCapture(0)
            self.realtime_running = True
            self.prev_person_count = 0
            self.prev_face_count = 0
            self.total_person_count = 0
            self.total_face_count = 0
            self.realtime_times = []
            self.realtime_person_counts = []
            self.realtime_face_counts = []
            self.realtime_start_time = time.time()
            self.update_graph1()
            self.update_graph2()
            self.reset_counters()
            self.start_realtime_button.configure(state="disabled")
            self.stop_realtime_button.configure(state="normal")
            threading.Thread(target=self.update_realtime, daemon=True).start()

    def stop_realtime(self):
        self.realtime_running = False
        if self.cap:
            self.cap.release()
        self.realtime_label.configure(image="")
        self.start_realtime_button.configure(state="normal")
        self.stop_realtime_button.configure(state="disabled")

    def update_realtime(self):
        while self.realtime_running:
            ret, frame = self.cap.read()
            if ret:
                # Detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                # YOLO for persons
                results = self.yolo_model(frame, classes=[0], verbose=False)
                bodies = []
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w = x2 - x1
                        h = y2 - y1
                        bodies.append((int(x1), int(y1), int(w), int(h)))
                current_face_count = len(faces)
                current_person_count = len(bodies)
                self.current_face_areas = [w * h for (x, y, w, h) in faces]
                self.current_body_areas = [w * h for (x, y, w, h) in bodies]
                face_delta = max(0, current_face_count - self.prev_face_count)
                person_delta = max(0, current_person_count - self.prev_person_count)
                self.total_face_count += face_delta
                self.total_person_count += person_delta
                self.prev_face_count = current_face_count
                self.prev_person_count = current_person_count
                elapsed = time.time() - self.realtime_start_time
                self.realtime_times.append(elapsed)
                self.realtime_person_counts.append(current_person_count)
                self.realtime_face_counts.append(current_face_count)
                print(f"Real-time: {current_face_count} faces, {current_person_count} bodies detected, +{face_delta} / +{person_delta}")  # Debug
                self.after(0, self.update_counters, current_face_count, current_person_count, self.total_face_count, self.total_person_count)
                self.after(0, self.update_graph1)
                self.after(0, self.update_graph2)
                
                # Draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)  # Blue for faces, thicker
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)  # Green for bodies, thicker
                
                # Resize to fit label if necessary
                height, width = frame.shape[:2]
                label_w = max(1, self.realtime_label.winfo_width() - 40)
                label_h = max(1, self.realtime_label.winfo_height() - 40)
                if width > label_w or height > label_h:
                    ratio = min(label_w / width, label_h / height)
                    new_w = int(width * ratio)
                    new_h = int(height * ratio)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.realtime_label.configure(image=imgtk)
                self.realtime_label.image = imgtk
            time.sleep(0.03)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.video_label.configure(text="Loading video file...", image="")
            self.video_loading = True
            threading.Thread(target=self.load_video_file, args=(self.video_path,), daemon=True).start()

    def load_video_file(self, path):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            cap.release()
            self.after(0, lambda: self.video_label.configure(text="", image=""))
            self.play_video_button.configure(state="normal")
            self.stop_video_button.configure(state="normal")
        else:
            self.after(0, lambda: self.video_label.configure(text="Error loading video file", image=""))
            self.video_path = None
        self.video_loading = False

    def play_video(self):
        if self.video_path and not self.video_running and not self.video_loading:
            self.video_cap = cv2.VideoCapture(self.video_path)
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.video_frame_index = 0
            self.video_times = []
            self.video_person_counts = []
            self.video_face_counts = []
            self.video_last_faces = []
            self.video_last_bodies = []
            self.prev_person_count = 0
            self.prev_face_count = 0
            self.total_person_count = 0
            self.total_face_count = 0
            self.update_graph1()
            self.update_graph2()
            self.update_counters(0, 0, 0, 0)
            self.video_running = True
            self.play_video_button.configure(state="disabled")
            self.stop_video_button.configure(state="normal")
            threading.Thread(target=self.update_video, daemon=True).start()

    def stop_video(self):
        self.video_running = False
        if self.video_cap:
            self.video_cap.release()
        self.video_label.configure(image="")
        self.play_video_button.configure(state="normal")
        self.stop_video_button.configure(state="disabled")

    def update_video(self):
        while self.video_running:
            frame_start = time.perf_counter()
            ret, frame = self.video_cap.read()
            if ret:
                detect_this_frame = (self.video_frame_index % self.video_detection_skip) == 0
                if detect_this_frame:
                    # Detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    # YOLO for persons
                    results = self.yolo_model(frame, classes=[0], verbose=False)
                    bodies = []
                    if results and len(results) > 0:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            w = x2 - x1
                            h = y2 - y1
                            bodies.append((int(x1), int(y1), int(w), int(h)))
                    self.video_last_faces = faces
                    self.video_last_bodies = bodies
                else:
                    faces = self.video_last_faces
                    bodies = self.video_last_bodies

                current_face_count = len(faces)
                current_person_count = len(bodies)
                self.current_face_areas = [w * h for (x, y, w, h) in faces]
                self.current_body_areas = [w * h for (x, y, w, h) in bodies]
                face_delta = max(0, current_face_count - self.prev_face_count)
                person_delta = max(0, current_person_count - self.prev_person_count)
                self.total_face_count += face_delta
                self.total_person_count += person_delta
                self.prev_face_count = current_face_count
                self.prev_person_count = current_person_count
                print(f"Video: {current_face_count} faces, {current_person_count} bodies detected, +{face_delta} / +{person_delta}")  # Debug
                self.after(0, self.update_counters, current_face_count, current_person_count, self.total_face_count, self.total_person_count)
                
                # Draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

                elapsed = self.video_frame_index / self.video_fps if self.video_fps else 0.0
                self.video_times.append(elapsed)
                self.video_person_counts.append(len(bodies))
                self.video_face_counts.append(len(faces))
                self.video_frame_index += 1
                if detect_this_frame:
                    self.after(0, self.update_graph1)
                    self.after(0, self.update_graph2)
                
                # Resize to fit label if necessary
                height, width = frame.shape[:2]
                label_w = max(1, self.video_label.winfo_width() - 40)
                label_h = max(1, self.video_label.winfo_height() - 40)
                if width > label_w or height > label_h:
                    ratio = min(label_w / width, label_h / height)
                    new_w = int(width * ratio)
                    new_h = int(height * ratio)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
            else:
                self.stop_video()
                break

            frame_elapsed = time.perf_counter() - frame_start
            frame_time = 1.0 / self.video_fps if self.video_fps else 0.03
            if frame_elapsed < frame_time:
                time.sleep(frame_time - frame_elapsed)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif")])
        if self.image_path:
            self.select_image_button.configure(state="disabled", text="Loading...")
            self.image_label.configure(text="file loading, please wait...", image="")
            self.image_loading = True
            threading.Thread(target=self.load_image_file, args=(self.image_path,), daemon=True).start()

    def load_image_file(self, path):
        img_cv = cv2.imread(path)
        if img_cv is None:
            self.after(0, lambda: self.image_label.configure(text="Erreur de chargement du media", image=""))
            self.image_loading = False
            self.select_image_button.configure(state="normal", text="Select Image")
            return

        # Detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        # YOLO for persons
        results = self.yolo_model(img_cv, classes=[0], verbose=False)
        bodies = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                bodies.append((int(x1), int(y1), int(w), int(h)))
        self.current_face_areas = [w * h for (x, y, w, h) in faces]
        self.current_body_areas = [w * h for (x, y, w, h) in bodies]
        print(f"Image: {len(faces)} faces, {len(bodies)} bodies detected")  # Debug
        self.total_face_count = len(faces)
        self.total_person_count = len(bodies)
        self.image_person_counts = [len(bodies)]
        self.image_face_counts = [len(faces)]
        self.after(0, self.update_counters, len(faces), len(bodies), self.total_face_count, self.total_person_count)
        self.after(0, self.update_graph1)  # Update graph with image data
        self.after(0, self.update_graph2)

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 4)
        for (x, y, w, h) in bodies:
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # Convert to PIL
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_cv)

        label_w = max(1, self.image_label.winfo_width() - 40)
        label_h = max(1, self.image_label.winfo_height() - 40)
        if img.width > label_w or img.height > label_h:
            ratio = min(label_w / img.width, label_h / img.height)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))

        imgtk = ImageTk.PhotoImage(img)
        def update_label():
            self.image_label.configure(image=imgtk)
            self.image_label.configure(text="")
            self.image_label.image = imgtk
        self.after(0, update_label)
        self.image_loading = False
        self.select_image_button.configure(state="normal", text="Select Image")

    def update_graph1(self):
        if self.realtime_running:
            self.graph1_person_line.set_data(self.realtime_times, self.realtime_person_counts)
            self.graph1_face_line.set_data(self.realtime_times, self.realtime_face_counts)
            self.graph1_ax.set_title("real-time detections")
        elif self.video_running:
            self.graph1_person_line.set_data(self.video_times, self.video_person_counts)
            self.graph1_face_line.set_data(self.video_times, self.video_face_counts)
            self.graph1_ax.set_title("Video detections")
        else:
            self.graph1_person_line.set_data(self.image_times, self.image_person_counts)
            self.graph1_face_line.set_data(self.image_times, self.image_face_counts)
            self.graph1_ax.set_title("Image detections")
        self.graph1_ax.relim()
        self.graph1_ax.autoscale_view()
        self.graph1_canvas.draw()

    def update_graph2(self):
        self.graph2_ax.clear()
        if self.current_face_areas:
            self.graph2_ax.hist(self.current_face_areas, bins=10, color='blue', alpha=0.7, label='Visages')
        if self.current_body_areas:
            self.graph2_ax.hist(self.current_body_areas, bins=10, color='green', alpha=0.7, label='Personnes')
        self.graph2_ax.set_title("Histogram of detection areas")
        self.graph2_ax.set_xlabel("Area (pixels)")
        self.graph2_ax.set_ylabel("Frequency")
        if self.current_face_areas or self.current_body_areas:
            self.graph2_ax.legend()
        self.graph2_canvas.draw()

    def update_counters(self, faces_count, bodies_count, total_faces_count, total_bodies_count):
        self.counter1_value1.configure(text=str(bodies_count))
        self.counter1_value2.configure(text=str(faces_count))
        self.counter2_value1.configure(text=str(int(total_bodies_count)))
        self.counter2_value2.configure(text=str(int(total_faces_count)))

    def get_current_mode(self):
        """Détecte le mode actuel (Real-Time, Video, ou Image)"""
        if self.realtime_running:
            return "realtime"
        elif self.video_running:
            return "video"
        else:
            return "image"

    def export(self):
        """data export function for JSON and CSV formats"""
        format_choice = self.export_choice.get()
        mode = self.get_current_mode()
        
        if mode == "realtime":
            times = self.realtime_times
            person_counts = self.realtime_person_counts
            face_counts = self.realtime_face_counts
        elif mode == "video":
            times = self.video_times
            person_counts = self.video_person_counts
            face_counts = self.video_face_counts
        else:  # image
            times = self.image_times
            person_counts = self.image_person_counts
            face_counts = self.image_face_counts
        
        if not times:
            tkinter.messagebox.showwarning("Error", "No data to export for the current mode.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{mode}_{timestamp}"
        
        if format_choice == "JSON":
            self.export_json(filename, times, person_counts, face_counts, mode)
        elif format_choice == "CSV":
            self.export_csv(filename, times, person_counts, face_counts, mode)

    def export_json(self, filename, times, person_counts, face_counts, mode):
        """Exporte les données en JSON"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=filename + ".json"
        )
        if not file_path:
            return
        
        data = {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "data": [
                {
                    "time": t,
                    "persons": p,
                    "faces": f
                }
                for t, p, f in zip(times, person_counts, face_counts)
            ],
            "summary": {
                "total_persons": sum(person_counts),
                "total_faces": sum(face_counts),
                "duration": times[-1] if times else 0
            }
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            tkinter.messagebox.showinfo("Succès", f"Données exportées en JSON:\n{file_path}")
        except Exception as e:
            tkinter.messagebox.showerror("Erreur", f"Erreur lors de l'export: {str(e)}")

    def export_csv(self, filename, times, person_counts, face_counts, mode):
        """Exporte les données en CSV"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=filename + ".csv"
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Mode", mode])
                writer.writerow(["Timestamp", datetime.now().isoformat()])
                writer.writerow([])
                writer.writerow(["Time (s)", "Persons", "Faces"])
                for t, p, fa in zip(times, person_counts, face_counts):
                    writer.writerow([t, p, fa])
                writer.writerow([])
                writer.writerow(["Summary", ""])
                writer.writerow(["Total Persons", sum(person_counts)])
                writer.writerow(["Total Faces", sum(face_counts)])
                writer.writerow(["Duration", times[-1] if times else 0])
            tkinter.messagebox.showinfo("Succès", f"Données exportées en CSV:\n{file_path}")
        except Exception as e:
            tkinter.messagebox.showerror("Erreur", f"Erreur lors de l'export: {str(e)}")

    def reset_counters(self):
        self.update_counters(0, 0, 0, 0)
        self.total_face_count = 0
        self.total_person_count = 0


if __name__ == "__main__":
    

    app = App()
    app.mainloop()
