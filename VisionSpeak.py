import queue
import sys
import threading
import time
import tkinter as tk

import gestures
from sign_camera import SignCamera, SignCameraError

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# -------------------------------------------------------------------- theme
BG = "#0f172a"
SIDEBAR = "#0b1120"
CARD = "#1e293b"
CARD_ALT = "#273449"
BORDER = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#7c5cff"
ACCENT_2 = "#38bdf8"
SUCCESS = "#22c55e"
DANGER = "#ef4444"
WARNING = "#f59e0b"

FONT = "Segoe UI"

# Google speech recogniser languages shown on the Speech to Text page.
LANGUAGES = {
    "English (India)": "en-IN",
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "Hindi": "hi-IN",
}

CAMERA_OFF_TEXT = ("Camera is off. Press “Start Camera” to begin.\n\n"
                   "Tip: keep one hand in view, palm facing the camera, in good light.")


def lighten(hex_color, amount=0.15):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_button(parent, text, command, color=ACCENT, size=12, **kwargs):
    btn = tk.Button(parent, text=text, command=command, font=(FONT, size, "bold"),
                    fg="white", activeforeground="white", disabledforeground=MUTED,
                    relief="flat", bd=0, padx=16, pady=8, cursor="hand2", takefocus=0, **kwargs)
    btn.bind("<Enter>", lambda e: btn.config(bg=lighten(btn.color)) if btn["state"] != "disabled" else None)
    btn.bind("<Leave>", lambda e: style_button(btn))
    style_button(btn, color)
    return btn


def style_button(btn, color=None, enabled=None):
    """Change a button's colour and/or enabled state; disabled buttons look dimmed."""
    if color is not None:
        btn.color = color
    if enabled is not None:
        btn.config(state="normal" if enabled else "disabled")
    disabled = btn["state"] == "disabled"
    btn.config(bg=CARD_ALT if disabled else btn.color, activebackground=lighten(btn.color),
               cursor="arrow" if disabled else "hand2")


def make_card(parent, **kwargs):
    return tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1, **kwargs)


def card_title(parent, text):
    return tk.Label(parent, text=text, font=(FONT, 11, "bold"), bg=CARD, fg=MUTED, anchor="w")


class Speaker:
    """
    Text to speech on a background thread, so speaking never freezes the UI.

    On Windows the SAPI voice is driven directly through comtypes (installed
    with pyttsx3), because pyttsx3's runAndWait() only speaks the first time
    on Windows and stays silent afterwards. Elsewhere a fresh pyttsx3 engine
    is used for every message for the same reason.
    """

    def __init__(self, post):
        self.post = post  # thread-safe way to run a callback on the Tk thread
        self.error = None
        if sys.platform == "win32":
            try:
                import comtypes.client  # noqa: F401
            except ImportError:
                self.error = "comtypes is missing (pip install pyttsx3)."
        elif pyttsx3 is None:
            self.error = "pyttsx3 is not installed."
        self._queue = queue.Queue()
        if self.error is None:
            threading.Thread(target=self._run, name="Speaker", daemon=True).start()

    @property
    def available(self):
        return self.error is None

    def say(self, text, on_done):
        """Queue `text`; on_done(error_or_None) is called on the Tk thread afterwards."""
        self._queue.put((text, on_done))

    def _run(self):
        voice = None
        if sys.platform == "win32":
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()  # COM must be initialised on every thread that uses it
        while True:
            text, on_done = self._queue.get()
            error = None
            try:
                if sys.platform == "win32":
                    if voice is None:
                        voice = comtypes.client.CreateObject("SAPI.SpVoice")
                    voice.Speak(text)
                else:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
            except Exception as exc:
                voice = None
                error = f"Speech failed: {exc}"
            self.post(on_done, error)


class VisionSpeakApp:
    def __init__(self, window):
        self.window = window
        window.title("VisionSpeak")
        window.geometry("1200x780")
        window.minsize(1050, 700)
        window.configure(bg=BG)

        # Background threads hand callbacks to the Tk thread through this queue.
        self.ui_calls = queue.Queue()
        self.speaker = Speaker(self.post)

        self.camera = SignCamera()
        self.camera_job = None
        self.listen_stop = threading.Event()
        self.listening = False

        self.sentence = []
        self.stable_gesture = None
        self.hold = gestures.HoldToAdd(hold_seconds=1.5)
        self.space_down = False
        self.video_image = None

        self.build_sidebar()
        self.content = tk.Frame(window, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)

        self.pages = {
            "speech": self.build_speech_page(),
            "tts": self.build_tts_page(),
            "sign": self.build_sign_page(),
        }
        self.current_page = None
        self.show_page("speech")

        window.bind("<KeyPress-space>", self.on_space_press)
        window.bind("<KeyRelease-space>", lambda e: setattr(self, "space_down", False))
        window.bind("<BackSpace>", self.on_backspace_key)
        window.protocol("WM_DELETE_WINDOW", self.on_close)
        self._pump_ui_calls()

    def post(self, func, *args):
        """Run func(*args) on the Tk main thread. Safe to call from any thread."""
        self.ui_calls.put((func, args))

    def _pump_ui_calls(self):
        try:
            while True:
                func, args = self.ui_calls.get_nowait()
                func(*args)
        except queue.Empty:
            pass
        self.window.after(50, self._pump_ui_calls)

    # ------------------------------------------------------------ navigation
    def build_sidebar(self):
        sidebar = tk.Frame(self.window, bg=SIDEBAR, width=250)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🧠 VisionSpeak", font=(FONT, 17, "bold"), bg=SIDEBAR, fg="white"
                 ).pack(anchor="w", padx=20, pady=(26, 2))
        tk.Label(sidebar, text="Assistive communication", font=(FONT, 10), bg=SIDEBAR, fg=MUTED
                 ).pack(anchor="w", padx=20, pady=(0, 24))

        self.nav_buttons = {}
        for key, label in (("speech", "🎤  Speech to Text"), ("tts", "🔊  Text to Speech"),
                           ("sign", "✋  Sign to Text")):
            btn = tk.Button(sidebar, text=label, font=(FONT, 12), anchor="w", relief="flat", bd=0,
                            padx=20, pady=12, cursor="hand2", takefocus=0, bg=SIDEBAR, fg=TEXT,
                            activebackground=CARD, activeforeground="white",
                            command=lambda k=key: self.show_page(k))
            btn.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[key] = btn

        tk.Label(sidebar, text="Gesture recognition runs\nfully offline on this PC.",
                 font=(FONT, 9), bg=SIDEBAR, fg=MUTED, justify="left"
                 ).pack(side="bottom", anchor="w", padx=20, pady=20)

    def show_page(self, key):
        if self.current_page == "sign" and key != "sign":
            self.stop_camera()
        if self.current_page:
            self.pages[self.current_page].pack_forget()
        self.pages[key].pack(fill="both", expand=True, padx=28, pady=24)
        self.current_page = key
        for name, btn in self.nav_buttons.items():
            active = name == key
            btn.config(bg=ACCENT if active else SIDEBAR, fg="white" if active else TEXT,
                       font=(FONT, 12, "bold" if active else "normal"))

    def page_header(self, page, title, subtitle):
        tk.Label(page, text=title, font=(FONT, 22, "bold"), bg=BG, fg="white", anchor="w").pack(fill="x")
        tk.Label(page, text=subtitle, font=(FONT, 11), bg=BG, fg=MUTED, anchor="w").pack(fill="x", pady=(2, 18))

    # -------------------------------------------------------- speech to text
    def build_speech_page(self):
        page = tk.Frame(self.content, bg=BG)
        self.page_header(page, "🎤 Speech to Text", "Speak into your microphone and the words appear below.")

        controls = tk.Frame(page, bg=BG)
        controls.pack(fill="x", pady=(0, 12))
        self.listen_button = make_button(controls, "🎤  Start Listening", self.toggle_listening, SUCCESS, 13)
        self.listen_button.pack(side="left")
        make_button(controls, "Clear Text", self.clear_text, DANGER).pack(side="left", padx=10)

        self.language = tk.StringVar(value=next(iter(LANGUAGES)))
        lang_menu = tk.OptionMenu(controls, self.language, *LANGUAGES)
        lang_menu.config(font=(FONT, 11), bg=CARD_ALT, fg=TEXT, activebackground=CARD, activeforeground=TEXT,
                         relief="flat", bd=0, highlightthickness=0, cursor="hand2", takefocus=0)
        lang_menu["menu"].config(font=(FONT, 11), bg=CARD, fg=TEXT, activebackground=ACCENT)
        lang_menu.pack(side="left")

        self.status_label = tk.Label(controls, text="", font=(FONT, 12), bg=BG, fg=ACCENT_2)
        self.status_label.pack(side="left", padx=10)

        card = make_card(page)
        card.pack(fill="both", expand=True)
        card_title(card, "TRANSCRIPT").pack(fill="x", padx=16, pady=(12, 4))
        self.result_text = tk.Text(card, font=(FONT, 14), bg=CARD, fg=TEXT, relief="flat", wrap="word",
                                   insertbackground=TEXT, padx=6, pady=6, state="disabled")
        self.result_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        tk.Label(page, text="Listening continues until you press Stop. Each sentence appears as soon as you pause.",
                 font=(FONT, 10), bg=BG, fg=MUTED, anchor="w").pack(fill="x", pady=(8, 0))
        return page

    def toggle_listening(self):
        if self.listening:
            self.stop_listening()
            return
        if sr is None:
            self.set_speech_status("⚠️ SpeechRecognition is not installed.")
            return
        self.listening = True
        self.listen_stop = threading.Event()  # a fresh flag, so an old worker still shutting down stays stopped
        self.listen_button.config(text="⏹  Stop Listening")
        style_button(self.listen_button, DANGER)
        self.set_speech_status("🎧 Adjusting to background noise…")
        threading.Thread(target=self._listen_worker, args=(self.listen_stop, LANGUAGES[self.language.get()]),
                         name="Listener", daemon=True).start()

    def stop_listening(self):
        self.listen_stop.set()
        self.listening = False
        self.listen_button.config(text="🎤  Start Listening")
        style_button(self.listen_button, SUCCESS)
        if not self.status_label.cget("text").startswith("⚠️"):
            self.set_speech_status("")

    def set_speech_status(self, text):
        self.status_label.config(text=text, fg=WARNING if text.startswith("⚠️") else ACCENT_2)

    def _listen_worker(self, stop, language):
        """
        Background thread: keeps the microphone open and captures one phrase at
        a time. Phrases go to a separate recognition thread, so the microphone
        keeps listening while Google transcribes the previous sentence.
        """
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True  # keep adapting to room noise
        recognizer.pause_threshold = 0.6            # end a phrase after 0.6 s of silence (default 0.8)
        recognizer.non_speaking_duration = 0.3
        recognizer.phrase_threshold = 0.2           # ignore clicks shorter than this
        phrases = queue.Queue()
        threading.Thread(target=self._recognize_worker, args=(stop, recognizer, phrases, language),
                         name="Recognizer", daemon=True).start()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                self.post(self._listening_status, stop)
                while not stop.is_set():
                    try:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                    except sr.WaitTimeoutError:
                        continue  # silence: check the stop flag and keep listening
                    if not stop.is_set():
                        self.post(self._listening_status, stop, "⏳ Recognising…")
                        phrases.put(audio)
        except (OSError, AttributeError):
            self.post(self._listen_failed, stop, "⚠️ No microphone found (is PyAudio installed?).")
        finally:
            phrases.put(None)

    def _recognize_worker(self, stop, recognizer, phrases, language):
        # One phrase at a time, so sentences appear in the order they were spoken.
        while (audio := phrases.get()) is not None:
            try:
                text = recognizer.recognize_google(audio, language=language)
                self.post(self._got_transcript, stop, text)
            except sr.UnknownValueError:
                self.post(self._listening_status, stop, "⚠️ Didn't catch that, please repeat.")
            except sr.RequestError:
                stop.set()
                self.post(self._listen_failed, stop, "⚠️ Check your internet connection.")
                return

    def _listening_status(self, stop, text="🎤 Listening… speak now"):
        if stop is self.listen_stop and self.listening:
            self.set_speech_status(text)

    def _got_transcript(self, stop, text):
        self.append_transcript(text)
        self._listening_status(stop)

    def _listen_failed(self, stop, message):
        if stop is not self.listen_stop:
            return
        if self.listening:
            self.stop_listening()
        self.set_speech_status(message)

    def append_transcript(self, text):
        self.result_text.config(state="normal")
        if self.result_text.get("1.0", "end-1c"):
            self.result_text.insert("end", "\n")
        self.result_text.insert("end", text)
        self.result_text.see("end")
        self.result_text.config(state="disabled")

    def clear_text(self):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.config(state="disabled")
        self.status_label.config(text="")

    # -------------------------------------------------------- text to speech
    def build_tts_page(self):
        page = tk.Frame(self.content, bg=BG)
        self.page_header(page, "🔊 Text to Speech", "Type a message and VisionSpeak will read it aloud.")

        card = make_card(page)
        card.pack(fill="x")
        card_title(card, "YOUR MESSAGE").pack(fill="x", padx=16, pady=(12, 4))
        self.entry = tk.Text(card, height=6, font=(FONT, 14), bg=CARD_ALT, fg=TEXT, relief="flat",
                             wrap="word", insertbackground=TEXT, padx=10, pady=8)
        self.entry.pack(fill="x", padx=16)
        self.entry.bind("<Control-Return>", lambda e: (self.speak(), "break")[1])

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=16, pady=14)
        make_button(row, "🔊  Convert to Speech", self.speak, ACCENT_2, 13).pack(side="left")
        tk.Label(row, text="Tip: Ctrl+Enter to speak", font=(FONT, 10), bg=CARD, fg=MUTED).pack(side="left", padx=14)
        self.tts_status = tk.Label(row, text="", font=(FONT, 11), bg=CARD, fg=WARNING)
        self.tts_status.pack(side="right")
        return page

    def speak(self):
        self.say(self.entry.get("1.0", "end-1c"), self.tts_status)

    def say(self, text, status_label):
        text = text.strip()
        if not text:
            return
        if not self.speaker.available:
            status_label.config(text=f"⚠️ Text to speech unavailable: {self.speaker.error}", fg=WARNING)
            return
        status_label.config(text="🔊 Speaking…", fg=ACCENT_2)

        def done(error):
            if error:
                status_label.config(text="⚠️ " + error, fg=WARNING)
            elif status_label.cget("text") == "🔊 Speaking…":
                status_label.config(text="")
        self.speaker.say(text, done)

    # ---------------------------------------------------------- sign to text
    def build_sign_page(self):
        page = tk.Frame(self.content, bg=BG)
        self.page_header(page, "✋ Sign to Text",
                         "Prototype: recognises a small set of static hand gestures, not full sign language.")

        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left: camera feed + sentence
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        video_card = make_card(left)
        video_card.pack(fill="both", expand=True)
        self.video_label = tk.Label(video_card, bg="#020617", fg=MUTED, font=(FONT, 12),
                                    text=CAMERA_OFF_TEXT)
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        cam_row = tk.Frame(left, bg=BG)
        cam_row.pack(fill="x", pady=10)
        self.camera_button = make_button(cam_row, "📷  Start Camera", self.toggle_camera, SUCCESS)
        self.camera_button.pack(side="left")
        self.camera_status = tk.Label(cam_row, text="", font=(FONT, 10), bg=BG, fg=MUTED, justify="left")
        self.camera_status.pack(side="left", padx=12)

        sentence_card = make_card(left)
        sentence_card.pack(fill="x")
        card_title(sentence_card, "SENTENCE").pack(fill="x", padx=16, pady=(10, 0))
        self.sentence_label = tk.Label(sentence_card, text="", font=(FONT, 17, "bold"), bg=CARD, fg="white",
                                       anchor="w", justify="left", wraplength=620)
        self.sentence_label.pack(fill="x", padx=16, pady=(2, 8))
        row = tk.Frame(sentence_card, bg=CARD)
        row.pack(fill="x", padx=16, pady=(0, 12))
        make_button(row, "⌫ Backspace", self.backspace_word, CARD_ALT, 11).pack(side="left")
        make_button(row, "Clear", self.clear_sentence, DANGER, 11).pack(side="left", padx=8)
        make_button(row, "🔊 Speak", self.speak_sentence, ACCENT_2, 11).pack(side="left")
        self.sign_tts_status = tk.Label(row, text="", font=(FONT, 10), bg=CARD, fg=WARNING)
        self.sign_tts_status.pack(side="left", padx=8)

        # Right: detection + controls + gesture guide
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        detect = make_card(right)
        detect.pack(fill="x")
        card_title(detect, "DETECTED GESTURE").pack(fill="x", padx=16, pady=(12, 0))
        self.gesture_label = tk.Label(detect, text="—", font=(FONT, 22, "bold"), bg=CARD, fg=MUTED, anchor="w")
        self.gesture_label.pack(fill="x", padx=16)
        self.gesture_detail = tk.Label(detect, text="", font=(FONT, 10), bg=CARD, fg=MUTED, anchor="w")
        self.gesture_detail.pack(fill="x", padx=16)

        self.conf_bar = tk.Canvas(detect, height=8, bg=CARD_ALT, highlightthickness=0)
        self.conf_bar.pack(fill="x", padx=16, pady=(8, 4))
        self.conf_fill = self.conf_bar.create_rectangle(0, 0, 0, 8, fill=ACCENT, width=0)

        self.add_button = make_button(detect, "➕  Add Sign   (Space)", self.add_current_sign, ACCENT, 12)
        self.add_button.pack(fill="x", padx=16, pady=(8, 4))

        self.auto_add = tk.BooleanVar(value=False)
        tk.Checkbutton(detect, text="Auto-add when held for 1.5 s", variable=self.auto_add,
                       font=(FONT, 10), bg=CARD, fg=TEXT, selectcolor=CARD_ALT, activebackground=CARD,
                       activeforeground=TEXT, takefocus=0, command=self.hold.reset).pack(anchor="w", padx=12)
        self.hold_bar = tk.Canvas(detect, height=4, bg=CARD_ALT, highlightthickness=0)
        self.hold_bar.pack(fill="x", padx=16, pady=(2, 12))
        self.hold_fill = self.hold_bar.create_rectangle(0, 0, 0, 4, fill=SUCCESS, width=0)

        guide = make_card(right)
        guide.pack(fill="both", expand=True, pady=(16, 0))
        card_title(guide, "SUPPORTED GESTURES").pack(fill="x", padx=16, pady=(10, 4))
        # Two compact columns so the whole vocabulary fits; hovering shows how to make a gesture.
        grid = tk.Frame(guide, bg=CARD)
        grid.pack(fill="x", padx=12)
        grid.columnconfigure((0, 1), weight=1, uniform="guide")
        guide_hint = tk.Label(guide, text="Hover over a gesture to see how to make it.", font=(FONT, 9),
                              bg=CARD, fg=MUTED, anchor="nw", justify="left", wraplength=360)
        guide_hint.pack(fill="both", expand=True, padx=16, pady=(6, 8))
        self.guide_rows = {}
        for i, (name, word) in enumerate(gestures.GESTURE_WORDS.items()):
            row = tk.Frame(grid, bg=CARD)
            row.grid(row=i // 2, column=i % 2, sticky="ew", padx=2)
            word_lbl = tk.Label(row, text=word, font=(FONT, 10, "bold"), bg=CARD, fg=TEXT, anchor="w")
            word_lbl.pack(side="left", padx=(4, 0))
            hint_lbl = tk.Label(row, text=name.replace("_", " ").lower(), font=(FONT, 9), bg=CARD, fg=MUTED, anchor="w")
            hint_lbl.pack(side="left", padx=6)
            hint = f"{word}: {gestures.GESTURE_HINTS[name]}"
            for widget in (row, word_lbl, hint_lbl):
                widget.bind("<Enter>", lambda e, h=hint: guide_hint.config(text=h, fg=TEXT))
            self.guide_rows[name] = (row, word_lbl, hint_lbl)

        self.set_gesture_display(None, None, 0.0, camera_on=False)
        return page

    def toggle_camera(self):
        if self.camera.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        try:
            self.camera.start()
        except SignCameraError as exc:
            self.show_camera_error(str(exc))
            return
        self.hold.reset()
        self.camera_button.config(text="⏹  Stop Camera")
        style_button(self.camera_button, DANGER)
        self.camera_status.config(text="Starting camera…", fg=MUTED)
        self.video_label.config(image="", text="Starting camera…")
        self.camera_job = self.window.after(30, self.update_camera)

    def stop_camera(self, message=None):
        message = message or CAMERA_OFF_TEXT
        if self.camera_job is not None:
            self.window.after_cancel(self.camera_job)
            self.camera_job = None
        self.camera.stop()
        self.stable_gesture = None
        self.hold.reset()
        self.video_image = None
        self.camera_button.config(text="📷  Start Camera")
        style_button(self.camera_button, SUCCESS)
        self.video_label.config(image="", text=message)
        if not message.startswith("⚠️"):
            self.camera_status.config(text="")
        self.set_gesture_display(None, None, 0.0, camera_on=False)

    def show_camera_error(self, message):
        self.stop_camera(message="⚠️ " + message)
        self.camera_status.config(text="Camera unavailable", fg=DANGER)

    def update_camera(self):
        """Runs on the Tk main thread every ~30 ms while the camera is on."""
        self.camera_job = None
        if not self.camera.running:
            self.show_camera_error(self.camera.error or "The camera stopped unexpectedly.")
            return

        data = self.camera.latest()
        if data is not None:
            self.draw_frame(data["frame"])
            self.stable_gesture = data["stable"]
            self.set_gesture_display(data["stable"], data["gesture"] if data["hand"] else None,
                                     data["confidence"], camera_on=True, hand=data["hand"])
            self.camera_status.config(text=f"Live • {data['fps']:.0f} FPS", fg=SUCCESS)

            now = time.monotonic()
            if self.auto_add.get():
                to_add = self.hold.update(self.stable_gesture, now)
                if to_add:
                    self.append_word(gestures.GESTURE_WORDS[to_add])
                self.set_bar(self.hold_bar, self.hold_fill, self.hold.progress(now))
            else:
                self.set_bar(self.hold_bar, self.hold_fill, 0.0)

        self.camera_job = self.window.after(30, self.update_camera)

    def draw_frame(self, rgb):
        from PIL import Image, ImageTk
        width = max(self.video_label.winfo_width() - 4, 160)
        height = max(self.video_label.winfo_height() - 4, 120)
        image = Image.fromarray(rgb)
        scale = min(width / image.width, height / image.height)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        self.video_image = ImageTk.PhotoImage(image)  # keep a reference or Tk drops it
        self.video_label.config(image=self.video_image, text="")

    def set_bar(self, canvas, item, fraction):
        canvas.coords(item, 0, 0, canvas.winfo_width() * fraction, int(canvas["height"]))

    def set_gesture_display(self, stable, raw, confidence, camera_on, hand=False):
        if stable:
            self.gesture_label.config(text=gestures.GESTURE_WORDS[stable], fg=SUCCESS)
            self.gesture_detail.config(text=f"{stable.replace('_', ' ').title()} • confidence {confidence:.0%}")
            style_button(self.add_button, enabled=True)
        else:
            style_button(self.add_button, enabled=False)
            if not camera_on:
                self.gesture_label.config(text="—", fg=MUTED)
                self.gesture_detail.config(text="Start the camera to recognise gestures.")
            elif not hand:
                self.gesture_label.config(text="No hand", fg=MUTED)
                self.gesture_detail.config(text="Show one hand, palm facing the camera, in good light.")
            else:
                self.gesture_label.config(text="Uncertain…", fg=WARNING)
                guess = "" if raw in (None, gestures.UNKNOWN) else f"Looks like {gestures.GESTURE_WORDS[raw]} • "
                self.gesture_detail.config(text=f"{guess}hold the gesture steady")
        self.set_bar(self.conf_bar, self.conf_fill, confidence if hand else 0.0)
        self.conf_bar.itemconfig(self.conf_fill, fill=SUCCESS if stable else ACCENT)
        for name, (row, word_lbl, hint_lbl) in self.guide_rows.items():
            bg = CARD_ALT if name == stable else CARD
            row.config(bg=bg)
            word_lbl.config(bg=bg, fg=SUCCESS if name == stable else TEXT)
            hint_lbl.config(bg=bg)

    def add_current_sign(self):
        if self.stable_gesture:
            self.append_word(gestures.GESTURE_WORDS[self.stable_gesture])
            self.hold.mark_used()

    def append_word(self, word):
        self.sentence.append(word)
        self.refresh_sentence()

    def backspace_word(self):
        if self.sentence:
            self.sentence.pop()
            self.refresh_sentence()

    def clear_sentence(self):
        self.sentence.clear()
        self.refresh_sentence()

    def refresh_sentence(self):
        self.sentence_label.config(text=" ".join(self.sentence))

    def speak_sentence(self):
        self.say(" ".join(self.sentence), self.sign_tts_status)

    def on_space_press(self, event):
        # Ignore key auto-repeat so holding Space adds the sign only once.
        if self.current_page != "sign" or self.space_down:
            return
        self.space_down = True
        self.add_current_sign()
        return "break"

    def on_backspace_key(self, event):
        if self.current_page == "sign" and not isinstance(event.widget, (tk.Entry, tk.Text)):
            self.backspace_word()

    def on_close(self):
        self.listen_stop.set()
        self.stop_camera()
        self.window.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    VisionSpeakApp(window)
    window.mainloop()
