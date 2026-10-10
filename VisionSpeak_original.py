import tkinter as tk
from tkinter import PhotoImage
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak():
    text = entry.get()
    engine.say(text)
    engine.runAndWait()

def listen_and_convert():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        status_label.config(text="🎤 Listening...")
        window.update()
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            status_label.config(text="⚠️ No speech detected.")
            return

        status_label.config(text="⏳ Processing...")
        window.update()

        try:
            text = recognizer.recognize_google(audio)
            previous_text = result_label.cget("text")
            new_text = previous_text + "\n" + text if previous_text else text
            result_label.config(text=new_text)
        except sr.UnknownValueError:
            result_label.config(text="⚠️ Could not understand audio.")
        except sr.RequestError:
            result_label.config(text="⚠️ Check your internet connection.")
        finally:
            status_label.config(text="")
            window.update()

def clear_text():
    result_label.config(text="")

def show_speech_to_text():
    clear_frame()

    title_label = tk.Label(main_frame, text="🎙️Speech to Text", font=("Segoe UI", 20), bg="#000000", fg="white")
    title_label.pack(pady=20)

    listen_button = tk.Button(main_frame, text="Start Listening", font=("Segoe UI", 14), bg="#4CAF50", fg="white", command=listen_and_convert)
    listen_button.pack(pady=10)

    clear_button = tk.Button(main_frame, text="Clear Text", font=("Segoe UI", 12), command=clear_text, bg="#f44336", fg="white")
    clear_button.pack(pady=5)

    global status_label, result_label
    status_label = tk.Label(main_frame, text="", font=("Segoe UI", 12), bg="#000000", fg="white")
    status_label.pack()

    result_label = tk.Label(main_frame, text="", wraplength=700, font=("Segoe UI", 12), justify="left", bg="#000000", fg="white")
    result_label.pack(pady=20)

def show_text_to_speech():
    clear_frame()

    title_label = tk.Label(main_frame, text=" 🗣️Text to Speech", font=("Segoe UI", 20), bg="#000000", fg="white")
    title_label.pack(pady=20)

    label = tk.Label(main_frame, text="Enter text below:", font=("Segoe UI", 14), bg="#000000", fg="white")
    label.pack()

    global entry
    entry = tk.Entry(main_frame, width=50, font=("Segoe UI", 14))
    entry.pack()

    speak_button = tk.Button(main_frame, text="Convert to Speech", command=speak, font=("Segoe UI", 14), bg="#2196F3", fg="white")
    speak_button.pack(pady=10)

def clear_frame():
    for widget in main_frame.winfo_children():
        widget.destroy()

# Main window setup
window = tk.Tk()
window.title("VisionSpeak")
window.geometry("1525x1000")
window.resizable(False, False)


bg_image = PhotoImage(file="")
background_label = tk.Label(window, image=bg_image)
background_label.place(relwidth=1, relheight=1)

# Top navigation frame
top_frame = tk.Frame(window, bg="black", height=60)
top_frame.pack(fill="x")

app_title = tk.Label(top_frame, text="🧠VisionSpeak", font=("Segoe UI", 20, "bold"), bg="black", fg="white")
app_title.pack(side="top", padx=20)

features_btn = tk.Menubutton(top_frame, text="Features ⏷", font=("Segoe UI", 14), bg="#1c1c1c", fg="white", relief="flat")
features_btn.config(activebackground="#333", activeforeground="white")
features_btn.pack(side="left", padx=10, pady=10)

features_menu = tk.Menu(features_btn, tearoff=0, font=("Segoe UI", 12), bg="white", fg="black")
features_btn.config(menu=features_menu)

features_menu.add_command(label="Speech to Text", command=show_speech_to_text)
features_menu.add_command(label="Text to Speech", command=show_text_to_speech)

# Main content frame (transparent over background)
main_frame = tk.Frame(window, bg="#000000",bd=0)
main_frame.pack(fill="both", expand=True)

# Load default screen
show_speech_to_text()

window.mainloop()
