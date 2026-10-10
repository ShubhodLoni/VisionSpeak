# 🧠 VisionSpeak

An assistive communication desktop app (Python + Tkinter) for people with speech or hearing difficulties.

| Feature | What it does | Internet needed? |
|---|---|---|
| 🎤 **Speech to Text** | Listens to the microphone and shows the spoken words | **Yes**: uses Google's free web speech recogniser |
| 🔊 **Text to Speech** | Reads typed text aloud | No (uses the voices built into Windows) |
| ✋ **Sign to Text** | Recognises a small set of hand gestures from the webcam and builds a sentence | No: runs fully offline |

> **Important:** Sign to Text is a **prototype gesture recogniser, not a sign-language translator.**
> It recognises 9 static, one-hand shapes. Real sign languages (ASL, ISL, BSL…) also depend on
> movement, both hands, facial expressions, body position and grammar, and none of that is modelled here.

---

## Project structure

```
VisionSpeak/
├── VisionSpeak.py          # The app: Tkinter UI (sidebar + 3 pages)
├── gestures.py             # Gesture rules, smoothing, hold-to-add (pure Python, unit-tested)
├── sign_camera.py          # Webcam + MediaPipe hand tracking on a background thread
├── models/
│   └── hand_landmarker.task  # MediaPipe hand-landmark model (Apache 2.0, ~7.5 MB)
├── tests/
│   ├── test_gestures.py    # Unit tests for the gesture logic
│   └── fixtures/sample_hand_landmarks.json  # real MediaPipe landmarks from 5 sample-photo hands
├── VisionSpeak_original.py # The original single-file UI, kept for reference
└── requirements.txt
```

## Setup (Windows)

1. Install **Python 3.10 or newer** (tested with 3.13) from python.org.
2. Open PowerShell in the `VisionSpeak` folder and activate the project's virtual environment
   (the existing `venv` folder already has everything installed; on a new PC create it with `python -m venv venv`):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If `PyAudio` fails to install, run `pip install --upgrade pip` and try again (recent versions ship ready-made Windows wheels).
4. The hand model is already in `models/hand_landmarker.task`. If it's ever missing, download it once from
   <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>
   and save it into the `models` folder.

## Run

```powershell
python VisionSpeak.py
```

## Using Speech to Text

1. Pick your language/accent from the drop-down (default **English (India)**). Choosing the one that matches how you speak noticeably improves accuracy.
2. Click **🎤 Start Listening**. It takes about half a second to measure the background noise, then shows **Listening… speak now**.
3. Speak normally. Each sentence is added to the transcript as soon as you pause, and the microphone keeps listening while it is transcribed.
4. Click **⏹ Stop Listening** when you are done.

## Using Sign to Text

1. Click **✋ Sign to Text** in the sidebar, then **📷 Start Camera**.
2. Hold **one hand** up, palm facing the camera, in good light, about 40–80 cm away.
3. When the "Detected gesture" panel shows the word in **green**, the gesture is confirmed.
   - **Uncertain…** (orange) means the hand is visible but the shape is unclear or still settling. Nothing is added.
   - **No hand** means no hand is in view.
4. Add the word to the sentence by either:
   - clicking **➕ Add Sign** or pressing **Space** (holding Space down adds the sign only once), or
   - ticking **Auto-add when held for 1.5 s**. Hold the gesture still until the green bar fills. It's added once; to add the same sign again, change gesture or drop your hand first.
5. Use **⌫ Backspace** (or the Backspace key), **Clear**, and **🔊 Speak** to read the sentence aloud.
6. Click **⏹ Stop Camera** to turn the webcam off. The camera is also released automatically when you switch to another page or close the window.

### Supported gestures

| Gesture | Hand shape | Adds |
|---|---|---|
| Open palm | All five fingers spread open | Hello |
| Fist | All fingers curled, thumb tucked over them | Stop |
| Thumbs up | Fist held sideways, thumb pointing **up** | Yes |
| Thumbs down | Fist held sideways, thumb pointing **down** | No |
| Point | Only the index finger extended | You |
| Victory | Index + middle finger extended (V) | Peace |
| OK | Thumb tip touching index tip, other three fingers up | OK |
| Call me | Thumb + little finger extended | Call me |
| I love you | Thumb + index + little finger extended (the ASL "ILY" handshape) | I love you |
| L shape | Thumb + index finger extended, like the letter L | Help |
| Three | Index + middle + ring finger extended (like the ASL letter W) | Water |
| Four | Four fingers up, thumb tucked into the palm | Thank you |
| Pinky | Only the little finger extended (like the ASL letter I) | I |
| Rock | Index + little finger extended, thumb tucked | Awesome |

Apart from "I love you", these word mappings are VisionSpeak's own choices, not official signs.
To change the words, edit `GESTURE_WORDS` in `gestures.py`.

### Verifying the webcam and each gesture

1. Start the camera: you should see a mirrored live image and "Live • ~30 FPS" next to the button.
   If not, see *Troubleshooting* below.
2. Raise your hand: purple lines and blue/white dots should follow your hand.
3. Make each gesture from the table and hold it for about half a second. Check that:
   - the correct word turns green in the "Detected gesture" panel;
   - the matching row in "Supported gestures" is highlighted;
   - pressing Space adds the word exactly once.
4. Make an unsupported shape (for example three fingers up). It should show **Uncertain…**, never a wrong word.

## How the recogniser works

1. **Hand landmarks:** MediaPipe's `HandLandmarker` (Tasks API, VIDEO mode) finds 21 points on one hand in each frame.
2. **Normalisation:** points are moved so the wrist is the origin and scaled so the palm length is 1, which makes the rules independent of hand size and distance. Every feature is a distance ratio, so the rules work for left and right hands and for slightly tilted hands.
3. **Finger features:**
   - a finger counts as *extended* when its tip is clearly farther from the wrist than its middle joint;
   - the thumb counts as *extended* when its tip is farther from the little-finger knuckle than its own base is;
   - thumbs up/down also check which way the thumb points;
   - OK checks that the thumb and index tips touch.
   Each check gives a 0–1 score rather than a hard yes/no.
4. **Confidence:** a gesture's confidence is its *weakest* finger match. If the best gesture scores below 0.6, or is within 0.15 of the runner-up, the result is **UNKNOWN**.
5. **Temporal smoothing:** a gesture only becomes *confirmed* when it wins at least 7 of the last 10 frames (about ⅓ second at 30 FPS), which filters out flicker.
6. **Deliberate adding:** nothing is added automatically every frame. You press Add/Space, or opt in to hold-to-add, which fires once per hold.

All the thresholds are named constants at the top of `gestures.py`.

## Tests

```powershell
python -m unittest discover tests -v
```

The tests use synthetic hand landmarks to check:
- every gesture rule;
- that size, position, left/right hand and tilt don't change the result;
- that unsupported or half-bent shapes come back as UNKNOWN;
- the smoothing and hold-to-add logic.

- real landmarks MediaPipe produced for its public sample photos (thumbs up, thumbs down, victory,
  pointing, and an unsupported thumb-tucked hand that must come back UNKNOWN), plus their mirror images.

They don't need a webcam. They **do not** measure real-world accuracy. Five hands is a sanity check, not an evaluation, and no accuracy figures have been measured on a real dataset.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Could not open the webcam" | Close other apps using the camera (Zoom, Teams, Camera app). Check *Settings → Privacy & security → Camera → Let desktop apps access your camera*. |
| "Missing Python packages" | `pip install -r requirements.txt` (inside your virtual environment, if you use one). |
| "Hand-landmark model not found" | Download the model (see Setup step 4) into `models/`. |
| Gestures flicker or stay "Uncertain" | Improve the lighting, keep your palm facing the camera, keep the whole hand in view, and hold still. |
| Thumbs up/down not recognised | Hold the fist **sideways** with the thumb clearly vertical. |
| Speech to Text says "Check your internet connection" | Speech to Text currently uses Google's online recogniser. |

## Limitations

- Only 9 **static** one-hand shapes. No movement-based signs, two-handed signs, facial expressions or grammar.
- One hand at a time.
- Rule-based: unusual hand angles (e.g. fingers pointing straight at the camera), poor lighting or partly hidden hands can produce *Uncertain*.
- Not trained or evaluated on a sign-language dataset, so no accuracy claims are made.
- Speech to Text needs internet access (Google Web Speech API).

## Future improvements

- Record your own landmark samples and train a small classifier (e.g. k-NN or a tiny neural network on the normalised landmarks) to support more gestures and fingerspelling.
- Add dynamic (movement-based) signs using sequences of landmarks over time.
- Use both hands (`num_hands=2`) for two-handed signs.
- Offline Speech to Text with Vosk (`vosk` + a downloaded model), so the whole app works without internet.
- Let users customise gesture words from the UI and save them to a settings file.
