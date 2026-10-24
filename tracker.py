"""ওয়েবক্যামের ভিডিও থেকে হাত শনাক্ত করে পাঁচটি আঙুল খোলা না বন্ধ, তা গণনা করার এক-ফাইলের Python প্রোগ্রাম।

PyCharm-এ চালানোর দ্রুত নির্দেশনা:
  ১. Python 3.11 বা 3.12 (64-bit) দিয়ে প্রজেক্ট তৈরি করুন।
  ২. প্রজেক্টের Terminal-এ নিচের প্যাকেজগুলো ইনস্টল করুন:
       python -m pip install mediapipe opencv-contrib-python numpy
  ৩. এই ফাইল খুলে Run করুন।

প্রথমবার চালালে Google-এর প্রায় ৮ MB হাত-শনাক্তকরণ মডেলটি কম্পিউটারের cache-এ নামবে।
মডেলটি ক্যাশে হয়ে গেলে পরবর্তীবার হাত শনাক্তকরণ ইন্টারনেট ছাড়াই স্থানীয়ভাবে চলবে।
এই প্রোগ্রাম ওয়েবক্যামের ফ্রেম ইন্টারনেটে পাঠায় বা ভিডিও রেকর্ড করে না।

কীবোর্ড নিয়ন্ত্রণ (আগে ক্যামেরা উইন্ডোতে ক্লিক করুন):
  Q / Esc: বন্ধ করুন; M: mirror view চালু/বন্ধ; R: smoothing-এর ইতিহাস রিসেট করুন।

বাহ্যিক ওয়েবক্যাম ব্যবহার করলে নিচের CAMERA_INDEX মান ০ থেকে ১ করুন।
পুরো হাত ক্যামেরায় রাখুন; পর্যাপ্ত আলো দিন এবং আঙুলগুলোকে একে অন্যের আড়াল থেকে সরিয়ে রাখুন।
আঙুলের খোলা/বন্ধ অবস্থা জ্যামিতিক নিয়ম দিয়ে অনুমান করা হয়; বাঁকা বা আড়াল হওয়া আঙুল ভুল শনাক্ত হতে পারে।
নিজের হাত ও ক্যামেরার জন্য নিচের angle/distance threshold-গুলো পরীক্ষা করে সমন্বয় করা যায়।

MediaPipe Hand Landmarker-এর API নির্দেশিকা:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
"""

# এই future import type annotation-গুলোকে runtime-এ তৎক্ষণাৎ evaluate না করে forward reference হিসেবে রাখতে দেয়।
from __future__ import annotations

# গণিতের ফাংশন—শেষে কোণের radian-কে degree-তে রূপান্তর করতে লাগবে।
import math
# OS environment variable ও ফাইল/ফোল্ডার-সংক্রান্ত সুবিধা নিতে os মডিউল লাগে।
import os
# Python runtime-এর তথ্য, বিশেষ করে operating system-এর ধরন, জানতে sys লাগে।
import sys
# সময় মাপতে perf_counter ব্যবহার হবে; এটি elapsed time/FPS হিসাবের জন্য উপযোগী monotonic timer।
import time
# URL অনুরোধের সময় network-সংক্রান্ত ত্রুটির নির্দিষ্ট ধরন ধরতে এই module লাগে।
import urllib.error
# HTTP/HTTPS URL থেকে মডেল ডাউনলোডের request ও response পরিচালনা করে।
import urllib.request
# ডাউনলোড করা মডেল ফাইলের ভেতরের bundle/ZIP কাঠামো যাচাই করতে লাগে।
import zipfile
# সর্বশেষ কয়েকটি ফ্রেমের আঙুলের অবস্থা জমিয়ে রাখতে সীমাবদ্ধ queue তৈরি করতে deque ব্যবহার হবে।
from collections import deque
# Path object দিয়ে operating system অনুযায়ী নিরাপদভাবে ফাইল-পথ তৈরি ও পরিচালনা করা হয়।
from pathlib import Path

# তৃতীয় পক্ষের লাইব্রেরি import করা হচ্ছে; নিচে ব্যর্থ হলে ব্যবহারকারীকে কাজের নির্দেশসহ ত্রুটি দেখানো হবে।
try:
    # OpenCV ক্যামেরা পড়া, ছবি রূপান্তর করা, হাত আঁকা এবং window দেখানোর কাজ করবে।
    import cv2
    # MediaPipe-এর Image ও image-format API-সহ সাধারণ namespace mp নামে ব্যবহার করা হবে।
    import mediapipe as mp
    # NumPy landmark-এর 3D vector, distance, average ও image canvas নিয়ে দ্রুত গাণিতিক কাজ করে।
    import numpy as np
    # MediaPipe Tasks-এর Python configuration class-গুলো এখানে পাওয়া যায়।
    from mediapipe.tasks import python
    # MediaPipe Vision-এর HandLandmarker এবং তার option/configuration class ব্যবহার করা হবে।
    from mediapipe.tasks.python import vision
# প্যাকেজ অনুপস্থিত, ভুল Python environment, বা native library লোডের ত্রুটি হলে এই শাখায় আসবে।
except (ImportError, OSError) as exc:
    # কাঁচা traceback না দেখিয়ে নিয়ন্ত্রিত startup error হিসেবে প্রোগ্রাম বন্ধ করা শুরু হচ্ছে।
    raise SystemExit(
        # কোন dependency import করতে ব্যর্থ হয়েছে, তার আসল error বার্তা এখানে যুক্ত হয়।
        f"Could not load a required package: {exc}\n\n"
        # ব্যবহারকারী কোথায় dependency ইনস্টল করবেন, সেই নির্দেশনা দেওয়া হচ্ছে।
        "In your PyCharm project's Terminal, run:\n"
        # প্রয়োজনীয় তিনটি package একই project interpreter-এ ইনস্টল করার command দেখানো হয়।
        "  python -m pip install mediapipe opencv-contrib-python numpy\n\n"
        # ইনস্টল ও Run-এ একই interpreter ব্যবহার না করলে package পাওয়া যাবে না—সে ঝুঁকি বোঝানো হয়।
        "Use the same Python interpreter for installation and Run.\n"
        # ফাইলের নাম cv2.py বা mediapipe.py হলে নিজের ফাইল আসল package-কে ঢেকে দিতে পারে, তাই সঠিক নাম বলা হয়েছে।
        "Name this file hand_finger_tracker.py, not cv2.py or mediapipe.py."
    # SystemExit তৈরি হওয়ার কারণ হিসেবে মূল import exception-টিও সংযুক্ত থাকে।
    ) from exc


# সহজে পরিবর্তনযোগ্য সেটিংস: ক্যামেরা, smoothing ও আঙুলের geometry threshold এখানে রাখা হয়েছে।
# কোন ক্যামেরা device খুলতে হবে—০ সাধারণত laptop-এর built-in webcam; বাহ্যিক ক্যামেরা হলে বদলান।
CAMERA_INDEX = 0  # ০ সাধারণত laptop-এর built-in webcam; বাহ্যিক camera হলে index বদলান।
# True থাকলে প্রতিটি frame অনুভূমিকভাবে উল্টে selfie/mirror view দেখানো হবে।
MIRROR = True
# ক্যামেরাকে ৯৬০ pixel প্রস্থের frame দিতে অনুরোধ করা হয়; camera driver এই মান উপেক্ষাও করতে পারে।
CAMERA_WIDTH = 960
# ক্যামেরার পছন্দসই frame height ৫৪০ pixel নির্ধারণ করে।
CAMERA_HEIGHT = 540
# একসঙ্গে সর্বোচ্চ কতটি হাত শনাক্ত করতে হবে তা MediaPipe-কে জানানো হয়।
MAX_HANDS = 2
# স্মুথিংয়ের জন্য প্রতি হাতে সর্বশেষ পাঁচটি frame-এর সিদ্ধান্ত জমা থাকবে; বেশি মান ঝাঁকুনি কমায় কিন্তু প্রতিক্রিয়া ধীর করে।
SMOOTHING_FRAMES = 5  # সর্বশেষ পাঁচটি frame-এর majority-vote state flicker কমায়।

# কোণের একক degree; সোজা joint-এর কোণ সাধারণত ১৮০ ডিগ্রির কাছাকাছি।
# হাতের স্বাভাবিক বাঁক ও 3D depth estimation-এর noise মেনে নেওয়ার জন্য threshold ১৮০-এর নিচে রাখা হয়।
# Index থেকে pinky পর্যন্ত আঙুলের PIP joint-এর কোণ এর বেশি হলে joint-টিকে যথেষ্ট সোজা ধরা হবে।
FINGER_PIP_ANGLE = 140
# আঙুলের ডগার কাছের DIP joint কতটা সোজা, তার জন্য দ্বিতীয় কোণ-সীমা।
FINGER_DIP_ANGLE = 120
# বুড়ো আঙুলের প্রথম গুরুত্বপূর্ণ joint-এর সোজাভাব যাচাইয়ের কোণ-সীমা।
THUMB_MCP_ANGLE = 140
# বুড়ো আঙুলের পরের IP joint সোজা কি না যাচাইয়ের কোণ-সীমা।
THUMB_IP_ANGLE = 150
# বুড়ো আঙুলের ডগা ও তর্জনীর গোড়ার দূরত্বকে palm width দিয়ে ভাগ করা হয়; অনুপাত সীমার বেশি হলে ছড়ানো ধরা হয়।
THUMB_MIN_SPREAD = 0.55  # thumb tip থেকে index base-এর দূরত্ব / palm width অনুপাত।

# MediaPipe-এর pre-trained Hand Landmarker মডেলের download URL-টি একাধিক string অংশে সাজানো হচ্ছে।
MODEL_URL = (
    # URL-এর প্রথম অংশ; পরের string-এর সঙ্গে Python স্বয়ংক্রিয়ভাবে জুড়ে দেবে, তাই + লাগে না।
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    # একই URL-এর শেষ অংশে নির্দিষ্ট model version ও ফাইলের নাম আছে।
    "hand_landmarker/float16/1/hand_landmarker.task"
# বন্ধনী ও একাধিক string জোড়ার পর সম্পূর্ণ URL-কে MODEL_URL-এ রাখা হয়।
)
# OpenCV window-এর title bar-এ যে নাম দেখাবে, সেটি নির্ধারণ করা হয়।
WINDOW_NAME = "Hand & Finger Tracker"
# UI-তে আঙুলের নাম ও states list-এর index মিলিয়ে রাখার নির্দিষ্ট ক্রম।
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
# প্রতিটি আঙুলের landmark index-এর chain: গোড়া থেকে ডগা পর্যন্ত চারটি গুরুত্বপূর্ণ point।
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                 # আগের লাইনের tuple-গুলো সম্পূর্ণ করে; প্রতিটি chain-এর শেষ index আঙুলের tip।
                 (13, 14, 15, 16), (17, 18, 19, 20))
# হাত আঁকার সময় কোন landmark-কে কোন landmark-এর সঙ্গে রেখা দিয়ে যুক্ত করতে হবে, তার তালিকা শুরু।
CONNECTIONS = (
    # কবজি থেকে বুড়ো আঙুলের ডগা পর্যন্ত সংযোগ এবং index finger-এর প্রথম অংশের সংযোগ।
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
    # Index finger-এর বাকি জোড় এবং index base থেকে middle finger-এর দিকে palm সংযোগ।
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    # Middle finger-এর অংশ এবং ring finger-এর দিকে palm-এর সংযোগ যোগ করা হয়।
    (10, 11), (11, 12), (9, 13), (13, 14),
    # Ring finger-এর joint-গুলো এবং palm-এর ring-to-pinky সংযোগ নির্ধারণ করে।
    (14, 15), (15, 16), (13, 17), (0, 17),
    # Pinky finger-এর বাকি joint-গুলো সম্পূর্ণ করে।
    (17, 18), (18, 19), (19, 20),
# CONNECTIONS tuple শেষ; এই তালিকা কেবল আঁকার skeleton-কে নিয়ন্ত্রণ করে, শনাক্তকরণকে নয়।
)
# OpenCV-তে রঙের ক্রম BGR; RGB দিলে লাল ও নীল চ্যানেল অদলবদল হয়ে যেতে পারে।
# খোলা আঙুল বোঝাতে ব্যবহৃত BGR রঙ; OpenCV-এর রঙের ক্রম RGB থেকে উল্টো।
GREEN = (110, 224, 106)
# বন্ধ আঙুলের জন্য কমলা/কমলা-লাল রঙের BGR মান নির্ধারণ করা হয়।
ORANGE = (80, 154, 255)
# সাদা রঙে landmark dot ও স্পষ্ট লেখা আঁকা হয়।
WHITE = (240, 243, 248)
# কম গুরুত্বের সহায়ক লেখা ধূসর-নীল রঙে দেখানো হয়।
MUTED = (160, 172, 187)
# পাশের তথ্য-প্যানেলের পটভূমির গাঢ় BGR রঙ।
BACKGROUND = (23, 28, 37)


# মডেল ফাইল স্থানীয় cache-এ নিশ্চিত করার ফাংশন; ফেরত মান একটি pathlib Path।
def ensure_model() -> Path:
    """মডেল আগে থেকেই বৈধভাবে ক্যাশে থাকলে সেটি ফেরত দেয়; না থাকলে ডাউনলোড, যাচাই ও সংরক্ষণ করে।"""
    # Windows-এ LOCALAPPDATA থাকলে প্রথমে সেটি বেছে নেওয়া হয়; না থাকলে পরের fallback দেখা হবে।
    root = Path(os.environ.get("LOCALAPPDATA") or
                # LOCALAPPDATA না থাকলে XDG_CACHE_HOME; সেটিও না থাকলে home directory-এর .cache ব্যবহার করা হয়।
                os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    # মডেল রাখার নির্দিষ্ট cache path তৈরি হয়; ফাইলটি hand_landmarker.task নামে থাকবে।
    path = root / "HandFingerTracker" / "hand_landmarker.task"

    # ভেতরের helper function candidate ফাইলটি প্রকৃত ও সম্পূর্ণ মডেল bundle কি না যাচাই করে।
    def valid(candidate: Path) -> bool:
        # ফাইল পড়া বা ZIP যাচাইয়ে সমস্যা হলে সেটি বৈধ নয় বলে ফেরত দেওয়ার জন্য try শুরু।
        try:
            # candidate ফাইলকে ZIP archive হিসেবে খুলে bundle নামে variable-এ ধরা হয়।
            with zipfile.ZipFile(candidate) as bundle:
                # প্রয়োজনীয় দুটি TensorFlow Lite model component archive-এ আছে কি না যাচাই শুরু।
                return {"hand_detector.tflite", "hand_landmarks_detector.tflite"}.issubset(
                    # archive-এর সব entry-এর নাম নিয়ে সেটের সঙ্গে তুলনা করা হয়; প্রয়োজনীয় দুই ফাইল থাকতেই হবে।
                    bundle.namelist()
                # নামের পাশাপাশি testzip() দিয়ে ZIP-এর ভেতরের corruption আছে কি না যাচাই হয়; None মানে ত্রুটি পাওয়া যায়নি।
                ) and bundle.testzip() is None
        # ফাইল না পাওয়া/পড়া না যাওয়া বা ZIP নষ্ট হলে এই exception ধরা হবে।
        except (OSError, zipfile.BadZipFile):
            # যাচাইয়ে সমস্যা হলে helper স্পষ্টভাবে False জানায়, যাতে download আবার করা যায়।
            return False

    # cache-এ থাকা ফাইলটি বৈধ হলে নতুন করে নেটওয়ার্ক ব্যবহার না করে সেটি পুনর্ব্যবহার করা হয়।
    if valid(path):
        # যাচাইকৃত local model path ফেরত দিয়ে download ধাপ এড়িয়ে যাওয়া হয়।
        return path
    # মডেল রাখার parent directory না থাকলে সেটিসহ প্রয়োজনীয় parent folder তৈরি হয়।
    path.parent.mkdir(parents=True, exist_ok=True)
    # অসম্পূর্ণ download সরাসরি আসল ফাইলের ওপর না লিখে .download temporary extension-এ রাখা হয়।
    temporary = path.with_suffix(".download")
    # প্রথমবার মডেল নামানোর সময় ব্যবহারকারীকে terminal-এ অগ্রগতির বার্তা দেওয়া হয়।
    print("First run: downloading Google's hand model. Please wait...")
    # নেটওয়ার্ক request তৈরির সময় ত্রুটি ধরার জন্য try block শুরু।
    try:
        # মডেল URL-এ User-Agent header যোগ করে HTTP request বানানো হয়।
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "HandFingerTracker/1.0"})
        # সর্বোচ্চ ৪৫ সেকেন্ড অপেক্ষা করে response খোলা এবং অস্থায়ী ফাইল binary-write মোডে খোলা হয়।
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as output:
            # response থেকে ২৫৬ KiB করে chunk পড়তে থাকে; ফাঁকা chunk এলে EOF, অর্থাৎ download শেষ।
            while chunk := response.read(256 * 1024):
                # যে chunk পাওয়া যায় তা অস্থায়ী ফাইলে লেখা হয়, ফলে পুরো মডেল একবারে RAM-এ জমাতে হয় না।
                output.write(chunk)
        # ডাউনলোড শেষ হলে একই bundle validator দিয়ে temporary ফাইলটি যাচাই করা হয়।
        if not valid(temporary):
            # যাচাই ব্যর্থ হলে ভুল বা অসম্পূর্ণ মডেল ব্যবহার আটকাতে RuntimeError ছোড়া হয়।
            raise RuntimeError("The model download is incomplete or invalid.")
        # যাচাইকৃত temporary ফাইলকে আসল cache path-এ replace করা হয়; একই filesystem-এ এটি নিরাপদ commit-এর মতো কাজ করে।
        temporary.replace(path)
    # ডাউনলোড, ফাইল বা validation-সংক্রান্ত প্রত্যাশিত ত্রুটি এলে ব্যবহারকারী-বান্ধব বার্তা বানানো হয়।
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        # নতুন RuntimeError তৈরি করে আসল সমস্যাটি নিচের কয়েকটি string-এ ব্যাখ্যা করা হবে।
        raise RuntimeError(
            # কোন কারণে model download ব্যর্থ হয়েছে, সেই মূল exception বার্তায় বসানো হয়।
            f"Could not download the hand model: {exc}\n"
            # ইন্টারনেট পরীক্ষা বা manual download-এর বিকল্প পথ ব্যবহারকারীকে জানানো হয়।
            "Connect to the internet and run again. You can also download:\n"
            # সঠিক URL এবং local destination path দেখানো হয়, যেন চাইলে মডেল নিজে নামিয়ে রাখতে পারেন।
            f"{MODEL_URL}\nThen save it as:\n{path}"
        # নতুন ত্রুটির সঙ্গে আসল exception-এর causal সম্পর্ক সংরক্ষণ করা হয়।
        ) from exc
    # download সফল হোক বা ব্যর্থ হোক, অস্থায়ী ফাইল পরিষ্কার করার finally block শুরু।
    finally:
        # অস্থায়ী ফাইল মুছে দেওয়া হয়; ফাইল ইতিমধ্যে না থাকলেও missing_ok=True-এর কারণে cleanup ভাঙবে না।
        temporary.unlink(missing_ok=True)
    # সফলভাবে cache-এ মডেল তৈরি হলে তার path terminal-এ দেখানো হয়।
    print(f"Model ready: {path}")
    # পরবর্তী ধাপে MediaPipe-কে দেওয়ার জন্য যাচাইকৃত মডেল ফাইলের Path ফেরত দেওয়া হয়।
    return path


# তিনটি NumPy vector a, b, c থেকে b-কে vertex ধরে joint angle গণনার typed function।
def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """তিনটি 3D landmark দিয়ে মাঝের বিন্দুতে (B) গঠিত ABC কোণ ডিগ্রিতে হিসাব করে।"""
    # vertex b থেকে a এবং b থেকে c-এর দিকে দুটি vector বানানো হয়; কোণ এই দুই vector-এর মধ্যবর্তী।
    first, second = a - b, c - b
    # দুটি vector-এর দৈর্ঘ্য গুণ করে dot-product সূত্রের denominator তৈরি করা হয়।
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    # দৈর্ঘ্য প্রায় শূন্য হলে ভাগ করে NaN/ভুল ফল পাওয়ার ঝুঁকি থাকে, তাই আগে পরীক্ষা।
    if denominator < 1e-10:
        # দুটি vector-এর যেকোনোটি প্রায় শূন্য হলে নিরাপদ fallback হিসেবে ০ ডিগ্রি ফেরত দেওয়া হয়।
        return 0.0
    # dot product / দৈর্ঘ্যের গুণফল হলো cos(theta); floating-point rounding সামলাতে মান -1 থেকে 1-এ clip করা হয়।
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    # arccos দিয়ে cosine-কে angle-এ ফিরিয়ে radian থেকে degree-তে রূপান্তর করে ফল ফেরত দেওয়া হয়।
    return math.degrees(math.acos(cosine))


# প্রতিটি আঙুলের landmark অবস্থান নিয়ে পাঁচটি Boolean state ফেরত দেওয়ার classifier শুরু।
def classify_fingers(points: np.ndarray) -> list[bool]:
    """Thumb, index, middle, ring ও pinky—এই ক্রমে প্রতিটি আঙুল সোজা/খোলা কি না অনুমান করে।

    MediaPipe-এর world landmarks হাতের 3D অবস্থান দেয়; তাই শুধু ছবির y-coordinate তুলনার বদলে
    joint angle ও হাতের আকারের তুলনায় দূরত্ব ব্যবহার করলে ঘোরানো হাত ও স্কেল বদল কিছুটা সামলানো যায়।
    এগুলো পরীক্ষাভিত্তিক জ্যামিতিক heuristic—সঠিক হওয়ার পরিসংখ্যানগত probability নয়।
    """
    # Landmark 5 ও 17—তর্জনী ও কনিষ্ঠার গোড়ার মধ্যবর্তী দূরত্বকে palm width-এর আনুমানিক মাপ ধরা হয়।
    palm_width = max(float(np.linalg.norm(points[5] - points[17])), 1e-8)
    # বুড়ো আঙুলের অবস্থা তিনটি শর্ত একসঙ্গে পূরণ করে কি না তা যাচাই শুরু।
    thumb_open = (
        # Landmark 1–2–3-এর কোণ সীমার বেশি হলে বুড়ো আঙুলের প্রথম অংশ যথেষ্ট সোজা বলে ধরা হয়।
        joint_angle(points[1], points[2], points[3]) > THUMB_MCP_ANGLE
        # Landmark 2–3–4-এর কোণ দিয়ে বুড়ো আঙুলের tip-সংলগ্ন joint-টিও যথেষ্ট সোজা কি না দেখা হয়।
        and joint_angle(points[2], points[3], points[4]) > THUMB_IP_ANGLE
        # Tip 4 ও index base 5-এর দূরত্ব palm width দিয়ে normalize করে বুড়ো আঙুল যথেষ্ট ছড়িয়েছে কি না যাচাই।
        and np.linalg.norm(points[4] - points[5]) / palm_width > THUMB_MIN_SPREAD
    # এই closing parenthesis thumb_open-এর বহু-শর্তের expression শেষ করে।
    )
    # প্রথম state হিসেবে thumb_open-কে Boolean-এ রূপান্তর করে states list-এ রাখা হয়।
    states = [bool(thumb_open)]
    # Thumb বাদ দিয়ে বাকি চারটি আঙুলের MCP, PIP, DIP ও tip landmark index ধরে loop চলে।
    for mcp, pip, dip, tip in FINGER_CHAINS[1:]:
        # MCP–PIP–DIP angle দিয়ে মূল মাঝের joint যথেষ্ট সোজা কি না হিসাব করা হয়।
        straight_pip = joint_angle(points[mcp], points[pip], points[dip]) > FINGER_PIP_ANGLE
        # PIP–DIP–tip angle দিয়ে ডগার কাছের joint-ও সোজা কি না পরীক্ষা করা হয়।
        straight_dip = joint_angle(points[pip], points[dip], points[tip]) > FINGER_DIP_ANGLE
        # আঙুল তালুর ওপর ভাঁজ হয়ে থাকলেও শেষের joint দুটি তুলনামূলক সোজা থাকতে পারে; তাই দূরত্বের শর্তও যোগ করা হয়েছে।
        # Tip কবজি থেকে PIP-এর চেয়ে বেশি দূরে গেলে আঙুলটি বাইরে প্রসারিত—এমন অতিরিক্ত শর্তের হিসাব শুরু।
        reaches_outward = np.linalg.norm(points[tip] - points[0]) > (
            # PIP থেকে কবজির দূরত্বের ১.০৩ গুণ ব্যবহার করে খুব সামান্য পার্থক্যকে noise ভেবে বাদ দেওয়া হয়।
            1.03 * np.linalg.norm(points[pip] - points[0])
        # বন্ধনী বন্ধ করে reaches_outward-এর তুলনামূলক distance expression সম্পূর্ণ করা হয়।
        )
        # joint-এর দুই কোণ ও outward distance—তিনটি শর্ত সত্য হলেই আঙুলকে খোলা ধরা হয়।
        states.append(bool(straight_pip and straight_dip and reaches_outward))
    # Thumb, index, middle, ring, pinky ক্রমের পাঁচটি Boolean ফল ফেরত দেওয়া হয়।
    return states


# StateSmoother class একই শারীরিক হাতকে একাধিক frame-এ চিনে রেখে classification-এর ঝাঁকুনি কমায়।
class StateSmoother:
    """কবজির অবস্থান মিলিয়ে ফ্রেমগুলোর মধ্যে হাতের পরিচয় ধরে রাখে এবং ফলের ঝাঁকুনি কমায়।"""

    # প্রতিটি নতুন smoother object-এর internal track dictionary ও ID counter শুরু করা হয়।
    def __init__(self) -> None:
        # প্রতিটি track ID-র বিপরীতে কবজির অবস্থান, সর্বশেষ দেখা সময়, history ও সিদ্ধান্তের তথ্য রাখা হবে।
        self.tracks: dict[int, dict] = {}
        # নতুন হাত পাওয়া গেলে দিতে হবে এমন পরবর্তী unique integer ID ১ থেকে শুরু।
        self.next_id = 1

    # বর্তমান frame-এর detections এবং বর্তমান সময় নিয়ে track মিলিয়ে স্মুথ states-সহ detections ফেরায়।
    def update(self, detections: list[dict], now: float) -> list[dict]:
        # পুরোনো track-গুলোর মধ্যে কেবল সময়সীমার ভেতরে থাকা track রেখে dictionary নতুন করে তৈরি হয়।
        self.tracks = {key: track for key, track in self.tracks.items()
                       # কোনো হাত ০.৩৫ সেকেন্ডের বেশি দেখা না গেলে সেটিকে হারিয়ে যাওয়া ধরে পুরোনো track বাদ দেওয়া হয়।
                       if now - track["last_seen"] < 0.35}
        # সর্বোচ্চ দুই হাতের জন্য সব সম্ভাব্য wrist pair distance অনুযায়ী সাজিয়ে greedy matching করা যথেষ্ট সহজ পদ্ধতি।
        # বর্তমান detection ও আগের track-এর সম্ভাব্য wrist-distance জোড়া সাজানোর জন্য sorted list তৈরি শুরু।
        pairs = sorted(
            # একটি detection-এর কবজি এবং একটি পুরোনো track-এর কবজির Euclidean distance বের করে tuple-এর প্রথম অংশ রাখা হয়।
            (float(np.linalg.norm(hand["wrist"] - track["wrist"])), index, key)
            # প্রতিটি detection-কে index দিয়ে চিহ্নিত করে সব detection-এর ওপর loop চলে।
            for index, hand in enumerate(detections)
            # প্রতিটি detection-এর সঙ্গে প্রতিটি বিদ্যমান track মিলিয়ে সম্ভাব্য জোড়া তৈরি হয়।
            for key, track in self.tracks.items()
        # sorted() call শেষ; সবচেয়ে কাছের wrist pair তালিকার শুরুতে আসে।
        )
        # matches detection-index থেকে track-ID ম্যাপ করে; used নিশ্চিত করে একটি পুরোনো track দুবার ব্যবহার না হয়।
        matches, used = {}, set()
        # distance অনুযায়ী সাজানো candidate pair-গুলো কাছ থেকে দূরে পরীক্ষা করা হয়।
        for distance, index, key in pairs:
            # দূরত্ব ০.২৫-এর কম এবং detection/track আগে ব্যবহৃত না হলে জোড়াটি গ্রহণযোগ্য।
            if distance < 0.25 and index not in matches and key not in used:
                # গ্রহণ করা detection-টির index-এর সঙ্গে সংশ্লিষ্ট পুরোনো track ID যুক্ত করা হয়।
                matches[index] = key
                # একই track আরেক detection-কে বরাদ্দ না দেওয়ার জন্য সেটিকে used-এ চিহ্নিত করা হয়।
                used.add(key)
        # বর্তমান frame-এর প্রতিটি detected hand-এর জন্য স্মুথিং ও ID update করা শুরু।
        for index, hand in enumerate(detections):
            # বর্তমান detection-এর জন্য matching ধাপে পাওয়া track ID আছে কি না নেওয়া হয়।
            key = matches.get(index)
            # যদি মিলে যাওয়া পুরোনো track না থাকে, নতুন track তৈরি করার শাখা।
            if key is None:
                # পরবর্তী unique ID-টি নতুন হাতের key হিসেবে বরাদ্দ হয়।
                key = self.next_id
                # পরের নতুন হাতের জন্য ID counter বাড়ানো হয়, যাতে একই ID পুনর্ব্যবহৃত না হয়।
                self.next_id += 1
                # এই track-এর জন্য সর্বশেষ SMOOTHING_FRAMES সংখ্যক states ধারণকারী সীমাবদ্ধ deque তৈরি হয়।
                self.tracks[key] = {"history": deque(maxlen=SMOOTHING_FRAMES)}
            # এখনকার detection-এর জন্য যে track বরাদ্দ হয়েছে, তার dictionary পাওয়া হয়।
            track = self.tracks[key]
            # কবজির সর্বশেষ অবস্থান ও দেখা হওয়ার সময় বর্তমান মান দিয়ে আপডেট করা হয়।
            track.update(wrist=hand["wrist"], last_seen=now)
            # এই frame-এর পাঁচ আঙুলের raw states history-তে যোগ হয়; deque পূর্ণ হলে সবচেয়ে পুরোনোটি বাদ পড়ে।
            track["history"].append(hand["states"])
            # history-র Boolean state-গুলোকে গড় করলে প্রতিটি আঙুল কত ভাগ frame-এ খোলা ছিল তার vote পাওয়া যায়।
            votes = np.mean(track["history"], axis=0)
            # আগের স্মুথ state সংরক্ষণ করা হয়, যাতে ঠিক ৫০/৫০ tie হলে সিদ্ধান্ত হঠাৎ না বদলায়।
            previous = track.get("states", hand["states"])
            # vote ঠিক ৫০/৫০ হলে এক frame থেকে অন্য frame-এ flicker আটকাতে আগের state-ই রাখা হয়।
            # প্রতিটি আঙুলের গড় vote ০.৫-এর বেশি হলে True; কম হলে False; ঠিক সমান হলে আগের সিদ্ধান্ত রাখা হয়।
            track["states"] = [bool(v > 0.5) if v != 0.5 else previous[i]
                               # enumerate-এর i দিয়ে votes-এর একই index-এর আগের state বেছে নেওয়া হয়; list comprehension সম্পূর্ণ।
                               for i, v in enumerate(votes)]
            # স্মুথিংয়ের ফল detection object-এ ফিরিয়ে লেখা হয়; পরের drawing/UI এই state-ই দেখাবে।
            hand["states"] = track["states"].copy()
            # এই হাতের track ID detection-এ যোগ করা হয়, যাতে UI-তে হাতের পরিচয় স্থিতিশীল থাকে।
            hand["id"] = key
        # সব detection এখন smoothed state ও ID-সহ ফেরত দেওয়া হয়।
        return detections


# ক্যামেরা চালু করে প্রথম সফল frame-সহ VideoCapture object ফেরত দেওয়ার ফাংশন।
def open_camera():
    """Windows-এ আগে DirectShow backend চেষ্টা করে; ব্যর্থ হলে সাধারণ OpenCV backend ব্যবহার করে।"""
    # Windows-এ DirectShow প্রথমে চেষ্টা হয়; অন্য OS-এ OpenCV-এর default backend একবারই ব্যবহার হয়।
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    # একটির পর একটি backend পরীক্ষা করার loop শুরু।
    for backend in backends:
        # CAMERA_INDEX দিয়ে পছন্দের ক্যামেরা খুলতে OpenCV capture object তৈরি হয়।
        camera = cv2.VideoCapture(CAMERA_INDEX, backend)
        # camera driver সত্যিই device খুলতে পেরেছে কি না আগে যাচাই করা হয়।
        if camera.isOpened():
            # ক্যামেরার কাঙ্ক্ষিত frame width সেট করতে অনুরোধ করা হয়।
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            # ক্যামেরার কাঙ্ক্ষিত frame height সেট করতে অনুরোধ করা হয়।
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            # প্রতি সেকেন্ডে প্রায় ৩০ frame পাওয়ার অনুরোধ করা হয়; নিশ্চিত guarantee নয়।
            camera.set(cv2.CAP_PROP_FPS, 30)
            # ক্যামেরা থেকে একটি পরীক্ষামূলক frame পড়ে প্রাথমিকভাবে ব্যবহারযোগ্য কি না যাচাই করা হয়।
            success, frame = camera.read()
            # read সফল হয়েছে এবং frame ফাঁকা নয়—দুটি শর্ত একসঙ্গে পরীক্ষা হয়।
            if success and frame is not None:
                # ক্যামেরা ব্যবহারযোগ্য হলে camera object ও ইতিমধ্যে পড়া প্রথম frame caller-কে দেওয়া হয়।
                return camera, frame
        # একটি backend ব্যর্থ হলে পরেরটি চেষ্টা করার আগে সংশ্লিষ্ট camera resource ছেড়ে দেওয়া হয়।
        camera.release()
    # সব backend ব্যর্থ হলে স্পষ্ট কারণসহ RuntimeError তোলা হয়।
    raise RuntimeError(
        # ব্যবহার করা camera index-সহ error message-এর প্রথম অংশ তৈরি হয়।
        f"Cannot read webcam {CAMERA_INDEX}.\n"
        # ক্যামেরা অন্য app ব্যবহার করছে কি না এবং Windows privacy permission চালু আছে কি না দেখার নির্দেশ।
        "Close other apps using your camera, enable camera access for desktop apps\n"
        # Windows-এর Settings-এ camera permission কোথায় পাওয়া যায় তা নির্দিষ্ট করে।
        "in Windows Settings > Privacy & security > Camera, then run again.\n"
        # বাহ্যিক webcam হলে অন্য index ১ পরীক্ষা করার ব্যবহারিক পরামর্শ দেয়।
        "For an external webcam, try CAMERA_INDEX = 1 at the top of this file."
    # RuntimeError message-এর বন্ধনী বন্ধ করে exception সম্পূর্ণ করা হয়।
    )


# OpenCV canvas-এ লেখা বসানোর জন্য ছোট wrapper; scale, color ও thickness-এ default মান আছে।
def write_text(canvas, text, position, scale=0.6, color=WHITE, thickness=1):
    # putText-এ font, অবস্থান ও অ্যান্টি-এলিয়াসিং পদ্ধতিসহ লেখা আঁকা শুরু।
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                # লেখার আকার, রঙ, রেখার পুরুত্ব ও LINE_AA দিয়ে edge মসৃণ করা নির্দিষ্ট হয়।
                scale, color, thickness, cv2.LINE_AA)


# MediaPipe landmark থেকে হাতের skeleton, joint dot এবং আঙুলের রঙের চিহ্ন frame-এ আঁকে।
def draw_hand(frame: np.ndarray, hand: dict) -> None:
    # frame-এর height ও width বের করা হয়, normalized অবস্থানকে pixel-এ রূপান্তর করতে লাগবে।
    height, width = frame.shape[:2]
    # landmark-এর normalized x,y মানকে image-এর width,height দিয়ে গুণ করে পূর্ণসংখ্যা pixel coordinate বানানো হয়।
    coordinates = [(int(mark.x * width), int(mark.y * height))
                   # সব landmark-এর জন্য coordinate list comprehension সম্পূর্ণ করে।
                   for mark in hand["landmarks"]]
    # হাতের skeleton-এর প্রতিটি সংযোগ-জোড়া ধরে loop চলে।
    for start, end in CONNECTIONS:
        # সংযোগ-তালিকার start ও end pixel-কে সবুজ রেখায় যুক্ত করে landmark skeleton আঁকা হয়।
        cv2.line(frame, coordinates[start], coordinates[end], GREEN, 2, cv2.LINE_AA)
    # সব landmark-এর coordinate-এ ছোট সাদা circle আঁকার loop শুরু।
    for index, point in enumerate(coordinates):
        # প্রতিটি landmark-এ radius ৪ pixel-এর ভরা সাদা dot আঁকা হয়; -1 মানে বৃত্ত ভরা।
        cv2.circle(frame, point, 4, WHITE, -1, cv2.LINE_AA)
    # পাঁচটি finger chain ও তাদের সিদ্ধান্তের ওপর loop চলে।
    for finger, chain in enumerate(FINGER_CHAINS):
        # সংশ্লিষ্ট আঙুল খোলা হলে সবুজ, বন্ধ হলে কমলা রঙ বেছে নেওয়া হয়।
        color = GREEN if hand["states"][finger] else ORANGE
        # chain-এর শেষ landmark—আঙুলের ডগায়—সিদ্ধান্ত অনুযায়ী বড় রঙিন dot আঁকা হয়।
        cv2.circle(frame, coordinates[chain[-1]], 8, color, -1, cv2.LINE_AA)
    # Landmark 0 হলো wrist; তার pixel coordinate label-এর জন্য নেওয়া হয়।
    x, y = coordinates[0]
    # লেবেল বাম ও ডান প্রান্তের বাইরে যেন না যায়, তাই x-কে frame-এর সীমার মধ্যে clamp করা হয়।
    x = max(8, min(x - 35, width - 190))
    # লেবেল উপরে/নিচে কেটে না যায়, তাই y-coordinate-ও frame-এর ভেতরে সীমাবদ্ধ করা হয়।
    y = max(28, min(y + 30, height - 14))
    # হাতের label ও খোলা আঙুলের সংখ্যা দিয়ে সংক্ষিপ্ত status string বানানো হয়।
    label = f'{hand["label"]}: {sum(hand["states"])} open'
    # প্রথমে মোটা কালো outline-সহ text আঁকা হয়, যাতে ভিডিওর যেকোনো পটভূমিতে পড়া যায়।
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                # outline-এর পুরুত্ব ৪ pixel এবং অ্যান্টি-এলিয়াসিং ব্যবহার করা হয়।
                (0, 0, 0), 4, cv2.LINE_AA)
    # এর ওপর সবুজ লেখাটি বসিয়ে label-এর দৃশ্যমানতা বাড়ানো হয়।
    write_text(frame, label, (x, y), color=GREEN, thickness=2)


# ভিডিও frame, হাতের detections, FPS ও mirror state নিয়ে সম্পূর্ণ output canvas বানায়।
def make_display(frame: np.ndarray, hands: list[dict], fps: float, mirror: bool) -> np.ndarray:
    """ভিডিওর পাশে হাত ও প্রতিটি আঙুলের অবস্থা দেখানোর জন্য আলাদা তথ্য-প্যানেল তৈরি করে।"""
    # ক্যামেরা অনুরোধ করা resolution উপেক্ষা করলেও প্রদর্শিত frame-কে ব্যবহারযোগ্য মাপের মধ্যে রাখা হয়।
    # বর্তমান frame-এর pixel-ভিত্তিক height ও width বের করা হয়।
    height, width = frame.shape[:2]
    # প্রস্থ ৯৬০ ও উচ্চতা ৭২০-এর মধ্যে fit করাতে scale বের হয়; ১-এর বেশি scale না হওয়ায় বড় করা হয় না।
    scale = min(960 / width, 720 / height, 1.0)
    # frame যদি সীমার তুলনায় বড় হয়, তবে resize করার শাখা।
    if scale < 1.0:
        # aspect ratio বজায় রেখে width ও height একই scale দিয়ে ছোট করা হয়।
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    # resize-এর পর প্রকৃত dimension আবার নেওয়া হয়, কারণ আগের মান আর মিলবে না।
    height, width = frame.shape[:2]
    # ডান পাশের তথ্য-প্যানেলের width ৩১০ pixel এবং উপরের header-এর height ৬৬ pixel নির্ধারণ।
    panel_width, header = 310, 66
    # canvas-এর ভিডিও-অংশ অন্তত ৬০০ pixel উঁচু রাখা হয়, যাতে প্যানেল নিচে অতিরিক্ত না চাপে।
    display_height = max(height, 600)
    # BACKGROUND রঙে পুরো ফাঁকা canvas বানানো হয়; শেষ dimension ৩-channel BGR image।
    canvas = np.full((display_height + header, width + panel_width, 3), BACKGROUND, dtype=np.uint8)
    # header-এর নিচে বাঁ দিকের অংশে resized camera frame copy করা হয়।
    canvas[header:header + height, :width] = frame
    # canvas-এর header-এ অ্যাপের নাম বড় ও মোটা অক্ষরে লেখা হয়।
    write_text(canvas, "HAND & FINGER TRACKER", (20, 29), 0.75, thickness=2)
    # সবুজ/কমলা রঙ কোন state বোঝায়, তার legend দেখানো হয়।
    write_text(canvas, "Green = open    Orange = closed", (20, 51), 0.46, MUTED)
    # ডান পাশের panel-এর text-এর x-coordinate হিসাব করা হয়।
    left = width + 18
    # প্রতিটি হাতের state list-এ True-এর সংখ্যা যোগ করে মোট খোলা আঙুল গণনা হয়।
    total_open = sum(sum(hand["states"]) for hand in hands)
    # প্রতি detected hand-এ পাঁচ আঙুল ধরে মোট আঙুল থেকে খোলা আঙুল বাদ দিয়ে বন্ধ আঙুল গণনা হয়।
    total_closed = 5 * len(hands) - total_open
    # মোট হাতের সংখ্যা ও গোল করে দেখানো FPS panel-এর উপরে লেখা হয়।
    write_text(canvas, f"HANDS {len(hands)}    FPS {fps:.0f}", (left, 31), 0.56, MUTED)
    # মোট খোলা আঙুল সবুজ রঙে বড় করে দেখানো হয়।
    write_text(canvas, f"OPEN: {total_open}", (left, 82), 1.05, GREEN, 2)
    # মোট বন্ধ আঙুল কমলা রঙে বড় করে দেখানো হয়।
    write_text(canvas, f"CLOSED: {total_closed}", (left, 117), 0.72, ORANGE, 2)

    # স্থিতিশীল ID অনুযায়ী হাতগুলো সাজিয়ে প্রত্যেকের জন্য আলাদা panel slot তৈরি হয়।
    for slot, hand in enumerate(sorted(hands, key=lambda item: item["id"])):
        # প্রতি হাতের panel শুরু করার y-coordinate; slot বাড়লে ২০৫ pixel নিচে সরানো হয়।
        top = 163 + slot * 205
        # এই হাতের পাঁচটি state-এর মধ্যে খোলা আঙুলের সংখ্যা বের করা হয়।
        opened = sum(hand["states"])
        # হাতের Left/Right label ও খোলা আঙুলের অনুপাত text-এ সাজানো হয়।
        write_text(canvas, f'{hand["label"]} hand  |  {opened}/5 open',
                   # আগের text-এর coordinate, font scale, রঙ ও thickness দিয়ে লেবেল আঁকার call সম্পূর্ণ হয়।
                   (left, top), 0.58, WHITE, 2)
        # আঙুলের নাম ও Boolean state-কে zip করে প্রতিটি আঙুলের UI row তৈরি হয়।
        for row, (name, is_open) in enumerate(zip(FINGER_NAMES, hand["states"])):
            # এই row-এর y-coordinate; প্রতিটি row ২৭ pixel ব্যবধানে।
            y = top + 31 + row * 27
            # বাম দিকে আঙুলের নাম ধূসর রঙে লেখা হয়।
            write_text(canvas, name, (left, y), 0.54, MUTED)
            # state True হলে OPEN, False হলে CLOSED লেখা বেছে নেওয়া হয় এবং সেই অনুযায়ী রঙও বেছে নেওয়া হয়।
            write_text(canvas, "OPEN" if is_open else "CLOSED", (left + 132, y),
                       # status text-এর font scale, রঙ ও thickness দিয়ে call শেষ হয়।
                       0.54, GREEN if is_open else ORANGE, 2)

    # যদি এই frame-এ কোনো হাত শনাক্ত না হয়, খালি panel-এর নির্দেশনা দেখানো হয়।
    if not hands:
        # ক্যামেরার সামনে এক বা দুই হাত দেখানোর বার্তা লেখা হয়।
        write_text(canvas, "Show one or both hands", (left, 179), 0.56)
        # সব আঙুল দৃশ্যমান রাখার পরামর্শ panel-এ দেখানো হয়।
        write_text(canvas, "Keep all fingers in view.", (left, 208), 0.5, MUTED)
        # হাত শনাক্ত না হলে আঙুলের ফল দেখানো সম্ভব নয়—এ কথা স্পষ্ট করা হয়।
        write_text(canvas, "No hand = no finger reading.", (left, 237), 0.47, MUTED)
    # canvas-এর নিচ থেকে ৪৯ pixel ওপরে footer text-এর y-coordinate নির্ধারিত হয়।
    footer = canvas.shape[0] - 49
    # quit এবং smoothing reset-এর keyboard shortcut footer-এ দেখানো হয়।
    write_text(canvas, "Q / Esc: quit    R: reset", (left, footer), 0.48, MUTED)
    # mirror চালু কি না, তার বর্তমান state-সহ footer-এর text তৈরি হয়।
    write_text(canvas, f'M: mirror ({"ON" if mirror else "OFF"})',
               # mirror status লেখা নির্দিষ্ট position-এ দেখানো হয়।
               (left, footer + 25), 0.48, MUTED)
    # আলো ও আংশিক আড়াল হওয়া আঙুলের সীমাবদ্ধতা সম্পর্কে সতর্কবার্তা শুরু।
    write_text(canvas, "Use good light. Partly bent / hidden fingers may be misread.",
               # সতর্কবার্তাটি canvas-এর নিচের বাম পাশে ছোট অক্ষরে লেখা হয়।
               (18, canvas.shape[0] - 17), 0.46, MUTED)
    # সম্পূর্ণ তৈরি image canvas main loop-কে ফেরত দেওয়া হয়।
    return canvas


# প্রোগ্রামের main entry function; model, MediaPipe, webcam, loop এবং cleanup এখানে একত্রিত।
def main() -> None:
    # মডেল যাচাই/ডাউনলোড করে ফাইলের local path পাওয়া হয়; model ছাড়া detection শুরু হবে না।
    model_path = ensure_model()
    # HandLandmarker চালানোর configuration object তৈরি শুরু।
    options = vision.HandLandmarkerOptions(
        # ফাইলের path-এ non-ASCII Windows username থাকলেও bytes দিয়ে model দেওয়া path-সংক্রান্ত সমস্যা এড়াতে সাহায্য করে।
        # মডেলের path string দেওয়ার বদলে ফাইলের bytes দেওয়া হয়, Windows username-এ Unicode থাকলেও path সমস্যা কমে।
        base_options=python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
        # VIDEO mode বেছে নেওয়া হয়, কারণ প্রতিটি webcam frame timestamp-সহ ধারাবাহিকভাবে প্রক্রিয়া হবে।
        running_mode=vision.RunningMode.VIDEO,
        # সর্বোচ্চ শনাক্তযোগ্য হাতের সংখ্যা global setting থেকে configuration-এ দেওয়া হয়।
        num_hands=MAX_HANDS,
        # নতুন হাতকে detected বলার ন্যূনতম confidence ০.৬ নির্ধারণ করা হয়।
        min_hand_detection_confidence=0.6,
        # দেখা হাতের landmark presence যাচাইয়ের ন্যূনতম confidence ০.৬ রাখা হয়।
        min_hand_presence_confidence=0.6,
        # আগের frame থেকে হাত track করার ন্যূনতম confidence ০.৫ নির্ধারণ করা হয়।
        min_tracking_confidence=0.5,
    # HandLandmarkerOptions-এর argument list বন্ধ করে configuration সম্পূর্ণ করা হয়।
    )
    # camera initialize হওয়ার আগে None রাখা হয়, যাতে finally block নিরাপদে resource পরীক্ষা করতে পারে।
    camera = None
    # প্রতি frame-এ finger-state smoothing ও hand-ID tracking-এর object তৈরি হয়।
    smoother = StateSmoother()
    # mirror-এর শুরু মান global setting MIRROR থেকে নেওয়া হয়।
    mirror = MIRROR
    # MediaPipe VIDEO mode-এর timestamp ক্রমবর্ধমান রাখতে শেষ timestamp-এর প্রাথমিক মান -১।
    last_timestamp = -1
    # FPS হিসাবের শুরুতে বর্তমান monotonic সময় সংরক্ষণ করা হয়।
    previous_time = time.perf_counter()
    # প্রথম frame-এর আগে smoothed FPS-এর প্রাথমিক মান ০।
    fps = 0.0
    # ব্যবহারকারীকে জানানো হয় যে ক্যামেরা চালু হচ্ছে এবং বন্ধের shortcut কী।
    print("Starting webcam. Click the video window; press Q or Esc to quit.")
    # landmarker resource নিরাপদে বন্ধ করার context manager-এর মধ্যে মূল কাজ শুরু।
    try:
        # options থেকে HandLandmarker তৈরি হয়; with block শেষ হলে এর resource বন্ধ হবে।
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            # ক্যামেরা খুলে প্রাথমিক frame camera ও frame variable-এ নেওয়া হয়।
            camera, frame = open_camera()
            # নির্ধারিত নামে resizable OpenCV window তৈরি করা হয়।
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # প্রাথমিক window size ১২০০×৬৪০ pixel করা হয়; এটি window size, camera resolution নয়।
            cv2.resizeWindow(WINDOW_NAME, 1200, 640)
            # যতক্ষণ ব্যবহারকারী quit না করেন বা ক্যামেরা ব্যর্থ না হয়, প্রতি frame-এ processing loop চলবে।
            while True:
                # এই frame-এর শুরুতে monotonic timestamp নেওয়া হয়; elapsed-time/FPS হিসাবের জন্য এটি wall-clock সময়ের চেয়ে উপযোগী।
                now = time.perf_counter()
                # mirror mode চালু থাকলে frame উল্টে দেওয়ার শাখা।
                if mirror:
                    # frame-কে অনুভূমিকভাবে flip করে selfie আয়না-ভিউ তৈরি করা হয়।
                    frame = cv2.flip(frame, 1)
                # OpenCV camera frame সাধারণত BGR; MediaPipe image input-এর জন্য RGB-তে রূপান্তর করা হয়।
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # RGB array contiguous করে MediaPipe-এর SRGB Image object বানানো হয়, যা detector-এ দেওয়া যাবে।
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                # VIDEO mode-এ webcam inference একই thread-এ করা যায়; প্রতিটি frame-এর timestamp আলাদা হতে হবে।
                # timestamp সব সময় বাড়তে থাকা আবশ্যক—খুব দ্রুত frame এলেও আগের timestamp পুনর্ব্যবহার করা যাবে না।
                # প্রতিটি timestamp আগের চেয়ে অন্তত ১ millisecond বড় রাখা হয়; খুব দ্রুত frame এলেও monotonic sequence বজায় থাকে।
                timestamp_ms = max(last_timestamp + 1, int(now * 1000))
                # পরের frame-এর তুলনার জন্য সর্বশেষ timestamp মনে রাখা হয়।
                last_timestamp = timestamp_ms
                # নতুন image ও timestamp দিয়ে MediaPipe-এর video detector চালিয়ে landmarks ফল পাওয়া হয়।
                result = landmarker.detect_for_video(image, timestamp_ms)
                # প্রতিটি শনাক্ত হাতের জন্য dictionary জমাতে ফাঁকা list তৈরি হয়।
                hands = []
                # MediaPipe যেসব hand landmark set ফেরত দিয়েছে, সেগুলোর ওপর index-সহ loop চলে।
                for index, landmarks in enumerate(result.hand_landmarks):
                    # একই index-এর hand-এর normalized 2D landmark নয়, angle calculation-এর জন্য 3D world landmarks নেওয়া হয়।
                    world = result.hand_world_landmarks[index]
                    # প্রতিটি world landmark-এর x,y,z নিয়ে shape (২১, ৩)-এর NumPy array বানানো হয়।
                    points = np.array([(point.x, point.y, point.z) for point in world])
                    # MediaPipe handedness ফল থেকে Left বা Right label নেওয়া হয়।
                    label = result.handedness[index][0].category_name
                    # MediaPipe handedness labels assume a mirrored/selfie view.
                    # mirror view বন্ধ থাকলে handedness label উল্টে দেওয়ার শাখা, কারণ model-এর label mirrored selfie-view ধরে।
                    if not mirror:
                        # Left-কে Right ও Right-কে Left করা হয়; অন্য অজানা label থাকলে সেটিই রেখে দেওয়া হয়।
                        label = {"Left": "Right", "Right": "Left"}.get(label, label)
                    # বর্তমান হাতের তথ্য একটি dictionary হিসেবে hands list-এ যোগ করা শুরু।
                    hands.append({
                        # আঁকার জন্য প্রতিটি landmark-এর normalized screen coordinate list-এ রাখা হয়।
                        "landmarks": landmarks,
                        # wrist landmark 0-এর normalized x,y দিয়ে হাত track করার anchor point রাখা হয়।
                        "wrist": np.array([landmarks[0].x, landmarks[0].y]),
                        # এই হাতের প্রদর্শনযোগ্য Left/Right নাম সংরক্ষণ করা হয়।
                        "label": label,
                        # 3D landmark array-কে geometry-based classifier-এ দিয়ে পাঁচ আঙুলের raw open/closed state বের করা হয়।
                        "states": classify_fingers(points),
                    # হাতের dictionary-র closing brace ও append call শেষ করে শনাক্ত হাতটি list-এ যোগ করা হয়।
                    })
                # বর্তমান raw detections-কে আগের frame-এর wrist track-এর সঙ্গে মিলিয়ে state smoothing ও stable ID যোগ করা হয়।
                hands = smoother.update(hands, now)
                # প্রতিটি smoothed হাতের ওপর drawing loop শুরু।
                for hand in hands:
                    # সেই হাতের skeleton, landmark এবং finger-state marker বর্তমান camera frame-এ আঁকা হয়।
                    draw_hand(frame, hand)
                # frame processing শেষের সময় মাপা হয়, FPS হিসাব করার জন্য।
                finished = time.perf_counter()
                # দুটি frame-এর মধ্যবর্তী সময়ের বিপরীত দিয়ে instantaneous FPS পাওয়া হয়; denominator-এর ক্ষুদ্র floor division-by-zero ঠেকায়।
                instantaneous_fps = 1.0 / max(finished - previous_time, 1e-6)
                # প্রথম মান সরাসরি নেওয়া হয়; পরে ৯০% পুরোনো ও ১০% নতুন মান মিশিয়ে FPS-এর ঝাঁকুনি কমানো হয়।
                fps = instantaneous_fps if fps == 0 else 0.9 * fps + 0.1 * instantaneous_fps
                # পরের iteration-এ elapsed time হিসাব করার জন্য বর্তমান finish time সংরক্ষণ করা হয়।
                previous_time = finished
                # প্রস্তুত display canvas তৈরি করে OpenCV window-তে দেখানো হয়।
                cv2.imshow(WINDOW_NAME, make_display(frame, hands, fps, mirror))
                # প্রায় ১ ms অপেক্ষা করে keyboard input পড়া হয়; & 0xFF key code-কে সামঞ্জস্যপূর্ণ পরিসরে রাখে।
                key = cv2.waitKey(1) & 0xFF
                # Escape, ছোট q বা বড় Q চাপা হয়েছে কি না যাচাই করা হয়।
                if key in (27, ord("q"), ord("Q")):
                    # quit key পাওয়া গেলে processing loop ত্যাগ করা হয়।
                    break
                # ব্যবহারকারী window-এর close button চাপলে window দৃশ্যমান কি না পরীক্ষা করা হয়।
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    # window বন্ধ হয়ে গেলে loop-ও থামানো হয়।
                    break
                # M/m key চাপলে mirror mode পাল্টানোর শাখা।
                if key in (ord("m"), ord("M")):
                    # mirror Boolean-কে উল্টে ON থেকে OFF বা OFF থেকে ON করা হয়।
                    mirror = not mirror
                    # mirror বদলালে wrist matching-এ stale coordinate/state না মেশাতে smoother-এর history নতুন করে শুরু হয়।
                    smoother = StateSmoother()
                # R/r key চাপলে smoothing history মুছে নতুন tracking state শুরু করার বিকল্প branch।
                elif key in (ord("r"), ord("R")):
                    # R চাপলে একই smoother reset করা হয়, কিন্তু mirror configuration অপরিবর্তিত থাকে।
                    smoother = StateSmoother()
                # পরবর্তী iteration-এর জন্য ক্যামেরা থেকে নতুন frame পড়া হয়।
                success, frame = camera.read()
                # ক্যামেরা frame পড়তে ব্যর্থ হয়েছে বা frame ফাঁকা—এমন পরিস্থিতি পরীক্ষা করা হয়।
                if not success or frame is None:
                    # ক্যামেরা disconnect/বন্ধ হলে কারণসহ error তোলা হয়, যাতে নীরবে খালি frame দিয়ে loop না চলে।
                    raise RuntimeError("Webcam stopped returning frames. Reconnect it and run again.")
    # try block সফল বা ব্যর্থ—যে অবস্থাই হোক resource cleanup চালানোর finally block শুরু।
    finally:
        # camera object সত্যিই তৈরি হয়ে থাকলে তবেই release করা হয়।
        if camera is not None:
            # ক্যামেরা device release করা হয়, যাতে অন্য অ্যাপ পরে সেটি ব্যবহার করতে পারে।
            camera.release()
        # OpenCV-এর খোলা সব window বন্ধ করে graphics resource পরিষ্কার করা হয়।
        cv2.destroyAllWindows()


# এই ফাইল সরাসরি run করা হলে তবেই নিচের main() চালানোর entry-point guard কাজ করে; import করলে নয়।
if __name__ == "__main__":
    # main() চালানোর সময় keyboard interrupt ও অন্য exception নিয়ন্ত্রিতভাবে ধরার try block শুরু।
    try:
        # প্রোগ্রামের মূল workflow শুরু হয়।
        main()
    # ব্যবহারকারী Ctrl+C চাপলে Python KeyboardInterrupt এখানে ধরা হয়।
    except KeyboardInterrupt:
        # স্বাভাবিকভাবে থামার সংক্ষিপ্ত বার্তা terminal-এ দেখানো হয়।
        print("\nStopped.")
    # অপ্রত্যাশিত runtime error হলে exception object-সহ এই handler চালু হয়।
    except Exception as error:
        # ব্যর্থতার বার্তা standard error stream-এ লেখা হয়, যা normal output থেকে আলাদা।
        print(f"\nCould not start or continue hand tracking:\n{error}", file=sys.stderr)
        # exit status ১ দিয়ে প্রোগ্রাম ব্যর্থ হয়েছে জানানো হয়; আসল exception-কে cause হিসেবে সংযুক্ত রাখা হয়।
        raise SystemExit(1) from error
