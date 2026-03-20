import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    # Zwingt PyTorch auf dem Mac dazu, auf die sichere CPU auszuweichen #diese Zelle ist von Gemini 3.1 Pro generiert
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import marimo as mo
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import torch

    # Sicherstellen, dass die CPU der Standard ist
    torch.set_default_device('cpu')

    import transformers
    import av
    from PIL import Image
    from ultralytics import YOLO
    import yt_dlp 
    from tqdm import tqdm

    return Path, YOLO, av, librosa, mo, np, plt, tqdm, transformers, yt_dlp


@app.cell
def _(Path, mo, yt_dlp):
    # YouTube Downloader

    def download_youtube_video(url, language_prefix="DE", video_id="01"):
        output_dir = Path("korpus")
        output_dir.mkdir(exist_ok=True)

        # Dateiname z.B. DE_01.mp4 oder KR_05.mp4
        filename = f"{language_prefix}_{video_id}.%(ext)s"

        ydl_opts = {
            # laedt das Video bis max 720p und das beste Audio dazu herunter
            'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]',
            'outtmpl': str(output_dir / filename),
            'quiet': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return mo.md(f"Video erfolgreich nach `{output_dir}` heruntergeladen!")

    return (download_youtube_video,)


@app.cell
def _(download_youtube_video):
    # Korpus zusammenstellen: den Link austauschen und Prefix/ID  anpassen
    download_youtube_video("https://youtu.be/OEisucdEznI?si=ZqK8mBLTSyxJ08Jj", language_prefix="DE", video_id="20")
    return


@app.cell
def _(librosa, np):
    # Audio-Features extrahieren
    import subprocess
    import tempfile

    def extract_audio_features(video_path):
        print(f"Lade Audiospur von: {video_path} ...")

        video_path_str = str(video_path)

        # temporaere WAV-Datei erstellen und Audio mit ffmpeg extrahieren
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:

            # subprocess Aufruf analog zu cube8.py
            subprocess.run([
                "ffmpeg",
                "-i", video_path_str,
                "-vn",                  # ignoriere die Videospur
                "-acodec", "pcm_s16le", # codiere als sauberes WAV
                "-ar", "22050",         # setze die sample rate fuer librosa
                "-y",                   # ueberschreibe die temporaere Datei
                temp_audio.name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            y, sr = librosa.load(temp_audio.name, sr=22050)

        # Pausenanteil via RMS
        rms_frames = librosa.feature.rms(y=y)[0]
        silence_threshold = 0.1 * np.max(rms_frames)
        pause_frames_count = np.sum(rms_frames < silence_threshold)
        silence_ratio = pause_frames_count / len(rms_frames)

        # Stimmcharakteristik Spectral Centroid
        s_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = np.mean(s_centroid)

        return {
            "silence_ratio": silence_ratio,
            "mean_centroid": mean_centroid,
            "rms_frames": rms_frames,
            "s_centroid": s_centroid,
            "threshold": silence_threshold
        }

    return (extract_audio_features,)


@app.cell
def _(plt):
    # Visualisierung Audio-Feautures
    def plot_audio_features(features, title="Audio Analyse"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

        # Plot RMS Energie und Pausen-Schwellenwert
        ax1.plot(features["rms_frames"], color='blue', alpha=0.7, label='RMS Energy')
        ax1.axhline(features["threshold"], color='red', linestyle='--', label='Stille-Schwellenwert')
        ax1.set_title(f"{title} | Pausenanteil: {features['silence_ratio']:.1%}")
        ax1.set_ylabel("Energie")
        ax1.legend()

        # Plot Spectral Centroid
        ax2.plot(features["s_centroid"], color='green', alpha=0.7)
        ax2.set_title(f"Spectral Centroid | Durchschnitt: {features['mean_centroid']:.0f} Hz")
        ax2.set_ylabel("Frequenz (Hz)")
        ax2.set_xlabel("Frames (Zeitverlauf)")

        plt.tight_layout()
        return fig


    return (plot_audio_features,)


@app.cell
def _(Path, extract_audio_features, plot_audio_features):
    # Aufruf Audio-Feature-Extraktion und Vis
    video_file = Path("korpus/KR_01.mp4") 

    if video_file.exists():
        audio_results = extract_audio_features(video_file)

        audio_plot = plot_audio_features(audio_results, title=f"Analyse von {video_file.name}")
    else:
        audio_plot = "Datei nicht gefunden."
        audio_results = None

    audio_plot
    return


@app.cell
def _(av, tqdm):
    # Preprocessing fuer YOLO und CLIP
    def extract_frames_from_video(video_path, extract_every_n_frames=30):
        """
        Lädt ein Video und extrahiert jedes N-te Frame als PIL-Image.
        Bei einem 30fps Video entspricht extract_every_n_frames=30 genau 1 Frame pro Sekunde.
        """
        print(f"Lade Video: {video_path}")
        container = av.open(str(video_path))

        frames = []
        # nutze tqdm fuer Ladebalken
        for i, frame in enumerate(tqdm(container.decode(video=0))):
            if i % extract_every_n_frames == 0:
                # wandelt PyAV-Frame in ein PIL Image um, fuer YOLO und CLIP
                img = frame.to_image()
                frames.append(img)

        print(f"{len(frames)} Frames erfolgreich extrahiert.")
        return frames

    return (extract_frames_from_video,)


@app.cell
def _(Path, extract_frames_from_video):
    # Pfad zu Korpus-Videos anpassen
    sample_video_path = Path("korpus/KR_01.mp4")

    # Frames extrahieren
    if sample_video_path.exists():
        video_frames = extract_frames_from_video(sample_video_path, extract_every_n_frames=30)

        if len(video_frames) > 10:
            preview_image = video_frames[10]
        else:
            preview_image = video_frames[0]
    else:
        preview_image = "Video noch nicht heruntergeladen."
        video_frames = []
    return (video_frames,)


@app.cell
def _(YOLO):
    print("Lade YOLOv8 Nano Modell (CPU Modus)...")
    # nutze yolov8n.pt anstelle von yolo11n oder yolo11x
    yolo_model = YOLO("yolov8n.pt")
    return (yolo_model,)


@app.cell
def _(np, yolo_model):
    def calculate_teacher_presence(frames):
        ratios = []

        # iteriere ueber alle extrahierten Frames
        for img in frames:
            # YOLO Vorhersage auf dem aktuellen Bild
            results = yolo_model(img, verbose=False)

            frame_ratio = 0
            for result in results:
                # Gesamtflaeche des Bildes berechnen
                img_width, img_height = img.size
                total_area = img_width * img_height

                max_person_area = 0
                for box in result.boxes:
                    # Klasse 0 ist "Person"
                    if int(box.cls[0]) == 0:  
                        # xywh : Center X, Center Y, Width, Height
                        _, _, w, h = box.xywh[0].tolist() 
                        area = w * h

                        # falls mehrere Personen erkannt werden (z.B. auf Plakaten im Hintergrund), 
                        # nehme die groesste Bounding Box (die echte Lehrkraft)
                        if area > max_person_area:
                            max_person_area = area

                # Verhaeltnis von Personengroesse zu Gesamtbild berechnen
                if max_person_area > 0:
                    frame_ratio = max_person_area / total_area

            ratios.append(frame_ratio)

        mean_ratio = np.mean(ratios) if ratios else 0
        return ratios, mean_ratio



    return (calculate_teacher_presence,)


@app.cell
def _(calculate_teacher_presence, plt, video_frames):
    # Aufruf Object detection
    if video_frames and len(video_frames) > 0:
        print(f"Analysiere {len(video_frames)} Frames mit YOLO...")
        presence_ratios, mean_presence = calculate_teacher_presence(video_frames)

        # Plot der Praesenz ueber die Zeit
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(presence_ratios, color='purple', linewidth=2)
        ax.set_title(f"Präsenz der Lehrkraft (Screen-Space-Ratio) | Durchschnitt: {mean_presence:.1%}")
        ax.set_ylabel("Bildschirmanteil")
        ax.set_xlabel("Frames (Zeitverlauf)")

        # Y-Achse geht von 0 bis 1 (entspricht 0% bis 100% des Bildschirms)
        ax.set_ylim(0, 1) 

        presence_plot = fig
    else:
        presence_plot = "Keine Frames zur Analyse vorhanden. Bitte zuerst load_sample_video() prüfen."
        presence_ratios = []
        mean_presence = 0

    presence_plot
    return (fig,)


@app.cell
def _(transformers):
    # Methode 3: Sentiment analysis
    print("Lade CLIP Modell (openai/clip-vit-base-patch32)...")
    # task und model wie in uebung10.py
    clip_pipeline = transformers.pipeline(
       task="zero-shot-image-classification",
       model="openai/clip-vit-base-patch32",
       device=-1  # -1 bedeutet CPU-Nutzung
    )
    return (clip_pipeline,)


@app.cell
def _(clip_pipeline, np):
    def calculate_teacher_sentiment(frames):
        # definiere Klassen fuer die Lehrkraft
        labels = [
            "a photo of a happy and smiling teacher",
            "a photo of a serious and neutral teacher"
        ]

        happiness_scores = []

        # ueber alle frames analysieren
        for img in frames:
            # clip_pipeline gibt eine Liste von Dictionaries zurück (Label und Score)
            results = clip_pipeline(img, candidate_labels=labels)

            # suche den Score fuer das "happy"-Label heraus
            happy_score = 0
            for res in results:
                if "happy" in res["label"]:
                    happy_score = res["score"]
                    break

            happiness_scores.append(happy_score)

        mean_happiness = np.mean(happiness_scores) if happiness_scores else 0
        return happiness_scores, mean_happiness

    return (calculate_teacher_sentiment,)


@app.cell
def _(calculate_teacher_sentiment, plt, video_frames):
    if video_frames and len(video_frames) > 0:
        print(f"Analysiere Emotionen in {len(video_frames)} Frames mit CLIP...")

        happy_scores, mean_happy = calculate_teacher_sentiment(video_frames)

        fig_sentiment, ax_sentiment = plt.subplots(figsize=(16, 4))
        ax_sentiment.plot(happy_scores, color='orange', linewidth=2)
        ax_sentiment.set_title(f"Sentiment (Wahrscheinlichkeit für 'Happy Teacher') | Durchschnitt: {mean_happy:.1%}")
        ax_sentiment.set_ylabel("Wahrscheinlichkeit")
        ax_sentiment.set_xlabel("Frames (Zeitverlauf)")
        ax_sentiment.set_ylim(0, 1) 

        sentiment_plot = fig_sentiment
    else:
        sentiment_plot = "Keine Frames zur Analyse vorhanden."
        happy_scores = []
        mean_happy = 0

    sentiment_plot
    return


@app.cell
def _(
    Path,
    calculate_teacher_presence,
    calculate_teacher_sentiment,
    extract_audio_features,
    extract_frames_from_video,
    tqdm,
):
    # ganzes Korpus analysieren
    import pandas as pd

    def run_full_corpus():
        korpus_dir = Path("korpus")
        video_files = list(korpus_dir.glob("*.mp4"))

        if not video_files:
            return "Keine Videos im Ordner 'korpus' gefunden.", None

        results = []
        print(f"Starte Batch-Verarbeitung für {len(video_files)} Videos...")

        # Schleife ueber alle Videos mit Fortschrittsbalken
        for video_path in tqdm(video_files, desc="Gesamtfortschritt Korpus"):
            print(f"\n--- Verarbeite: {video_path.name} ---")

            # Sprache aus Dateinamen ableiten (z.B. "KR" aus "KR_01.mp4")
            language = video_path.stem.split("_")[0] if "_" in video_path.stem else "Unbekannt"

            # 1. Audio-Features
            try:
                audio_res = extract_audio_features(video_path)
                silence_ratio = audio_res["silence_ratio"]
                mean_centroid = audio_res["mean_centroid"]
            except Exception as e:
                print(f"Fehler bei Audio ({video_path.name}): {e}")
                silence_ratio, mean_centroid = 0, 0

            # 2. Frames extrahieren (jedes 60. Frame = alle 2 Sekunden, um Zeit zu sparen)
            frames = extract_frames_from_video(video_path, extract_every_n_frames=60)

            # 3. Visuelle Praesenz & Sentiment
            if frames:
                _, yolo_mean = calculate_teacher_presence(frames)
                _, sentiment_mean = calculate_teacher_sentiment(frames)
            else:
                yolo_mean, sentiment_mean = 0, 0

            # Ergebnisse im Dictionary speichern
            results.append({
                "Video": video_path.name,
                "Sprache": language,
                "Pausenanteil": silence_ratio,
                "Stimm_Centroid_Hz": mean_centroid,
                "Praesenz_Lehrkraft": yolo_mean,
                "Sentiment_Happy": sentiment_mean
            })

        df = pd.DataFrame(results)

        # Daten als CSV-Datei speichern
        csv_path = "korpus_ergebnisse.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n Verarbeitung abgeschlossen! Ergebnisse gespeichert in: {csv_path}")

        return df, csv_path



    return (run_full_corpus,)


@app.cell
def _(run_full_corpus):
    # Schleife starten, kann dauern
    df_results, file_path = run_full_corpus()
    return


@app.cell
def _(mo, plt):
    # Visualisierung
    import pandas as pds
    import seaborn as sns

    df = pds.read_csv("korpus_ergebnisse.csv")

    summary1 = df.groupby("Sprache").mean(numeric_only=True).reset_index()
    summary_table1 = mo.ui.table(summary1, selection=None)

    # Boxplots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pausenanteil
    sns.boxplot(data=df, x="Sprache", y="Pausenanteil", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Pausenanteil")
    axes[0, 0].set_ylabel("Anteil an Stille")

    # Stimm-Centroid
    sns.boxplot(data=df, x="Sprache", y="Stimm_Centroid_Hz", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Stimm-Centroid (Helligkeit der Stimme)")
    axes[0, 1].set_ylabel("Frequenz in Hz")

    # Praesenz (YOLO)
    sns.boxplot(data=df, x="Sprache", y="Praesenz_Lehrkraft", ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("Präsenz der Lehrkraft (Screen-Space-Ratio)")
    axes[1, 0].set_ylabel("Bildschirmanteil")

    # Emotion (CLIP)
    sns.boxplot(data=df, x="Sprache", y="Sentiment_Happy", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Emotionale Tonalität ('Happy Teacher')")
    axes[1, 1].set_ylabel("Wahrscheinlichkeit")

    plt.suptitle("Multimodale Rhetorik: Sprachlernvideos auf YouTube (DE vs. KR)", fontsize=16)
    plt.tight_layout()

    plt.savefig("ergebnisse_boxplots.png", dpi=300)
    return fig, summary_table1


@app.cell
def _(fig, mo, summary_table1):
    output1 = mo.vstack([
        mo.md("### 1. Zusammenfassung der Mittelwerte pro Sprache"),
        summary_table1,
        mo.md("---"),
        mo.md("### 2. Verteilung der Ergebnisse (Boxplots für das PDF)"),
        mo.md("*Die Datei 'ergebnisse_boxplots.png wurde im Ordner gespeichert.*"),
        mo.as_html(fig)
    ])
    return


if __name__ == "__main__":
    app.run()
