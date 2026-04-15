"""
Handwriting Digit Recognizer
Uses a CNN trained on MNIST dataset.
Requirements: pip install tensorflow numpy matplotlib pillow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw, ImageFilter
import sys


# ─────────────────────────────────────────────
#  1.  Build Model
# ─────────────────────────────────────────────

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
#  2.  Train / Load Model
# ─────────────────────────────────────────────

MODEL_PATH = "mnist_cnn.keras"

def get_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"[✓] Loaded saved model from '{MODEL_PATH}'")
        return model
    except Exception:
        print("[i] No saved model found — training from scratch...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    ]

    model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[✓] Test accuracy: {acc*100:.2f}%")

    model.save(MODEL_PATH)
    print(f"[✓] Model saved to '{MODEL_PATH}'")
    return model


# ─────────────────────────────────────────────
#  3.  Preprocess a drawn / loaded image
# ─────────────────────────────────────────────

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert any PIL image to a 28×28 normalised array ready for the model."""
    img = img.convert("L")                    # grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img).astype("float32")

    # If the image is light-on-dark already (MNIST style) keep it;
    # if it looks like dark-on-light (e.g. scanned paper) invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)


# ─────────────────────────────────────────────
#  4.  Interactive drawing canvas (matplotlib)
# ─────────────────────────────────────────────

class DrawingApp:
    CANVAS_SIZE = 280   # display pixels
    BRUSH_RADIUS = 14   # px on the 280×280 surface → ~1.4 cells on 28×28

    def __init__(self, model):
        self.model = model
        self._reset_canvas()

        self.fig = plt.figure(figsize=(11, 5), facecolor="#000000")
        self.fig.canvas.manager.set_window_title("Handwriting Digit Recognizer")

        # ── Layout ──────────────────────────────────────────
        gs = self.fig.add_gridspec(
            2, 3,
            left=0.04, right=0.96,
            top=0.88, bottom=0.12,
            hspace=0.35, wspace=0.35,
        )
        self.ax_draw   = self.fig.add_subplot(gs[:, 0])
        self.ax_digit  = self.fig.add_subplot(gs[0, 1])
        self.ax_bar    = self.fig.add_subplot(gs[:, 2])
        self.ax_conf   = self.fig.add_subplot(gs[1, 1])

        self._setup_axes()
        self._draw_canvas()
        self._connect_events()
        self._add_buttons()

        self.fig.text(
            0.5, 0.96,
            "Handwriting Digit Recognizer  ·  MNIST CNN",
            ha='center', va='top',
            fontsize=13, color='#ffffff', fontweight='bold'
        )

    # ── Canvas helpers ───────────────────────────────────────

    def _reset_canvas(self):
        self.pixel_canvas = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE), dtype=np.float32)
        self.drawing = False
        self.last_xy = None

    def _setup_axes(self):
        dark = "#000000"
        for ax in [self.ax_draw, self.ax_digit, self.ax_bar, self.ax_conf]:
            ax.set_facecolor(dark)
            for spine in ax.spines.values():
                spine.set_color('#333333')

        self.ax_draw.set_title("Draw Here", color='#ffffff', fontsize=10, pad=6)
        self.ax_draw.set_xlim(0, self.CANVAS_SIZE)
        self.ax_draw.set_ylim(self.CANVAS_SIZE, 0)
        self.ax_draw.set_xticks([])
        self.ax_draw.set_yticks([])
        self.ax_draw.set_aspect('equal')

        self.ax_digit.set_title("28×28 Input", color='#ffffff', fontsize=9, pad=4)
        self.ax_digit.set_xticks([])
        self.ax_digit.set_yticks([])

        self.ax_bar.set_title("Confidence per digit", color='#ffffff', fontsize=9, pad=4)
        self.ax_bar.set_facecolor(dark)
        self.ax_bar.tick_params(colors='#aaaaaa')
        for spine in self.ax_bar.spines.values():
            spine.set_color('#333333')

        self.ax_conf.set_facecolor(dark)
        self.ax_conf.set_xticks([])
        self.ax_conf.set_yticks([])
        for spine in self.ax_conf.spines.values():
            spine.set_color('#333333')

    def _draw_canvas(self):
        self.ax_draw.clear()
        self.ax_draw.set_facecolor("#000000")
        self.ax_draw.set_xlim(0, self.CANVAS_SIZE)
        self.ax_draw.set_ylim(self.CANVAS_SIZE, 0)
        self.ax_draw.set_xticks([])
        self.ax_draw.set_yticks([])
        self.ax_draw.set_title("Draw Here  (click + drag)", color='#ffffff', fontsize=10, pad=6)

        if self.pixel_canvas.max() > 0:
            # White digit on black background
            self.ax_draw.imshow(
                self.pixel_canvas,
                cmap='gray', vmin=0, vmax=1,
                extent=[0, self.CANVAS_SIZE, self.CANVAS_SIZE, 0],
                origin='upper'
            )

        self.fig.canvas.draw_idle()

    # ── Mouse / touch events ─────────────────────────────────

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)

    def _on_press(self, event):
        if event.inaxes == self.ax_draw and event.button == 1:
            self.drawing = True
            self.last_xy = (event.xdata, event.ydata)
            self._paint(event.xdata, event.ydata)

    def _on_release(self, event):
        if event.button == 1:
            self.drawing = False
            self.last_xy = None
            # Auto-predict when the user lifts the mouse
            if self.pixel_canvas.max() > 0:
                self._predict()

    def _on_motion(self, event):
        if self.drawing and event.inaxes == self.ax_draw:
            self._paint(event.xdata, event.ydata)
            self.last_xy = (event.xdata, event.ydata)

    def _paint(self, x, y):
        if x is None or y is None:
            return
        cx, cy = int(x), int(y)
        r = self.BRUSH_RADIUS
        for py in range(max(0, cy - r), min(self.CANVAS_SIZE, cy + r + 1)):
            for px in range(max(0, cx - r), min(self.CANVAS_SIZE, cx + r + 1)):
                dist = ((px - cx)**2 + (py - cy)**2) ** 0.5
                if dist <= r:
                    val = 1.0 - (dist / r) * 0.4   # soft brush
                    self.pixel_canvas[py, px] = min(1.0, self.pixel_canvas[py, px] + val)
        self._draw_canvas()

    # ── Buttons ──────────────────────────────────────────────

    def _add_buttons(self):
        ax_pred  = self.fig.add_axes([0.22, 0.02, 0.14, 0.07])
        ax_clear = self.fig.add_axes([0.06, 0.02, 0.14, 0.07])

        btn_style = dict(color='#111111', hovercolor='#222222')
        self.btn_pred  = Button(ax_pred,  'Predict ▶', **btn_style)
        self.btn_clear = Button(ax_clear, 'Clear ✕',   **btn_style)

        for btn in [self.btn_pred, self.btn_clear]:
            btn.label.set_color('#ffffff')
            btn.label.set_fontsize(10)

        self.btn_pred.on_clicked(self._predict)
        self.btn_clear.on_clicked(self._clear)

    # ── Predict ──────────────────────────────────────────────

    def _predict(self, _event=None):
        # Downscale 280→28
        pil_img = Image.fromarray((self.pixel_canvas * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(1))
        inp = preprocess_image(pil_img)

        probs = self.model.predict(inp, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

        self._show_results(inp[0, :, :, 0], probs, predicted, confidence)

    def _show_results(self, small_img, probs, predicted, confidence):
        # 28×28 preview — white digit on black
        self.ax_digit.clear()
        self.ax_digit.set_facecolor("#000000")
        self.ax_digit.imshow(small_img, cmap='gray', vmin=0, vmax=1)
        self.ax_digit.set_xticks([])
        self.ax_digit.set_yticks([])
        self.ax_digit.set_title("28×28 Input", color='#ffffff', fontsize=9, pad=4)

        # Big prediction label — white digit on pure black
        self.ax_conf.clear()
        self.ax_conf.set_facecolor("#000000")
        self.ax_conf.set_xticks([])
        self.ax_conf.set_yticks([])
        for sp in self.ax_conf.spines.values():
            sp.set_color('#333333')

        # "You wrote:" label
        self.ax_conf.text(
            0.5, 0.88, "You wrote:",
            transform=self.ax_conf.transAxes,
            ha='center', va='center',
            fontsize=11, color='#aaaaaa'
        )
        # The predicted digit in large WHITE text
        self.ax_conf.text(
            0.5, 0.50, str(predicted),
            transform=self.ax_conf.transAxes,
            ha='center', va='center',
            fontsize=58, fontweight='bold', color='#ffffff'
        )
        # Confidence underneath
        conf_col = '#4ade80' if confidence >= 0.80 else '#facc15' if confidence >= 0.50 else '#f87171'
        self.ax_conf.text(
            0.5, 0.10, f"{confidence*100:.1f}% confident",
            transform=self.ax_conf.transAxes,
            ha='center', va='center',
            fontsize=9, color=conf_col
        )

        # Bar chart
        self.ax_bar.clear()
        self.ax_bar.set_facecolor("#000000")
        colors = ['#ffffff' if i == predicted else '#333333' for i in range(10)]
        bars = self.ax_bar.barh(range(10), probs, color=colors, height=0.65)
        self.ax_bar.set_yticks(range(10))
        self.ax_bar.set_yticklabels([str(i) for i in range(10)], color='#ffffff', fontsize=9)
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xlabel("Probability", color='#aaaaaa', fontsize=8)
        self.ax_bar.tick_params(axis='x', colors='#666666', labelsize=8)
        for sp in self.ax_bar.spines.values():
            sp.set_color('#333333')
        self.ax_bar.set_title("Confidence per digit", color='#ffffff', fontsize=9, pad=4)

        for i, (bar, p) in enumerate(zip(bars, probs)):
            self.ax_bar.text(
                min(p + 0.02, 0.95), i,
                f"{p*100:.1f}%",
                va='center', fontsize=7,
                color='#cccccc'
            )

        self.fig.canvas.draw_idle()

    # ── Clear ────────────────────────────────────────────────

    def _clear(self, _event=None):
        self._reset_canvas()
        self._draw_canvas()
        for ax in [self.ax_digit, self.ax_bar, self.ax_conf]:
            ax.clear()
            ax.set_facecolor("#000000")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#333333')
        self.ax_digit.set_title("28×28 Input", color='#ffffff', fontsize=9, pad=4)
        self.ax_bar.set_title("Confidence per digit", color='#ffffff', fontsize=9, pad=4)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ─────────────────────────────────────────────
#  5.  Predict from an image file (CLI mode)
# ─────────────────────────────────────────────

def predict_from_file(model, image_path):
    img = Image.open(image_path)
    inp = preprocess_image(img)
    probs = model.predict(inp, verbose=0)[0]
    predicted = int(np.argmax(probs))

    print(f"\nImage : {image_path}")
    print(f"Predicted digit : {predicted}  ({probs[predicted]*100:.1f}% confidence)\n")
    print("All probabilities:")
    for i, p in enumerate(probs):
        bar = '█' * int(p * 30)
        print(f"  {i}: {bar:<30} {p*100:5.1f}%")

    # Quick visual
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), facecolor='#1a1a1a')
    axes[0].imshow(inp[0, :, :, 0], cmap='gray')
    axes[0].set_title(f"Prediction: {predicted}", color='white')
    axes[0].axis('off')
    colors = ['#4ade80' if i == predicted else '#4b5563' for i in range(10)]
    axes[1].barh(range(10), probs, color=colors)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([str(i) for i in range(10)], color='white')
    axes[1].set_facecolor('#1a1a1a')
    axes[1].tick_params(colors='gray')
    for sp in axes[1].spines.values():
        sp.set_color('#333')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  6.  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = get_model()

    if len(sys.argv) > 1:
        # CLI: python digit_recognizer.py my_digit.png
        predict_from_file(model, sys.argv[1])
    else:
        # Interactive drawing canvas
        app = DrawingApp(model)
        app.show()