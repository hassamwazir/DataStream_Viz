# PyQtGraph Serial Viewer (Polling '1')

This refactors a single-file script into a small, manageable package.

## Structure
```
pyqtgraph_serial_viewer/
├─ app.py
├─ config.py
├─ serial_reader.py
├─ main_window.py
├─ requirements.txt
└─ README.md
```

- `config.py` — shared defaults and constants
- `serial_reader.py` — `SerialReader` QThread for polling + streaming
- `main_window.py` — `MultiPlot` QMainWindow UI
- `app.py` — entry point
- `requirements.txt` — Python deps

## Install
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

## Notes
- Polling sends byte `b"1"`. If your device needs newline, edit `serial_reader.py` line marked with "WRITE HERE" to `b"1\n"` or `b"1\r\n"`.
- Keyboard: **P** (pause), **C** (clear).
- The rolling window auto-scales from the observed incoming sample rate.
