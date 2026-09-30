"""Graphite theme, from the plugin's tokens in docs/ui/ui-design.md (Visual system).

Amber marks a value or the selected item, blue marks something running, red-orange a warning,
as in the plugin. Nothing else is coloured.
"""

TOKENS = {
    "bg": "#141518",
    "panel": "#1C1E22",
    "line": "#2A2D33",
    "text": "#ECE8E1",
    "text2": "#A7A39B",
    "accent": "#E9A23B",
    "mod": "#6AA9E0",
    "warn": "#E07A5F",
    "ok": "#8DBF8B",
}

SANS = '"IBM Plex Sans", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif'
MONO = '"IBM Plex Mono", "SF Mono", Menlo, Consolas, monospace'


def stylesheet(t: dict = TOKENS) -> str:
    return f"""
* {{ font-family: {SANS}; font-size: 13px; color: {t['text']}; }}
QMainWindow, QWidget#page, QStackedWidget {{ background: {t['bg']}; }}
QWidget {{ background: transparent; }}
QLabel#h1 {{ font-size: 20px; font-weight: 600; }}
QLabel#h2 {{ font-size: 14px; font-weight: 600; }}
QLabel#muted, QLabel.muted {{ color: {t['text2']}; }}
QLabel#warn {{ color: {t['warn']}; }}
QLabel#ok {{ color: {t['ok']}; }}
QLabel#mono {{ font-family: {MONO}; }}
QFrame#card {{ background: {t['panel']}; border: 1px solid {t['line']}; border-radius: 8px; }}
QFrame#card[installed="true"] {{ border-color: {t['accent']}; }}

QListWidget#nav {{ background: {t['panel']}; border: none; border-right: 1px solid {t['line']};
    padding-top: 8px; outline: 0; }}
QListWidget#nav::item {{ padding: 10px 16px; margin: 2px 8px; border-radius: 6px; color: {t['text2']}; }}
QListWidget#nav::item:selected {{ background: {t['line']}; color: {t['text']};
    border-left: 3px solid {t['accent']}; }}
QListWidget#nav::item:hover:!selected {{ background: #22252A; }}

QPushButton {{ background: {t['panel']}; border: 1px solid {t['line']}; border-radius: 6px;
    padding: 6px 14px; min-height: 20px; }}
QPushButton:hover {{ border-color: {t['text2']}; }}
QPushButton:focus {{ border-color: {t['accent']}; }}
QPushButton:disabled {{ color: #5E5B55; border-color: {t['line']}; }}
QPushButton#primary {{ background: {t['accent']}; color: #1A1407; border-color: {t['accent']};
    font-weight: 600; }}
QPushButton#primary:disabled {{ background: #5A4526; color: #2A2112; }}
QPushButton#danger {{ color: {t['warn']}; }}
QPushButton#danger:disabled {{ color: #5E5B55; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{ background: {t['bg']};
    border: 1px solid {t['line']}; border-radius: 5px; padding: 4px 6px;
    selection-background-color: {t['accent']}; selection-color: #1A1407; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {t['accent']}; }}
QSpinBox, QDoubleSpinBox, QLineEdit#mono {{ font-family: {MONO}; }}
QComboBox QAbstractItemView {{ background: {t['panel']}; border: 1px solid {t['line']};
    selection-background-color: {t['line']}; }}
QPlainTextEdit#log {{ font-family: {MONO}; font-size: 12px; background: #0F1012; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t['text2']};
    border-radius: 3px; background: {t['bg']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}

QTableWidget {{ background: {t['panel']}; border: 1px solid {t['line']}; border-radius: 6px;
    gridline-color: {t['line']}; alternate-background-color: #1F2126; }}
QTableWidget::item {{ padding: 4px 8px; }}
QTableWidget::item:selected {{ background: {t['line']}; color: {t['text']}; }}
QHeaderView::section {{ background: {t['panel']}; color: {t['text2']}; border: none;
    border-bottom: 1px solid {t['line']}; padding: 6px 8px; font-weight: 600; }}
QTableCornerButton::section {{ background: {t['panel']}; border: none; }}

QProgressBar {{ background: {t['bg']}; border: 1px solid {t['line']}; border-radius: 4px;
    height: 10px; text-align: center; font-family: {MONO}; font-size: 11px; color: {t['text2']}; }}
QProgressBar::chunk {{ background: {t['mod']}; border-radius: 3px; }}
QStatusBar {{ background: {t['panel']}; border-top: 1px solid {t['line']}; color: {t['text2']}; }}
QStatusBar QLabel {{ color: {t['text2']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {t['line']}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {t['line']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['line']}; }}
"""
