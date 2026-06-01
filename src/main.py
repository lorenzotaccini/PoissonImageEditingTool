import sys
from PySide6.QtWidgets import QApplication
from ui import PoissonApp

def main():
    app = QApplication(sys.argv)
    window = PoissonApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
