import serial
import time
import sys

# --- KONFIGURACJA ---
# Zmień 'COM7' na port, pod którym wykryło Twoje ESP32
# (ten sam, który widzisz w Arduino IDE)
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200


def main():
    print(f"Próba połączenia z {SERIAL_PORT} przy prędkości {BAUD_RATE}...")

    try:
        # Otwarcie portu szeregowego
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Daj chwilę na reset ESP32 po otwarciu portu
        print("Połączono! Naciśnij Ctrl+C, aby zakończyć.\n")

    except serial.SerialException as e:
        print(f"\nBŁĄD: Nie można otworzyć portu {SERIAL_PORT}.")
        print("Sprawdź czy:")
        print("1. Podałeś poprawny numer portu COM.")
        print("2. Zamknąłeś Monitor Portu Szeregowego w Arduino IDE (port może być zajęty).")
        print(f"Szczegóły błędu: {e}")
        sys.exit(1)

    try:
        while True:
            # Sprawdź, czy są jakieś dane w buforze
            if ser.in_waiting > 0:
                try:
                    # Odczytaj linię, zdekoduj z bajtów na tekst i usuń białe znaki z końców
                    line = ser.readline().decode('utf-8', errors='ignore').strip()

                    # --- PARSOWANIE DANYCH ---
                    # Format z Arduino to: "Distance: 150 mm"
                    if "Distance:" in line:
                        parts = line.split(" ")  # Dzielimy po spacjach
                        # parts[0] = "Distance:"
                        # parts[1] = "150"
                        # parts[2] = "mm"

                        if len(parts) >= 2:
                            distance_value = parts[1]

                            # Tutaj możesz zrobić coś z tą wartością
                            # np. zapisać do pliku, wysłać do bazy danych itp.
                            print(f"Odebrano dystans: {distance_value} mm")

                    # Opcjonalnie: wypisuj też komunikaty o błędach czujnika
                    elif "Couldn't" in line or "Error" in line:
                        print(f"LOG Z ESP32: {line}")

                except UnicodeDecodeError:
                    # Czasami na początku transmisji mogą pojawić się śmieci
                    pass

    except KeyboardInterrupt:
        print("\nZatrzymywanie skryptu...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Port zamknięty.")


if __name__ == "__main__":
    main()