import sys
from pathlib import Path

COMMAND_FILE = Path(__file__).resolve().parent / 'command.txt'

def send_command(command: str):
    """Creates a command file to be read by the main engine."""
    print(f"Sending command: {command.upper()}")
    try:
        with open(COMMAND_FILE, 'w') as f:
            f.write(command.upper())
        print("Command sent successfully. The engine will pick it up on its next cycle.")
    except Exception as e:
        print(f"Error sending command: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 control_bot.py <command>")
        print("Available commands: withdraw, resume")
        sys.exit(1)

    command_to_send = sys.argv[1]

    if command_to_send.lower() not in ['withdraw', 'resume']:
        print(f"Invalid command: {command_to_send}")
        print("Available commands: withdraw, resume")
        sys.exit(1)

    send_command(command_to_send)
