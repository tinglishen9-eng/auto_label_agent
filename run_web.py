from pathlib import Path
import argparse
import logging
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_label_agent.web_app import start_web_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Label Agent Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", default=18081, type=int, help="监听端口")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    start_web_app(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
