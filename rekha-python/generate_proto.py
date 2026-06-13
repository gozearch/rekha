"""Regenerate Python protobuf/gRPC stubs from proto/rekha.proto.

Usage:
    python generate_proto.py

Requires: grpcio-tools, protobuf
    pip install grpcio-tools
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    proto_dir = repo_root / "proto"
    output_dir = repo_root / "rekha-python" / "src" / "rekha" / "proto"

    proto_file = proto_dir / "rekha.proto"
    if not proto_file.exists():
        print(f"ERROR: {proto_file} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            "rekha.proto",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: protoc failed\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Fix the import in the generated grpc file to use relative import
    grpc_file = output_dir / "rekha_pb2_grpc.py"
    content = grpc_file.read_text()
    content = content.replace(
        "import rekha_pb2 as rekha__pb2",
        "from . import rekha_pb2 as rekha__pb2",
    )
    grpc_file.write_text(content)

    # Write __init__.py if missing
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    print(f"Generated stubs in {output_dir}")


if __name__ == "__main__":
    main()
