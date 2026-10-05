import argparse
import json
import sys
from pathlib import Path

from motionsense_app.data.importer import download_archive, prepare_dataset
from motionsense_app.data.store import DatasetStore
from motionsense_app.errors import DomainError, api_error
from motionsense_app.models.prediction import predict_rows
from motionsense_app.models.registry import ModelRegistry
from motionsense_app.models.training import train_model
from motionsense_app.settings import Settings


class _LocalizedArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "Cách dùng: ", 1)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(
            2,
            f"{self.prog}: lỗi: Tham số không hợp lệ. Dùng --help để xem hướng dẫn.\n"
            f"Chi tiết: {message}\n",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _LocalizedArgumentParser(prog="motionsense")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Tải và chuẩn bị dữ liệu UCI HAR")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="Tải archive chính thức qua HTTPS")
    source.add_argument("--archive", type=Path, help="Dùng archive ZIP đã tải sẵn")
    train = commands.add_parser("train", help="Huấn luyện và lưu phiên bản mô hình mới")
    train.add_argument("--profile", choices=("full", "reduced", "both"), required=True,
                       help="Cấu hình đầy đủ, rút gọn hoặc cả hai")
    train.add_argument("--trees", type=int, choices=(25, 50, 100, 150), default=50,
                       help="Số cây (mặc định 50)")
    predict = commands.add_parser("predict", help="Nhận diện một mẫu bằng mô hình đã lưu")
    predict.add_argument("--model-id", required=True, help="Mã phiên bản mô hình")
    predict.add_argument("--sample-id", required=True, help="Mã mẫu, ví dụ test:000001")
    serve = commands.add_parser("serve", help="Khởi động MotionSense tại máy")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        from scripts.launch import main as launch
        options = ['--port', str(args.port)]
        if args.no_browser:
            options.append('--no-browser')
        return launch(options)
    settings = Settings.default()
    try:
        if args.command == "prepare-data":
            archive = args.archive
            if args.download:
                archive = download_archive(settings.var_root / "source" / "uci-har.zip")
            dataset_id = prepare_dataset(archive, settings.data_dir)
            print(json.dumps(DatasetStore(settings.data_dir).info(), ensure_ascii=False, indent=2))
            print(f"dataset_id={dataset_id}")
        elif args.command == "train":
            store = DatasetStore(settings.data_dir)
            registry = ModelRegistry(settings.model_dir)
            profiles = ("full", "reduced") if args.profile == "both" else (args.profile,)
            items = []
            for profile in profiles:
                items.append(registry.publish(train_model(store, profile, args.trees)))
            print(json.dumps({"items": items}, ensure_ascii=False, allow_nan=False, indent=2))
        elif args.command == "predict":
            artifact = ModelRegistry(settings.model_dir).load(args.model_id)
            sample = DatasetStore(settings.data_dir).sample(args.sample_id)
            if artifact["manifest"]["dataset_id"] != sample["dataset_id"]:
                raise DomainError("MODEL_DATASET_MISMATCH", "Mô hình không tương thích dữ liệu hiện tại.", 503)
            prediction = predict_rows(artifact, [sample["features"]])[0]
            prediction.update(sample_id=sample["sample_id"], actual_label=sample["actual_label"])
            print(json.dumps(prediction, ensure_ascii=False, allow_nan=False, indent=2))
    except DomainError as exc:
        print(json.dumps(api_error(exc.code, exc.message, exc.details), ensure_ascii=False), file=sys.stderr)
        return 1
    except OSError:
        print(json.dumps(api_error("STORAGE_ERROR", "Không thể đọc hoặc lưu dữ liệu."), ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
