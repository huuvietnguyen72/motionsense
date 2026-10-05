import csv
import io
import math
from decimal import Decimal, InvalidOperation

from motionsense_app.errors import DomainError

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_UPLOAD_ROWS = 5000
ALLOWED_METADATA = {"activity", "subject_id"}
FLOAT32_MAX = 3.4028234663852886e38
# Configure once at import, rather than changing the process-wide setting per request.
# A field inside an allowed upload must not hit csv's smaller default 128 KiB limit.
csv.field_size_limit(max(csv.field_size_limit(), MAX_UPLOAD_BYTES))


def _cell_position(text: str) -> tuple[int, int]:
    """Locate the first broken quoted field, or the cell at the end of a prefix.

    Used only for diagnostics: csv.reader remains the parser. Respect quoted commas,
    escaped quotes and embedded newlines when locating syntax/encoding failures.
    """
    row, line, column = 1, 1, 0
    quoted, closed, start = False, False, True
    i = 0
    while i < len(text):
        char = text[i]
        if quoted:
            if char == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    i += 2
                    continue
                quoted, closed = False, True
        elif char == ',':
            column += 1
            closed, start = False, True
        elif char in '\r\n':
            column, closed, start = 0, False, True
        elif closed:
            break
        elif start and char == '"':
            quoted, start = True, False
        else:
            start = False
        if char == '\n' or (char == '\r' and text[i:i + 2] != '\r\n'):
            line += 1
            if not quoted:
                row = line
        i += 1
    return row, column


def validate_csv(
    content: bytes, required_features: list[str], known_features: list[str]
) -> tuple[list[dict[str, float]], list[int | None]]:
    """Validate the whole upload by column name before returning any prediction input."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise DomainError("CSV_TOO_LARGE", "Tệp CSV vượt giới hạn 16 MiB.", 413)
    details = []
    total_errors = 0

    def add(row, column, message):
        nonlocal total_errors
        total_errors += 1
        if len(details) < 50:
            details.append({"row": row, "column": column, "message": message})

    def reject():
        raise DomainError(
            "CSV_INVALID", f"Tệp CSV không hợp lệ: tổng cộng {total_errors} lỗi.",
            422, details,
        )

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        prefix = exc.object[:exc.start].decode("utf-8")
        row, index = _cell_position(prefix)
        try:
            header = next(csv.reader(io.StringIO(prefix, newline="")))
        except (StopIteration, csv.Error):
            header = []
        add(row, header[index] if index < len(header) else None, "Tệp phải dùng UTF-8.")
        reject()

    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, strict=True)
    try:
        header = next(reader, [])
    except csv.Error:
        add(1, None, "Cú pháp header CSV không hợp lệ.")
        reject()
    allowed = set(known_features) | ALLOWED_METADATA
    seen = set()
    for column in header:
        if column in seen:
            add(1, column, "Tên cột bị trùng.")
        elif column not in allowed:
            add(1, column, "Cột không thuộc lược đồ đặc trưng hoặc metadata cho phép.")
        seen.add(column)
    for column in required_features:
        if column not in seen:
            add(1, column, "Thiếu cột đặc trưng bắt buộc của mô hình.")
    if total_errors:
        reject()

    rows, labels = [], []
    count = 0
    required = set(required_features)
    while True:
        row_number = reader.line_num + 1
        position = stream.tell()
        try:
            cells = next(reader)
        except StopIteration:
            break
        except csv.Error:
            _, index = _cell_position(text[position:stream.tell()])
            add(row_number, header[index] if index < len(header) else None,
                "Cú pháp CSV không hợp lệ.")
            reject()
        count += 1
        if count > MAX_UPLOAD_ROWS:
            raise DomainError("CSV_TOO_MANY_ROWS", "Tệp CSV vượt giới hạn 5000 hàng.", 413, [
                {"row": row_number, "column": None, "message": "Tối đa 5000 hàng dữ liệu."}
            ])
        if len(cells) != len(header):
            missing = header[len(cells)] if len(cells) < len(header) else None
            add(row_number, missing, "Số ô không khớp header CSV; không được có hàng trống.")
            continue
        row, label = {}, None
        for column, cell in zip(header, cells):
            if column in ALLOWED_METADATA:
                try:
                    value = Decimal(cell)
                    if (not value.is_finite() or value != value.to_integral_value()
                            or value <= 0 or (column == "activity" and value > 6)):
                        raise ValueError
                    if column == "activity":
                        label = int(value)
                except (InvalidOperation, ValueError, OverflowError):
                    message = ("Nhãn phải là số nguyên từ 1 đến 6." if column == "activity"
                               else "Người tham gia phải là số nguyên dương.")
                    add(row_number, column, message)
            else:
                try:
                    value = float(cell)
                    if not math.isfinite(value):
                        raise ValueError
                    if column in required and abs(value) > FLOAT32_MAX:
                        raise ValueError
                    if column in required:
                        row[column] = value
                except (ValueError, OverflowError):
                    add(row_number, column,
                        "Đặc trưng phải là số hữu hạn; đầu vào mô hình phải thuộc miền float32.")
        rows.append({fid: row[fid] for fid in required_features if fid in row})
        labels.append(label)
    if count == 0:
        add(reader.line_num + 1, header[0] if header else None, "Tệp CSV không có hàng dữ liệu.")
    if total_errors:
        reject()
    return rows, labels
