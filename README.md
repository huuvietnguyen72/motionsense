# MotionSense

Ứng dụng Windows tại máy để **phát lại hoạt động của một người**, nhận diện từ
đặc trưng cảm biến và phân tích mô hình Random Forest. Giao diện tiếng Việt gồm
**Theo dõi · Nhận diện · Phân tích mô hình · Dữ liệu**. Dịch vụ và giao diện cùng
nguồn tại `http://127.0.0.1:8765`; tài nguyên không dùng CDN.

## Cài đặt lần đầu

1. Cài **CPython Windows 3.13, 64-bit**, gồm Python Launcher (`py -3.13`).
   Không dùng bản Python MSYS2 thay thế. Kiểm tra bằng `py -3.13 --version`.
2. Cần Internet để cài gói Python, tải UCI HAR, và cài gói giao diện khi cần build.
   **Node.js >=22.12 và npm >=10** chỉ cần khi chưa có bản build hợp lệ.
   Nếu bản phát hành có `frontend/dist/build-manifest.json` khớp source,
   `package-lock.json` và các tài nguyên build, bộ cài dùng lại bản đó mà không cần Node.
3. Mở `Cai_dat.bat` trong thư mục MotionSense. Giữ cửa sổ mở cho đến thông báo
   **“MotionSense sẵn sàng”**. Lần đầu có thể mất vài phút để tải dữ liệu và
   huấn luyện hai cấu hình mặc định: đầy đủ/rút gọn, 50 cây, seed 42.
4. Mở `Khoi_dong.bat`. Trình duyệt chỉ mở khi dịch vụ đã sẵn sàng.

Đường dẫn có dấu và khoảng trắng được hỗ trợ; có thể gọi batch bằng đường dẫn
tuyệt đối từ thư mục khác. Bộ cài dùng `.venv` của sản phẩm, khóa cài đặt để tránh
hai lượt thiết lập cùng lúc, dùng lock có SHA-256 và chạy `pip check`. Chạy lại
`Cai_dat.bat` dùng lại môi trường, build, snapshot và mô hình tương thích; chỉ tạo
phiên bản mô hình còn thiếu. Không xóa dữ liệu phiên hoặc ghi đè mô hình cũ.

## Khởi động, kiểm tra và dừng

Sau thiết lập thành công, vận hành từ các tài nguyên tại máy không cần Internet.
Không cần chạy Node hoặc Vite để sử dụng.

```bat
Khoi_dong.bat
Khoi_dong.bat --port 8766
Khoi_dong.bat --no-browser
Khoi_dong.bat --check
```

- `--check` chỉ in JSON `{ready, missing}`, exit 0 khi sẵn sàng, exit 2 khi thiếu;
  không mở cổng hay trình duyệt. Tên thành phần: `environment`, `frontend`, `dataset`, `models`.
- Mở lại cùng thư mục **và cùng vùng trạng thái** trên cùng cổng sẽ dùng dịch vụ đang
  chạy. Cổng thuộc ứng dụng hoặc vùng trạng thái khác sẽ báo bận; chọn cổng trống.
- Giữ cửa sổ khởi động mở. **Nhấn Ctrl+C** ở cửa sổ đó để dừng dịch vụ và đóng
  tài nguyên lưu trữ. Đóng tab trình duyệt không dừng dịch vụ. Trình khởi động chỉ
  quản lý tiến trình con của chính nó.
  Nếu Windows hỏi `Terminate batch job (Y/N)?` sau khi dừng, chọn `Y` để đóng batch.
- Khi mở lại dịch vụ, phiên chưa kết thúc trở về **Tạm dừng**; mở phiên đã lưu
  và chọn **Tiếp tục** sau khi đồng bộ. Nếu đóng cưỡng bức, lần khởi động sau dùng
  cơ chế khôi phục từ các kết quả đã ghi.

Lệnh tương đương từ thư mục sản phẩm (PowerShell):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/launch.py --check
.\.venv\Scripts\python.exe -X utf8 -m motionsense_app.cli serve --port 8765 --no-browser
.\.venv\Scripts\python.exe -X utf8 scripts/install.py
```

Chỉ một worker được chạy, chỉ lắng nghe `127.0.0.1`. Nhật ký dịch vụ nằm tại
`var/logs/app.log`; nhật ký thiết lập tại `var/logs/install.log`.

## Bốn màn hình và luồng thao tác

### Theo dõi

Chọn người tham gia (ưu tiên tập kiểm tra) và mô hình sẵn sàng, tạo phiên rồi
**Bắt đầu**. Quan sát cửa sổ gia tốc X/Y/Z, hoạt động dự đoán, xác suất và lịch sử.
**Tạm dừng / Tiếp tục**, chọn tốc độ **0,5× / 1× / 2×**, hoặc **Kết thúc**.
1× là một cửa sổ mỗi giây phát lại để quan sát, không phải tốc độ ghi cảm biến gốc.
Phiên ghim vào mô hình đã chọn; huấn luyện mô hình mới không đổi mô hình của phiên.
Đổi người/mô hình tạo phiên mới sau khi tạm dừng hoặc kết thúc phiên hiện tại.

Mở phiên trong lịch sử để xem lại các mẫu và tổng kết, tiếp tục phiên tạm dừng,
hoặc **Xuất CSV**. Kết thúc sớm vẫn giữ kết quả đã xử lý. Tải lại trang, đổi
trình duyệt hoặc mất kết nối yêu cầu đồng bộ và tiếp tục có chủ ý; không tự chạy lại.

Nguồn là **Phát lại dữ liệu**. Các cửa sổ của một người được chọn theo thứ tự
dòng nguồn; phiên sản phẩm không phải một phiên ghi liên tục của nghiên cứu gốc.
Trục thời gian biểu đồ là thời gian **tương đối trong cửa sổ**. Mỗi cửa sổ 128
điểm ở 50 Hz, dài 2,56 giây, chồng lấn 50%. Phân bố hoạt động tính bằng **số mẫu**
và tỷ lệ, không cộng 2,56 giây/mẫu để suy ra thời lượng hoạt động.

### Nhận diện

Chọn mô hình và **Mẫu có sẵn** để nhận diện mẫu UCI HAR với tín hiệu đi kèm;
hoặc chọn **Tải CSV**. Tải **tệp CSV mẫu** của mô hình đang chọn trước khi nhập:

- Cột dùng mã `f001`…`f561`, khớp theo **tên**, không theo vị trí cột.
- Mô hình đầy đủ cần 561 cột; mô hình rút gọn cần đúng năm đặc trưng đã lưu trong
  phiên bản đó. CSV đầy đủ cũng dùng được cho mô hình rút gọn.
- Giá trị phải là số hữu hạn, không rỗng, không NaN/vô cực; không có cột trùng
  hoặc cột lạ. Dùng dấu phẩy phân cột và dấu chấm thập phân, mã hóa UTF-8 (BOM được chấp nhận).
- Metadata tùy chọn: `activity` (mã nhãn 1–6), `subject_id` (số nguyên dương).
  Metadata không đi vào mô hình. Giới hạn **16 MiB / 5.000 hàng dữ liệu**.
- Khi lỗi, sửa theo dòng/cột được báo rồi gửi lại; tệp chưa hợp lệ không được dự đoán.

CSV nhận diện chứa **đặc trưng đã trích xuất**, không nhận trực tiếp CSV tín hiệu
thô X/Y/Z. CSV đặc trưng không tự có biểu đồ tín hiệu. CSV xuất phiên là lịch sử
kết quả, không phải tệp đầu vào nhận diện.

Sáu hoạt động: 1 Đi bộ, 2 Lên cầu thang, 3 Xuống cầu thang, 4 Ngồi, 5 Đứng, 6 Nằm.

### Phân tích mô hình

Chọn phiên bản để xem cấu hình, đặc trưng, ma trận nhầm lẫn, độ quan trọng,
accuracy, macro F1 và chỉ số theo lớp. Có thể huấn luyện lại cấu hình đầy đủ
hoặc rút gọn với **25 / 50 / 100 / 150 cây**; mặc định 50, seed 42.
Tác vụ hiển thị giai đoạn thực tế; mô hình mới chỉ sẵn sàng sau lưu/tải kiểm chứng.
Thất bại giữ phiên bản cũ và báo lỗi; không ghi đè mô hình đang dùng.

**Đọc chỉ số đúng:**

- **Xác suất** là đầu ra cho từng lớp của một mẫu; không bảo đảm dự đoán đúng.
- **Accuracy kiểm tra** là tỷ lệ đúng trên tập test chính thức, khác accuracy train
  và accuracy của phiên đang chọn. **Macro F1** lấy trung bình F1 của sáu lớp.
- **OOB** là ước lượng trên tập huấn luyện, hiển thị riêng; cảnh báo nếu số cây
  khiến ước lượng thiếu tin cậy. OOB không thay thế đánh giá test.
- Train/test chia theo **người tham gia** của UCI, không trùng người giữa hai tập.
  Chọn đặc trưng rút gọn chỉ trên train: xếp hạng bằng 25 cây, lấy bốn vị trí đầu
  và vị trí thứ bảy. Không chọn/tối ưu dựa trên test.
- Chỉ số là kết quả mô hình đã huấn luyện, không phải accuracy cũ từ notebook.
  Thử nhiều cấu hình trên cùng test không tạo một tập đánh giá độc lập mới.

### Dữ liệu

Xem tình trạng snapshot, nguồn/giấy phép, 10.299 mẫu, 561 đặc trưng, 30 người,
train/test, phân bố hoạt động và lược đồ. Nếu snapshot chưa sẵn sàng, chạy thiết
lập; nếu có snapshot hỏng, bộ cài báo lỗi để khôi phục thay vì tự ghi đè.

## Lưu trữ và sao lưu

| Đường dẫn trong thư mục MotionSense | Nội dung |
|---|---|
| `.venv/` | Môi trường CPython và phụ thuộc đã khóa |
| `frontend/dist/` | Giao diện đóng gói và manifest hash |
| `var/source/uci-har.zip` | Archive tải từ nguồn chính thức |
| `var/datasets/current.json` | Định danh snapshot hiện tại |
| `var/datasets/<SHA-256>/` | Snapshot train/test, đặc trưng, manifest |
| `var/models/model-…/` | Mô hình, cấu hình/runtime, hash, báo cáo |
| `var/motionsense.sqlite3` | Phiên, kết quả và tác vụ huấn luyện |
| `var/logs/` | Nhật ký và tệp khóa OS (tệp khóa tồn tại không có nghĩa đang bận) |

Dừng dịch vụ trước khi sao lưu toàn bộ `var/`. Giữ dataset và mô hình được phiên
tham chiếu; thiếu chúng vẫn xem được lịch sử nhưng không thể tiếp tục nhận diện.
Không nhập mô hình `joblib` từ bên ngoài. Biến `MOTIONSENSE_STATE_ROOT` dùng cho
vùng trạng thái riêng: dữ liệu nằm tại `<state_root>/var`, code/frontend vẫn ở
thư mục sản phẩm. Trong sử dụng bình thường không cần đặt biến này. Đường dẫn
tương đối của biến được neo vào thư mục sản phẩm, không theo thư mục cửa sổ lệnh;
installer và launcher phải dùng cùng giá trị. Nên dùng đường dẫn tuyệt đối cho vùng riêng.

## Khắc phục lỗi thường gặp

| Tình huống | Hành động |
|---|---|
| Không có `py -3.13` | Cài CPython Windows 3.13 64-bit và Python Launcher; mở lại cửa sổ |
| Thiếu `.venv` hoặc runtime lock không khớp | Chạy `Cai_dat.bat`; kiểm tra `install.log` |
| `.venv` có sẵn nhưng thiếu interpreter/sai nền tảng | Sao lưu và kiểm tra môi trường; bộ cài không tự xóa hoặc sửa đè môi trường không hợp lệ |
| Cổng bận / cùng code nhưng khác state | Dùng `Khoi_dong.bat --port 8766` hoặc cổng trống khác; không dừng tiến trình không rõ nguồn |
| Giao diện thiếu hoặc manifest không khớp | Cài Node >=22.12/npm >=10 rồi chạy lại `Cai_dat.bat` để build |
| Download/pip/npm thất bại | Kiểm tra Internet và log, sửa lỗi rồi chạy lại; các phần hợp lệ được dùng lại |
| Bộ cài báo guard `.pending` sau timeout/ngắt cưỡng bức | Chọn **Restart / Khởi động lại Windows**, rồi chạy lại `Cai_dat.bat`; bộ cài chỉ tự khôi phục khi xác nhận định danh boot đã đổi |
| `current.json`/snapshot hỏng | Dừng dịch vụ, sao lưu `var`, khôi phục snapshot và pointer từ bản sao hợp lệ; không ghi đè dữ liệu người dùng |
| Model runtime khác hoặc thiếu/hỏng | Chạy bộ cài để thêm các mô hình mặc định tương thích; phiên ghim model cũ cần khôi phục model đó hoặc tạo phiên mới |
| Không ghi được dữ liệu/log | Kiểm tra dung lượng, quyền ghi của thư mục; thao tác chưa lưu không được coi là thành công |
| Mất kết nối / server không sẵn sàng sau 30 giây | Xem `app.log`, chạy lại launcher, đồng bộ phiên rồi chọn Tiếp tục |
| CSV không hợp lệ | Dùng tệp mẫu của đúng mô hình; sửa dòng/cột theo thông báo |

### Bộ cài bị ngắt bất thường (hiếm gặp)

Khi bước thiết lập bị timeout, Ctrl+C hoặc phải dừng cây tiến trình bất thường,
bộ cài trả lỗi và giữ guard `var/logs/.setup.lock.pending` trong thư mục sản phẩm,
cùng `var/logs/.install.lock.pending` trong vùng trạng thái. Cả tệp guard và bản
ghi bên trong tệp khóa lưu **định danh boot Windows** trước khi bước thiết lập chạy.
Ngay cả khi các tiến trình đã quan sát có vẻ đã dừng, bộ cài vẫn chặn retry trong
cùng boot: danh sách tiến trình tức thời không chứng minh đã bao trùm mọi descendant.

1. Xem `var/logs/install.log` và sửa nguyên nhân lỗi tải/build nếu có.
2. Lưu công việc của bạn, chọn **Restart / Khởi động lại Windows**. Đóng cửa sổ bộ
   cài, sleep/hibernate hoặc chỉ đăng xuất không phải là xác nhận reboot.
3. Chạy lại `Cai_dat.bat`. Sau khi lấy khóa OS, bộ cài đối chiếu định danh boot mới
   với bản ghi cũ và tự khôi phục guard nếu đã đổi. Không cần xóa tệp thủ công.

**Không xóa/sửa `.pending` hoặc các tệp khóa để bỏ qua kiểm tra.** Xóa `.pending`
không được coi là bằng chứng an toàn: bản ghi pending trong tệp khóa vẫn chặn
retry cùng boot. Nếu bản ghi boot bị hỏng hoặc là guard cũ không có định danh,
bộ cài giữ trạng thái lỗi; giữ log và khôi phục bản ghi guard hợp lệ từ bản sao,
không tự suy đoán an toàn theo tuổi tệp/PID. Nếu không đọc được định danh boot
Windows, bộ cài báo lỗi trước khi chạy tool.

Thiết lập thành công bình thường tự xóa guard và nhả khóa như trước. Guard này
chỉ chặn bộ cài; `Khoi_dong.bat` vẫn dùng bản cài hiện tại nếu preflight hợp lệ.

## Nguồn dữ liệu và giới hạn

UCI Human Activity Recognition Using Smartphones — Davide Anguita,
Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz (2012).
[Nguồn chính thức](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), **CC BY 4.0**.
Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) để biết ghi công và giấy phép.

Phiên bản này phát lại dữ liệu nghiên cứu, không thu cảm biến trực tiếp, không
dùng camera, không theo dõi nhiều người đồng thời và không cung cấp cảnh báo sức khỏe.

## Dành cho phát triển

`requirements.in` / `requirements-dev.in` và `package-lock.json` là nguồn khóa
phụ thuộc. Bộ cài không tạo lại lock. Chỉ nhà phát triển khi đổi phụ thuộc chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install --require-hashes -r requirements-dev.lock.txt
.\.venv\Scripts\python.exe -X utf8 scripts/lock_dependencies.py
.\.venv\Scripts\python.exe -m pytest tests/test_launch.py -q
.\.venv\Scripts\python.exe -m ruff check motionsense_app scripts tests
```

Tạo lại lock có thể cần Internet và mất nhiều phút; có giới hạn retry/timeout.
Sau thay đổi frontend, chạy bộ cài để build và tạo manifest mới. Không tự ký
manifest cho một bản dist chưa được build/kiểm chứng. Nghiệm thu đầy đủ với dữ
liệu thật, ngoại tuyến và phục hồi chạy bằng `scripts/acceptance.py`; kết quả
sinh trong `reports/` và không thuộc kho mã nguồn.
