# V6: thử nghiệm giảm số lượt LUT của FFN

Đây là mã nguồn thử nghiệm, **chưa được biên dịch Furiosa hoặc chạy RNGD trong môi trường của người hỗ trợ**. Không có số cycle hay PASS v6 được xác nhận. Python/shell và tính bao phủ block được kiểm tra riêng; chúng không thay thế kiểm thử NPU.

## Vì sao có bản này?

Report v5 cho thấy FFN thực hiện 40 cặp nạp/chuẩn bị LUT thay vì 3 cặp ở v3. Các phép nhân scale và contraction gần như không nhanh hơn. V6 thử gom nhiều hàng hơn vào mỗi lần gọi LUT, vẫn giữ phép toán và kiểm thử gốc. Đọc `analysis/BOTTLENECK_VI.md` để xem số liệu và giới hạn kết luận.

| Candidate | QKV | Attention output | FFN |
|---|---|---|---|
| `hybrid-control` | v3 | v5 scale56 | v3, up/gate4, down12 |
| `ffn-batch8x16` | v3 | v5 scale56 | Inline LUT; up/gate7×8+4, down7×16+8 |
| `ffn-batch8x12` | v3 | v5 scale56 | Dự phòng: up/gate7×8+4, down10×12 |
| `v3-control` | v3 | v3 | v3 |

`hybrid-control` ghép những phần đã có PASS riêng trong log cũ; bản ghép mới vẫn phải chạy lại trên RNGD. Không lấy số đo kernel của các bản khác nhau làm số đo của bản ghép.

## 1. Giải nén vào thư mục mới

Không ghi đè thư mục v5. ZIP có các file ngay ở thư mục gốc, không lồng thêm một thư mục cùng tên.

Mở Ubuntu rồi chạy:

```bash
mkdir -p ~/FURIOSA-LLM-v6-lutbatch
cd ~/FURIOSA-LLM-v6-lutbatch
```

Trong Windows File Explorer (Win+E), mở:

```text
\\wsl$\Ubuntu-22.04\home\thanh\FURIOSA-LLM-v6-lutbatch
```

Giải nén nội dung ZIP vào đây. Quay lại Ubuntu và kiểm tra:

```bash
ls run_all.sh
bash run_all.sh --verify-only
```

Không dùng `sudo`. Không cần cài numpy hay tạo lại fixtures. Runner dùng compiler 0.6.0 và nightly-2026-05-01 đã cài; không tự tải hoặc đổi phiên bản.

## 2. Kiểm tra schedule trước

```bash
bash run_all.sh --schedule-only
```

Mặc định biên dịch schedule cả 3 kernel của `hybrid-control` và `ffn-batch8x16`. Dòng FFN đối chứng dự kiến khớp report cũ 504130, nhưng kết quả lần build này mới là căn cứ. Số lượt LUT v6 dự kiến 24 từ cấu trúc source, **phải kiểm tra schedule mới để xác nhận compiler thực sự sinh như vậy**.

Nếu build lỗi hoặc FFN không giảm, gửi `report-v6-lutbatch.zip` ở đường dẫn được in cuối lệnh. Không cần upload RNGD ngay. Không đổi test, tolerance hoặc fixture để làm PASS.

## 3. Build binary sau khi kiểm tra schedule

```bash
bash run_all.sh
```

Runner tạo thư mục chạy mới, không dùng binary của lần trước. `LAST_RUN.txt` luôn trỏ đến lần chạy gần nhất. Có thể xem thư mục bằng:

```bash
result_dir="$(<LAST_RUN.txt)"
ls "$result_dir"
wslpath -w "$result_dir"
```

Dán đường dẫn cuối vào thanh địa chỉ Windows File Explorer. Không cần gọi `explorer.exe` từ Ubuntu nếu WSL của bạn báo `Exec format error`.

## 4. Gửi RNGD

Với từng thư mục `upload-hybrid-control` và `upload-ffn-batch8x16`, chọn đúng 3 file **cùng thư mục**:

1. `test_runtime`
2. `fixtures.safetensors`
3. `remote_entrypoint.sh`

Thiết lập: Entrypoint `remote_entrypoint.sh`; Timeout `70`; Args để trống; Env `TUC_PROFILE_LEVEL=info`. Đặt Name theo candidate và số lần chạy.

Chạy xen kẽ đối chứng và candidate, ít nhất 3 lần độc lập mỗi bản. Cả 3 kernel phải PASS. Lưu mỗi log vào một file riêng, giữ dòng `Candidate` và `Source-SHA256`.

```bash
python3 compare_runs.py --control logs/hybrid-control --candidate logs/ffn-batch8x16
```

Giữ bản mới chỉ khi FFN giảm trên median RNGD và hai kernel còn lại không thoái lui đáng kể. Nếu batch8x16 không compile, có thể thử riêng bản down12:

```bash
bash run_all.sh --candidate ffn-batch8x12 --schedule-only
```

## Những gì không thay đổi

Tests gốc, fixture, Cargo.lock, toolchain, RMSNorm, GeGLU, global scale, residual và phép reduce 8 slice của down projection được giữ nguyên. Không có giá trị output cài sẵn, không giảm số test, không thay tolerance. V6 không sửa mapping QKV hoặc attention ngoài phần scale56 đã có trong v5.

## Kiểm tra local không có NPU

```bash
python3 validate_local.py
```

Script kiểm tra checksum, copy source, biên tile, đủ hàng và cấu trúc phép toán. Các kiểm tra runner sử dụng **mock compiler**, chỉ chứng minh đóng gói/xử lý lỗi hoạt động; không tạo một kết quả benchmark hợp lệ.
