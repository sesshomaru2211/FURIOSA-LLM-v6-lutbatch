# Phân tích report v3–v5 và hướng thử v6

Nguồn: ZIP report do bạn gửi, chứa 6 schedule JSON và source tương ứng. Những số ở mục 1–4 là **ước lượng compiler**, không phải thời gian thiết bị. File `v3-v5-evidence.json` ghi lại số liệu và index instruction để kiểm tra lại.

## 1. Kết quả cả 3 kernel

| Kernel | Schedule v3 | Schedule v5 | Thay đổi | Instruction v3 → v5 |
|---|---:|---:|---:|---:|
| QKV | 75209 | 75206 | −0,004% | 124 → 122 |
| Attention output | 62816 | 54418 | −13,37% | 279 → 134 |
| FFN | 504130 | 523288 | +3,80% | 634 → 705 |

V5 không phải bản cải thiện đồng đều: giữ thay đổi attention có căn cứ; thay đổi FFN phải được xem lại. Giảm 2 instruction ở RoPE của QKV hầu như không rút ngắn schedule.

Log RNGD v5 bạn gửi trước đó có QKV 159618, attention 102988, FFN 1106500, tất cả PASS. Đây là một lần chạy, không phải median của 3 lần, và không phải số trong các JSON này. Không cộng số kernel từ các bản khác nhau để công bố latency của một binary mới.

## 2. FFN: bỏ tensor lớn nhưng phát sinh nhiều lần chuẩn bị LUT

| Thành phần | v3 | v5 |
|---|---:|---:|
| Cặp load/expand LUT | 3 | 40 |
| Tổng thời lượng các load LUT | 2514 | 33520 |
| Tổng thời lượng các expand LUT | 3879 | 41360 |
| Decode F4→F8 riêng trước vòng lặp | 3 × 18263 | Không còn |
| Scale trọng số up + gate | 30 × 4121 | 30 × 4120 |
| Scale trọng số down | 10 × 6041 | 10 × 6040 |
| Contraction up + gate | 30 × 1225 | Không đổi |
| Contraction down | 10 × 1712 | Không đổi |

Các instruction LUT không có description. Việc gán chúng vào LUT dựa trên **đồ thị dependency**: load 4096 byte → Sub mở rộng → tensor phụ được dùng bởi instruction có `fetch_table_lookup`. Không chỉ đếm tên rỗng một cách tùy ý.

Số instruction tăng được giải thích chính xác: thêm 37 DmaLoad + 37 Sub, bỏ 3 Main decode = **thêm 71 instruction**. V5 xóa tensor F8 toàn ma trận, nhưng gọi LUT trong 15 block up + 15 block gate + 10 block down. Phần scale và matmul gần như không giảm thời lượng.

Các thời lượng operator ở bảng có thể chồng lấp. Không cộng chúng rồi xem là latency kernel. Chênh lệch makespan thực sự là **19158 cycle**, gồm ảnh hưởng thứ tự và khoảng chờ trong schedule mới.

### Tensor và tài nguyên

- Mỗi tensor packed up/gate/down trong layout hiện tại: 56,25 MiB. Mỗi tensor F8 đã decode ở v3: 112,5 MiB.
- Tensor BF16 tạm cho block up/gate4: 15 MiB; down12: 22,5 MiB. V5 giữ những tensor này, chỉ bỏ tensor F8 lớn.
- Đỉnh hợp các khoảng địa chỉ SRAM sống: v3 **185,65625 MiB**, v5 **89,8984375 MiB**. Tính trên lifetime nửa mở `[begin,end)`, hợp vùng địa chỉ để tránh đếm đôi alias; đây không phải số đo allocator trên thiết bị hoặc chứng minh mọi tile lớn đều vừa bộ nhớ.
- MainContext bận: 303780 → 248951 cycle; DmaEngine bận: 215617 → 246623; SubContext bận: 27861 → 65342. Đây là hợp khoảng thời gian mỗi context, không phải số cộng giữa các context.

**Kết luận:** giảm SRAM là thật, nhưng không tự động nhanh hơn. V5 chuyển chi phí từ decode ma trận sang thiết lập LUT lặp lại và thay đổi lịch DMA/Sub. Không có bằng chứng phép contraction nhanh hơn.

## 3. Attention output: cải thiện đúng phần nhiều lượt nhỏ

V3 xử lý channel scale bằng 15 lượt × 4 hàng. V5 dùng 56 hàng + 4 hàng cuối, giảm 15 lượt xuống 2 lượt. Đoạn scale hoàn tất ở cycle 49909 trong v3, ở cycle 41511 trong v5, chênh **8398**, đúng chênh makespan 62816 − 54418.

Load ma trận trọng số vẫn mất 26734 cycle; contraction chính vẫn 9863 cycle. Sau v5, đây là các phần lớn hơn để nghiên cứu, không còn nên tập trung gom channel scale thêm một vài hàng. Vì không có kết quả chứng minh thay đổi khác tốt hơn, v6 giữ nguyên source attention v5.

## 4. QKV: nạp trọng số quan trọng hơn giảm thêm một tensor RoPE

Ba load trọng số Q, K, V có tổng thời lượng **53499 cycle**. Hợp thời gian DmaEngine của toàn QKV là **66835/75209 ≈ 88,9%** ở v3. Contraction Q là 9863 cycle, K/V là 5063 cycle mỗi cái.

V5 giảm 658 cycle tổng thời lượng MainContext, nhưng makespan chỉ giảm 3 cycle: phần RoPE bị rút ngắn phần lớn nằm ngoài đường quyết định thời điểm hoàn tất. Điều này gợi ý ưu tiên layout/nạp trọng số và overlap nếu tối ưu QKV tiếp, không bảo đảm cứ đổi layout là nhanh. V6 giữ QKV v3 để cô lập phép thử FFN.

## 5. V6: giả thuyết cần kiểm chứng, chưa phải kết quả

Giữ đường inline LUT của v5 nhưng tăng số hàng mỗi lượt:

| Phép chiếu | v5 | v6 batch8x16 | Scale VRF mỗi slice ở block lớn |
|---|---|---|---:|
| Up | 15 × 4 | 7 × 8 + 4 | 7680 byte |
| Gate | 15 × 4 | 7 × 8 + 4 | 7680 byte |
| Down | 10 × 12 | 7 × 16 + 8 | 7680 byte |

Số lượt gọi LUT theo source: **40 → 24**. Mục tiêu là giảm thiết lập lặp lại mà không tái tạo tensor F8 toàn ma trận. VRF scale riêng dưới giới hạn 8192 byte đã gặp ở các lần build trước, nhưng compiler vẫn phải xác nhận tổng sử dụng và ràng buộc mapping. Tensor BF16 mỗi block lớn tăng lên 30 MiB, có thể làm scheduling xấu đi; không bỏ qua rủi ro này.

Không đổi thứ tự các cột reduction, vị trí làm tròn BF16, GeGLU, global scale, RMSNorm, residual hoặc reduce 8 slice của down. Batch8x12 là bản dự phòng giữ down12 nếu down16 không phù hợp.

Tham khảo API gốc: [Furiosa 0.6.0 TuTensor / fetch_table_lookup](https://docs.rs/furiosa-opt-std/0.6.0/furiosa_opt_std/prelude/struct.TuTensor.html#method.fetch_table_lookup). Số lần setup cụ thể nêu trên lấy từ schedule bạn gửi, không suy ra từ tài liệu API.

## 6. Điều kiện chấp nhận

1. Compiler 0.6.0 biên dịch được source; kiểm tra LUT thực tế có giảm và không phát sinh nút thắt lớn khác.
2. Không dùng kết quả mock/local để kết luận NPU PASS. Chạy nguyên bộ 3 test và nguyên fixture trên RNGD.
3. So sánh ít nhất 3 lần PASS độc lập, chạy xen kẽ hybrid-control với candidate; dùng median theo kernel.
4. Nếu FFN không cải thiện, giữ hybrid-control. Không nới tolerance, thay fixture hoặc lấy cycle của một lần FAIL.

Môi trường hỗ trợ hiện không có compiler Furiosa hoặc kết nối RNGD. Không có benchmark v6 được thực hiện ở đây.
