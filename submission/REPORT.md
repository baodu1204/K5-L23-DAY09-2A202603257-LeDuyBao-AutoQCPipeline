# Báo cáo Day 09 — Lê Duy Bảo, 2A202603257

## 1. Kết luận trước
Pipeline quét **224 ảnh**, loại **31**, còn **193 ảnh** trong `clean_v1`. Sau mô phỏng duyệt 80 ảnh, không còn lỗi bắt buộc nhưng còn cảnh báo ở **34/193 ảnh (17.62%)**, vượt ngưỡng 10%, nên kết luận là **Hold**; `clean_v1` chỉ là bản nháp, chưa được phát hành.

## 2. Lỗi và phân việc
`02_error_list_v1.csv` ghi 212 dòng lỗi/cảnh báo: 67 lỗi (gồm lỗi intake và lỗi QC) và 145 cảnh báo.

| Nhóm/mã lỗi ở v1 | Số dòng/ảnh | Ai xử lý | Cách xử lý |
|---|---:|---|---|
| `BAD_SIZE` 3, `CORRUPT_IMAGE` 2, `DUP_IMAGE_EXACT` 8, `DUP_IMAGE_NEAR` 7, `LEAK_TEST` 11 | 31 ảnh | Máy (intake) | Loại ảnh hỏng/nhỏ, ảnh trùng và ảnh trùng với test/val khỏi dataset. |
| `BAD_CLASS_ID` 6, `BAD_CLASS_NAME` 2, `BAD_FORMAT` 3, `BOX_OUT_OF_IMAGE` 8, `BOX_ZERO` 4, `DUP_BOX` 7 | 30 dòng | Máy (autofix) | Sửa tên lớp, dòng nhãn hỏng/trùng và clip box vượt biên khi có thể; `03_fixes.csv` ghi 35 thao tác, tính cả các lượt chuyển việc. |
| `MISSING_LABEL_FILE` 6 | 6 ảnh | Người | Máy không tự tạo nhãn; chuyển sang người gán. |
| `BOX_HUGE` 3, `BOX_TINY` 6, `CLASS_RARE` 4, `EMPTY_LABEL` 11, `POSSIBLE_FALSE_POS` 24, `POSSIBLE_LOOSE_BOX` 27, `POSSIBLE_MISSING` 53, `POSSIBLE_WRONG_CLASS` 17 | 145 dòng cảnh báo | Người | Đây là cảnh báo/ý kiến cần xem xét, không phải kết luận chắc chắn của model. |

Hàng đợi có **121 ảnh**, xếp theo tổng trọng số ưu tiên của các mã trên mỗi ảnh. Ảnh đầu `img_0020` có priority **11.524** vì đồng thời có `FIXED_DROP_LINE`, `POSSIBLE_MISSING` và `POSSIBLE_WRONG_CLASS`. Với ngân sách 80, mô phỏng đã duyệt 80 ảnh; **74/80** có ít nhất một cảnh báo ngữ nghĩa, tổng cộng **88 cờ ngữ nghĩa** (51 missing, 17 wrong-class, 8 loose-box, 12 false-positive).

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 16 | `DUP_IMAGE_NEAR`: 1.00 / 0.50 → 1.00 / 0.83 | Tăng recall bắt ảnh trùng gần mà precision trên tập calibration vẫn là 1.00; ở 20, F1 giảm còn 0.67. |
| `second_opinion.conf` | 0.25 | 0.6 | `WRONG_CLASS`: 0.56 / 0.83 → 0.62 / 0.83; `MISSING`: 0.50 / 1.00 → 0.57 / 1.00; `LOOSE_BOX`: 0.29 / 0.40 → 0.20 / 0.20; `FALSE_POS`: 0.36 / 1.00 → 0.36 / 1.00 | Tăng precision cho wrong-class và missing mà giữ recall; đổi lại khả năng bắt loose-box thấp hơn. |

**Điểm calibration có thể đánh lừa:** hàng đợi nhỏ hơn không có nghĩa chất lượng tốt hơn. Thử `conf=0.8` giảm hàng đợi từ 121 xuống 98 ảnh, nhưng recall `POSSIBLE_MISSING` giảm từ 1.00 xuống 0.75. Mô phỏng đủ pipeline ở ngưỡng đó trả **Release** dù còn 11/193 ảnh cảnh báo (5.70%): đó là Release giả do bỏ sót cảnh báo. Calibration chỉ có 39 ảnh nên các chỉ số cũng không đảm bảo tổng quát hóa.

## 4. Cửa chặn và giới hạn
Chạy thử không có người duyệt bằng `python -m pipeline.run --budget 0 --work work/rep_nb_work --reports work/rep_nb --export-dir work/exp_nb`: kết luận **Rework**, còn **6 lỗi bắt buộc**, mã thoát **20**, `exported: null` (không xuất dataset). Sáu ảnh thiếu file nhãn không thể được máy tự bịa nhãn.

Một số giới hạn có bằng chứng:
- `phash_dist=16` gắn `img_0011` (ảnh tennis) với `img_0008` (ảnh trượt tuyết) là `DUP_IMAGE_NEAR`; xem ảnh cho thấy đây là báo trùng sai. Không nên tự động loại mọi cặp chỉ theo pHash.
- Ở `conf=0.6`, calibration cho `POSSIBLE_LOOSE_BOX` recall **0.20** (bỏ sót 4/5 lỗi được chèn); model thứ hai không đáng tin như ground truth.
- Trên cùng calibration, `POSSIBLE_FALSE_POS` precision **0.36** (5 TP, 9 FP). Bất đồng với model cần người phân xử; không thể coi mọi cờ là lỗi nhãn thật.

## 5. Train: raw vs clean
Hai biến thể dùng `yolo26n.pt`, 20 epoch, seeds 0 và 1; raw có 224 ảnh, clean có 193 ảnh.

| Biến thể | #ảnh train | Seed | mAP50 (TB ± SD) | mAP50-95 (TB ± SD) | Precision (TB ± SD) | Recall (TB ± SD) |
|---|---:|---|---:|---:|---:|---:|
| raw | 224 | 0, 1 | 0.7258 ± 0.0002 | 0.5117 ± 0.0038 | 0.7635 ± 0.0612 | 0.6490 ± 0.0281 |
| clean | 193 | 0, 1 | 0.7301 ± 0.0014 | 0.5007 ± 0.0013 | 0.8001 ± 0.0194 | 0.6243 ± 0.0119 |

`mAP50(clean − raw) = +0.004`, lớn hơn SD giữa hai seed được báo cáo cho từng biến thể; tuy nhiên chỉ có **2 seed** và test gồm **100 ảnh**, nên chưa đủ cơ sở kết luận clean tổng quát tốt hơn. Clean tăng precision nhưng giảm recall và mAP50-95. Hai khả năng cần kiểm tra thêm: (1) raw có **11 ảnh bị gắn `LEAK_TEST`**, có thể làm điểm raw lạc quan hơn — train lại raw sau khi loại các rò rỉ, giữ nguyên seed; (2) clean ít hơn 31 ảnh và có các sửa nhãn tự động — audit nhãn đã sửa, so AP từng lớp và chạy thêm seed/ablation với số ảnh tương đương.

## 6. Quyết định và ghi chú AI
**Quyết định: Hold.** Trước khi phát hành cần người xem các cảnh báo còn lại, sửa lỗi nhãn thật và xác nhận cảnh báo còn lại dưới 10% (tối đa 19/193 ảnh), sau đó chạy lại gate; không nâng `conf` chỉ để làm hàng đợi vừa ngân sách.

- **Dùng AI:** nhờ AI diễn giải sweep calibration, so sánh hàng đợi/gate và tổng hợp các mục báo cáo; không giao AI tự sửa nhãn dữ liệu.
- **Kiểm chứng:** chạy calibration và pipeline, kiểm tra CSV/summary, mở xem các cặp ảnh bị báo trùng; bộ test có **79 passed**. Chạy riêng `--budget 0` để xác nhận gate Rework và không xuất.
