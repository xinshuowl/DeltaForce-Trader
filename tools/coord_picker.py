"""
坐标拾取工具 — 打开截图，点击即显示像素坐标
用法: python tools/coord_picker.py [图片路径]
默认打开 debug_full.png
"""
import sys
import os
import cv2
import numpy as np

clicks = []


def on_mouse(event, x, y, flags, param):
    display_img, scale, full_img, win_name = param
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x / scale)
        real_y = int(y / scale)
        clicks.append((real_x, real_y))

        display = display_img.copy()

        h, w = display.shape[:2]
        cv2.line(display, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(display, (0, y), (w, y), (0, 255, 255), 1)

        label = f"({real_x}, {real_y})"
        tx = min(x + 15, w - 200)
        ty = max(y - 15, 30)
        cv2.putText(display, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(display, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 放大镜 (点击位置周围 40x40 区域放大 4 倍), 使用内存中的 full_img, 避免重复读盘
        orig_x, orig_y = real_x, real_y
        fh, fw = full_img.shape[:2]
        crop_r = 20
        cx1 = max(0, orig_x - crop_r)
        cy1 = max(0, orig_y - crop_r)
        cx2 = min(fw, orig_x + crop_r)
        cy2 = min(fh, orig_y + crop_r)
        crop = full_img[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            zoom = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                              interpolation=cv2.INTER_NEAREST)
            zh, zw = zoom.shape[:2]
            cv2.line(zoom, (zw // 2, 0), (zw // 2, zh), (0, 255, 0), 1)
            cv2.line(zoom, (0, zh // 2), (zw, zh // 2), (0, 255, 0), 1)
            # 放在左上角, 限制尺寸不超过 display 范围
            max_zh = max(0, h - 12)
            max_zw = max(0, w - 12)
            use_zh = min(zh, max_zh)
            use_zw = min(zw, max_zw)
            if use_zh > 0 and use_zw > 0:
                display[10:10 + use_zh, 10:10 + use_zw] = zoom[:use_zh, :use_zw]
                cv2.rectangle(display, (9, 9),
                              (11 + use_zw, 11 + use_zh), (0, 255, 0), 2)

        cv2.imshow(win_name, display)
        print(f"  [{len(clicks)}] ({real_x}, {real_y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        real_x = int(x / scale)
        real_y = int(y / scale)
        cv2.setWindowTitle(win_name,
                           f"坐标拾取 | 当前: ({real_x}, {real_y}) | 左键点击记录 | Q退出")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "debug_full.png"

    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        print("请先在游戏出售页面按 Ctrl+4 生成 debug_full.png")
        return

    full = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if full is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w = full.shape[:2]
    print(f"图片尺寸: {w} x {h}")

    max_w, max_h = 1600, 900
    scale = min(max_w / w, max_h / h, 1.0)
    display_w = int(w * scale)
    display_h = int(h * scale)
    display_img = cv2.resize(full, (display_w, display_h))

    print(f"显示比例: {scale:.2f}x ({display_w}x{display_h})")
    print(f"坐标已自动换算为原图像素坐标 (即 config.py 需要的值)")
    print(f"左键点击记录坐标, 按 Q 退出\n")

    win_name = f"坐标拾取 | {os.path.basename(image_path)} ({w}x{h})"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse, (display_img, scale, full, win_name))
    cv2.imshow(win_name, display_img)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    cv2.destroyAllWindows()

    if clicks:
        print(f"\n已记录 {len(clicks)} 个坐标:")
        for i, (cx, cy) in enumerate(clicks, 1):
            print(f"  [{i}] ({cx}, {cy})")


if __name__ == "__main__":
    main()
