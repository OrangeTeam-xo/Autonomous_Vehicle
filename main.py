import argparse, sys, time
import cv2, numpy as np

# ---- HSV rəng diapazonları ----
HSV_RED_1 = (0, 120, 70),  (10, 255, 255)
HSV_RED_2 = (170, 120, 70),(180, 255, 255)
HSV_GREEN = (35, 80, 60),  (85, 255, 255)

def make_masks(hsv):
    r1 = cv2.inRange(hsv, np.array(HSV_RED_1[0]),  np.array(HSV_RED_1[1]))
    r2 = cv2.inRange(hsv, np.array(HSV_RED_2[0]),  np.array(HSV_RED_2[1]))
    red   = cv2.bitwise_or(r1, r2)
    green = cv2.inRange(hsv, np.array(HSV_GREEN[0]), np.array(HSV_GREEN[1]))
    return red, green

def clean(mask, k=3):
    k = max(1, k|1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ker, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)
    return mask

def largest_cnt(mask, min_area=300):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, 0
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < min_area: return None, 0
    return c, a

def to_byte(val, vmin, vmax):
    if vmax <= vmin: return 0
    x = (val - vmin) / float(vmax - vmin)
    return int(np.clip(round(x*255.0), 0, 255))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--backend", type=str, default="dshow", help="auto|dshow|msmf")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="preview scale")
    ap.add_argument("--font",  type=float, default=0.7, help="HUD font scale")
    ap.add_argument("--th",    type=int,   default=2,   help="HUD thickness")
    args = ap.parse_args()

    backends = {
        "auto": 0,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }
    backend = backends.get(args.backend, 0)

    cap = cv2.VideoCapture(args.cam, backend)

    if args.backend in ("auto","dshow"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)

    if not cap.isOpened():
        print("❌ Camera open failed", file=sys.stderr)
        sys.exit(1)

    if args.show:
        cv2.namedWindow("view", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("view", int(args.w * args.scale), int(args.h * args.scale))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab fail", file=sys.stderr)
                break
            H, W = frame.shape[:2]

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask, green_mask = make_masks(hsv)
            red_mask   = clean(red_mask,   3)
            green_mask = clean(green_mask, 3)

            r_cnt, r_area = largest_cnt(red_mask,   300)
            g_cnt, g_area = largest_cnt(green_mask, 300)

            color = -1  # -1: NONE, 0: RED, 1: GREEN
            cnt = None
            if r_area == 0 and g_area == 0:
                color = -1
            elif r_area >= g_area:
                color = 0; cnt = r_cnt
            else:
                color = 1; cnt = g_cnt

            cx = 0; bw = 0; left_px = 0; right_px = 0
            overlay = frame.copy()

            if cnt is not None:
                x,y,w,h = cv2.boundingRect(cnt)
                cx = x + w/2.0
                bw = w
                left_px  = cx
                right_px = W - cx

                if args.show:
                    cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,255,255), 2)
                    cv2.circle(overlay, (int(cx), int(y+h/2)), 5, (255,255,255), -1)
                    cv2.line(overlay, (W//2,0), (W//2,H), (200,200,200), 1)

            left_b  = to_byte(left_px,  0, W)
            right_b = to_byte(right_px, 0, W)
            width_b = to_byte(bw,       0, W)

            print(f"color={color}  left_px={int(left_px)}  right_px={int(right_px)}  width_px={int(bw)} | bytes: {color},{left_b},{right_b},{width_b}", flush=True)

            if args.show:
                # HUD fon qutusu
                hud_h = 55
                bg = overlay.copy()
                cv2.rectangle(bg, (5, 5), (int(0.6*W), 5+hud_h), (0,0,0), -1)
                overlay = cv2.addWeighted(bg, 0.6, overlay, 0.4, 0)

                hud1 = f"COLOR={['RED','GREEN','NONE'][color if color>=0 else 2]}  W={int(bw)}"
                hud2 = f"L={int(left_px)}  R={int(right_px)}"

                cv2.putText(overlay, hud1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, args.font, (0,255,0), args.th, cv2.LINE_AA)
                cv2.putText(overlay, hud2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, args.font, (0,255,0), args.th, cv2.LINE_AA)

                if args.scale != 1.0:
                    overlay = cv2.resize(overlay, (int(W*args.scale), int(H*args.scale)))

                cv2.imshow("view", overlay)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            else:
                time.sleep(0.01)
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
