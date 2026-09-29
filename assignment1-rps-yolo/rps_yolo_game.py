# 모듈 로딩
#import tflite_runtime.interpreter as tflite
import ai_edge_litert.interpreter as tflite
import numpy as np
import time
import cv2
import threading
import os

# 효과음 재생 라이브러리 (미설치 시 효과음 없이 동작)
try:
    import simpleaudio as sa
except ImportError:
    sa = None

# LiteRT 모델 선택
#modelPath = "best.tflite"
#modelPath = "best_int8.tflite"
modelPath = "best_w8a32.tflite"
print('model path:', modelPath)

# LiteRT 모델 로딩
interpreter = tflite.Interpreter(model_path = modelPath) # 모델 로딩
interpreter.allocate_tensors() # tensor 할당

# 모델 정보 얻기 
input_details = interpreter.get_input_details()  # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
print(input_details)
print(output_details)
input_index = input_details[0]['index']
output_index = output_details[0]['index']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
height = input_details[0]['shape'][2]
width = input_details[0]['shape'][3]
print('model input shape:', (height, width))

# BB 텍스트 및 색상 정의
ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]

# 모델 입력 크기
IMG_SIZE = 320

# Threshold 설정
CONF_TH = 0.4
IOU_TH  = 0.45

# 타이밍 설정
COUNTDOWN_SEC = 1.0 # 카운트다운 숫자 한 개당 유지 시간
FREEZE_SEC    = 3.0 # 판정 후 프레임 고정(Freeze) 유지 시간
STABLE_SEC    = 0.7 # 카운트다운 시작 전, 손 모양이 변하지 않아야 하는 유지 시간

# 텍스트 표시 위치 (겹침 방지를 위해 영역을 구분)
LABEL_Y = 18   # 상단 P1 / P2 라벨
SCORE_Y = 35   # 상단 중앙 스코어보드
FPS_Y   = 235  # 좌측 하단 FPS
MSG_Y   = 210  # 중앙 하단 상태/결과 메시지

# 판정 결과 저장 설정
SAVE_DIR  = 'captures'          # 판정 순간 이미지 저장 폴더
SLIDE_DIR = 'slide_captures'    # 발표 슬라이드용 타이밍 진행 캡처 저장 폴더
LOG_FILE  = 'results_log.csv'   # 판정 결과 기록 파일

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SLIDE_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('timestamp,result,p1_class,p2_class,p1_score,draw_score,p2_score,image_path\n')

def putCenteredText(frame, text, y, scale=1.1, color=(0,255,255), thickness=2):
    # 텍스트를 프레임 가로 중앙에 맞춰 표시 (겹침 방지를 위해 위치를 지정된 y로 통일)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, text, (text_x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)

def drawScoreboard(frame, p1_score, draw_score, p2_score):
    # 화면 상단 중앙에 스코어보드 표시 (모든 상태에서 항상 보이도록 매 프레임 호출)
    score_text = f'P1 {p1_score} : {draw_score} DRAW : {p2_score} P2'
    putCenteredText(frame, score_text, SCORE_Y, scale=0.9, color=(255,255,0), thickness=2)

def saveSlideCapture(frame, name):
    # 발표 슬라이드 증빙용 타이밍 캡처 저장 (Hold pose / 카운트다운 진행 과정)
    image_path = os.path.join(SLIDE_DIR, f'{name}.jpg')
    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

def letterbox(img, new_shape=(320,320), color=(114,114,114)):
    h, w = img.shape[:2]    
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_y, pad_y,
        pad_x, pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, pad_x, pad_y

def processImage(frame):

    # 화면을 좌/우로 나누는 경계선의 x좌표
    boundary_x = frame.shape[1] // 2

    # 경계선 표시
    cv2.line(frame, (boundary_x, 0), (boundary_x, frame.shape[0]), (255, 255, 0), 2)

    # 좌/우 영역에 P1, P2 고정 라벨 표시
    cv2.putText(frame, 'P1', (8, LABEL_Y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, 'P2', (boundary_x + 8, LABEL_Y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)

    # BGR을 RGB로 변경
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # letterbox 적용
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))

    # 0 ~ 1 사이 값으로 변경
    img = img_lb.astype(np.float32) / 255.0

    # 모델의 입력 형태로 수정: (1,3,320,320)
    #   최상위 차원 증가: (320,320,3) -> (1,320,320,3)
    img = np.expand_dims(img, axis=0)
    #   축 위치 변경: (1,320,320,3) -> (1,3,320,320)
    img = np.transpose(img, (0, 3, 1, 2))

    # 모델에 입력하여 결과 얻기
    #   input tensor 설정
    interpreter.set_tensor(input_index, img)
    #   모델 실행
    interpreter.invoke()
    #   output tensor 얻기: (1,7,2100) -> (7,2100) -> (2100,7)
    raw = interpreter.get_tensor(output_index)[0].transpose()

    # raw에서 각 Object 별로 confidence score의 최대값과 해당 클래스의 id를 추출
    class_scores = raw[:, 4:]                      # confidence score: (2100, 3)
    confidences = np.max(class_scores, axis=1)     # confidence score의 최대값: (2100,)
    class_ids = np.argmax(class_scores, axis=1)    # 해당 클래스의 id: (2100,)

    # confidence score가 설정한 임계값(CONF_TH)보다 높은 Object만 필터링
    keep_mask = confidences > CONF_TH
    filtered_raw = raw[keep_mask] # (N,7)
    scores = confidences[keep_mask] # (N,)
    classes = class_ids[keep_mask]  # (N,)

    # 중심점 좌표 [cx, cy, w, h]를 추출하고 좌상단 좌표 [x, y, w, h]로 변환
    cx, cy, w, h = filtered_raw[:, 0], filtered_raw[:, 1], filtered_raw[:, 2], filtered_raw[:, 3]
    x = cx - (w / 2)
    y = cy - (h / 2)
    boxes = np.stack([x, y, w, h], axis=-1) # (N,4)

    # Batched NMS 처리
    keep = cv2.dnn.NMSBoxesBatched(
            boxes, 
            scores, 
            classes, 
            score_threshold=CONF_TH, 
            nms_threshold=IOU_TH
    )

    # 최종 BB만 그리기
    draw_boxes = [(boxes[i], scores[i], classes[i]) for i in keep]

    # 경계선 기준 P1(왼쪽) / P2(오른쪽) 판정 결과 저장
    p1_detections = []
    p2_detections = []

    for (x,y,w,h), sc, cid in draw_boxes:
        x *= IMG_SIZE; y *= IMG_SIZE
        w *= IMG_SIZE; h *= IMG_SIZE

        x1 = (x - pad_x) / r
        y1 = (y - pad_y) / r
        x2 = (x + w - pad_x) / r
        y2 = (y + h - pad_y) / r

        x1 = int(np.clip(x1,0,frame.shape[1]))
        y1 = int(np.clip(y1,0,frame.shape[0]))
        x2 = int(np.clip(x2,0,frame.shape[1]))
        y2 = int(np.clip(y2,0,frame.shape[0]))

        # BB 중심 x좌표로 P1(왼쪽) / P2(오른쪽) 구분
        center_x = (x1 + x2) / 2
        if center_x < boundary_x:
            player = 'P1'
            p1_detections.append((ansToText[cid], sc))
        else:
            player = 'P2'
            p2_detections.append((ansToText[cid], sc))

        # BB 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)

        # 판정 결과 표시 (플레이어 구분 포함)
        # 박스가 상단 라벨 영역(P1/P2)과 가까우면 라벨을 박스 아래쪽에 표시
        label_y = y1 - 7 if y1 - 7 > LABEL_Y + 15 else y2 + 15
        cv2.putText(frame, f'[{player}] {ansToText[cid]} {int(sc*100)}%', (x1,label_y), cv2.FONT_HERSHEY_PLAIN, 1, colorList[cid], 2)

    return p1_detections, p2_detections

def judgeRPS(p1_class, p2_class):
    # 가위바위보 승/무/패 판정 (if~else 사용)
    if p1_class == p2_class:
        result = 'DRAW'
    elif p1_class == 'rock' and p2_class == 'scissors':
        result = 'P1 WIN'
    elif p1_class == 'scissors' and p2_class == 'paper':
        result = 'P1 WIN'
    elif p1_class == 'paper' and p2_class == 'rock':
        result = 'P1 WIN'
    else:
        result = 'P2 WIN'
    return result

def playTone(freq_list, duration=0.15, sample_rate=44100):
    # 주파수 리스트를 순서대로 재생 (simpleaudio 미설치 시 무시)
    if sa is None:
        return
    for freq in freq_list:
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(freq * t * 2 * np.pi)
        audio = (tone * (2**15 - 1) / np.max(np.abs(tone))).astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
        play_obj.wait_done()

def playResultSound(result_text):
    # 메인 루프(영상) 멈춤 없이 재생되도록 별도 쓰레드로 실행
    if result_text == 'DRAW':
        sound_thread = threading.Thread(target=playTone, args=([440, 440],), daemon=True)
        sound_thread.start()
    elif result_text == 'P1 WIN' or result_text == 'P2 WIN':
        sound_thread = threading.Thread(target=playTone, args=([523, 659, 784],), daemon=True)
        sound_thread.start()

def saveResult(save_frame, result_text, p1_class, p2_class, p1_score, draw_score, p2_score):
    # 디스크 쓰기(이미지 저장, 로그 기록)를 메인 루프와 분리해서 영상이 끊기지 않도록 처리
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # JPEG 품질을 낮춰 SD카드 저장 용량과 쓰기 시간을 절약
    image_path = os.path.join(SAVE_DIR, f'{timestamp}_{result_text.replace(" ", "_")}.jpg')
    cv2.imwrite(image_path, save_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'{timestamp},{result_text},{p1_class},{p2_class},{p1_score},{draw_score},{p2_score},{image_path}\n')

# 카메라 설정
cap = cv2.VideoCapture(0) # 0번 카메라 열기
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

# 윈도우 설정
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320+40, 240+60)

# 상태 변수 초기화
state = 'WAITING' # 'WAITING'(손 모양 안정 대기) -> 'COUNTDOWN'(카운트다운 진행) -> 'FREEZE'(판정 결과 고정)
count = 3            # 3-2-1 카운트다운 숫자
count_start = time.time()
freeze_start = 0
frozen_frame = None
result_text = ''
waiting_classes = (None, None) # WAITING 상태에서 직전 프레임의 (P1, P2) 손 모양
stable_start = time.time()     # 현재 손 모양이 유지되기 시작한 시각
hold_captured = False          # 'Hold pose...' 슬라이드 캡처 완료 여부 (라운드당 1회)
last_captured_count = None     # 마지막으로 슬라이드 캡처한 카운트다운 숫자

# 스코어보드 변수 초기화
p1_score = 0
p2_score = 0
draw_score = 0

startTime = time.time()
while(cap.isOpened()):

    if state == 'FREEZE':
        # 판정 순간 프레임 고정: 새 프레임을 읽지 않고 저장해둔 프레임을 그대로 사용
        display_frame = frozen_frame.copy()
    else:
        ret,frame=cap.read() # 사진 찍기 -> (240,320,3)
        if not ret: break
        display_frame = frame

        # FPS 표시 (실제로 새 프레임을 읽어올 때만 계산)
        curTime = time.time()
        fps = 1/(curTime - startTime)
        startTime = curTime
        cv2.putText(display_frame,f'FPS: {fps:.1f}',(5, FPS_Y),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,255),1)

    # 스코어보드는 상태와 무관하게 항상 표시 (캡처 이미지에도 함께 남음)
    drawScoreboard(display_frame, p1_score, draw_score, p2_score)

    if state == 'WAITING':
        # 이미지 처리 (P1: 왼쪽, P2: 오른쪽 판정 결과 반환)
        p1_detections, p2_detections = processImage(display_frame)

        if len(p1_detections) == 1 and len(p2_detections) == 1:
            cur_classes = (p1_detections[0][0], p2_detections[0][0])

            if cur_classes != waiting_classes:
                # 손 모양이 바뀜 -> 안정 유지 시간 리셋
                waiting_classes = cur_classes
                stable_start = time.time()
                hold_captured = False

            stable_elapsed = time.time() - stable_start
            if stable_elapsed >= STABLE_SEC:
                # 손 모양이 STABLE_SEC 동안 유지됨 -> 카운트다운 시작
                state = 'COUNTDOWN'
                count = 3
                count_start = time.time()
                last_captured_count = None
            else:
                putCenteredText(display_frame, f'Hold pose... {stable_elapsed:.1f}s', MSG_Y, scale=0.9)
                # 슬라이드용: 'Hold pose...' 진행 중인 모습을 라운드당 1회 자동 캡처
                if not hold_captured and stable_elapsed >= 0.5:
                    saveSlideCapture(display_frame, 'hold_pose')
                    hold_captured = True
        else:
            # 양쪽에 손이 정확히 1개씩 없으면 안정 유지 시간 리셋
            waiting_classes = (None, None)
            stable_start = time.time()
            hold_captured = False

            putCenteredText(display_frame, 'Show your hand!', MSG_Y, scale=0.9)

    elif state == 'COUNTDOWN':
        # 이미지 처리 (P1: 왼쪽, P2: 오른쪽 판정 결과 반환)
        p1_detections, p2_detections = processImage(display_frame)

        # 카운트다운 시간 경과 체크
        elapsed = time.time() - count_start
        if elapsed >= COUNTDOWN_SEC:
            count -= 1
            count_start = time.time()

        if count > 0:
            # 3, 2, 1 숫자 표시
            count_text = str(count)
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_PLAIN, 5, 5)[0]
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = (display_frame.shape[0] + text_size[1]) // 2
            cv2.putText(display_frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5)

            # 슬라이드용: 숫자가 바뀔 때마다 카운트다운 진행 캡처 (3 -> 2 -> 1)
            if count != last_captured_count:
                saveSlideCapture(display_frame, f'countdown_{count}')
                last_captured_count = count
        else:
            # 카운트다운 종료 -> 판정 수행 후 현재 프레임을 고정(Freeze)
            boxes = p1_detections + p2_detections

            if len(boxes) != 2:
                result_text = f'Need 2 hands! ({len(boxes)})'

                # 예외 상황도 판정과 동일하게 이미지+로그 자동 저장 (증거 캡처)
                save_frame = display_frame.copy()
                putCenteredText(save_frame, result_text, MSG_Y, scale=1.0, color=(0,0,255), thickness=2)
                save_thread = threading.Thread(
                    target=saveResult,
                    args=(save_frame, result_text, 'N/A', 'N/A', p1_score, draw_score, p2_score),
                    daemon=True
                )
                save_thread.start()
            elif len(p1_detections) != 1 or len(p2_detections) != 1:
                result_text = '1 hand per side!'

                # 예외 상황도 판정과 동일하게 이미지+로그 자동 저장 (증거 캡처)
                save_frame = display_frame.copy()
                putCenteredText(save_frame, result_text, MSG_Y, scale=1.0, color=(0,0,255), thickness=2)
                save_thread = threading.Thread(
                    target=saveResult,
                    args=(save_frame, result_text, 'N/A', 'N/A', p1_score, draw_score, p2_score),
                    daemon=True
                )
                save_thread.start()
            else:
                p1_class = p1_detections[0][0]
                p2_class = p2_detections[0][0]
                result_text = judgeRPS(p1_class, p2_class)

                # 스코어보드 갱신 (정상적으로 판정된 라운드만 집계)
                if result_text == 'P1 WIN':
                    p1_score += 1
                elif result_text == 'P2 WIN':
                    p2_score += 1
                else:
                    draw_score += 1

                # 승/무 효과음 재생
                playResultSound(result_text)

                # 스코어보드 콘솔 출력
                print(f'[{result_text}] P1 {p1_score} : {draw_score} DRAW : {p2_score} P2')

                # 판정 순간 이미지 저장 + 로그 기록 (결과 텍스트를 포함해서 저장, 별도 쓰레드로 실행)
                save_frame = display_frame.copy()
                putCenteredText(save_frame, result_text, MSG_Y, scale=2.2, color=(0,0,255), thickness=3)
                save_thread = threading.Thread(
                    target=saveResult,
                    args=(save_frame, result_text, p1_class, p2_class, p1_score, draw_score, p2_score),
                    daemon=True
                )
                save_thread.start()

            frozen_frame = display_frame.copy()
            state = 'FREEZE'
            freeze_start = time.time()

    elif state == 'FREEZE':
        # 판정 결과 텍스트를 고정된 프레임 위에 표시
        # 승/무 결과는 크게, 경고 메시지(Need 2 hands 등)는 작게 표시
        if result_text == 'P1 WIN' or result_text == 'P2 WIN' or result_text == 'DRAW':
            putCenteredText(display_frame, result_text, MSG_Y, scale=2.2, color=(0,0,255), thickness=3)
        else:
            putCenteredText(display_frame, result_text, MSG_Y, scale=1.0, color=(0,0,255), thickness=2)

        # FREEZE_SEC 동안 결과를 유지한 뒤 다음 라운드 대기 상태로 복귀
        if time.time() - freeze_start >= FREEZE_SEC:
            state = 'WAITING'
            waiting_classes = (None, None)
            stable_start = time.time()

    # 이미지 출력
    cv2.imshow('cam',display_frame)

     # 10ms 동안 키 입력 대기
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('r'):
        # 재시작: 스코어보드 초기화 후 손 모양 대기 상태부터 다시 시작
        p1_score = 0
        p2_score = 0
        draw_score = 0
        state = 'WAITING'
        waiting_classes = (None, None)
        stable_start = time.time()
        hold_captured = False
        last_captured_count = None
        print('[RESTART] P1 0 : 0 DRAW : 0 P2')

cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기
