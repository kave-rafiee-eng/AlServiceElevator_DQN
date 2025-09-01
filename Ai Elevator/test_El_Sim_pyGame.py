import pygame
import sys
from enum import Enum

pygame.init()

# ---------- تنظیمات ----------
WIDTH, HEIGHT = 1000, 800
SimulatorWidth = WIDTH/2
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Elevator Simulation")

NumOfFloor = 8
FloorHeight = HEIGHT // NumOfFloor
PassengerWidth = 40

ElevatorX = 200
ElevatorWidth = PassengerWidth * 5
CabinPos = 0

clock = pygame.time.Clock()

# ---------- وضعیت مسافر ----------
class PassengerStatus(Enum):
    IN_HALL = 1
    IN_CABIN = 2
    ARRIVED = 3

reward = 0
TotalMove = 0
elevator_positions = []   # تاریخچه‌ی موقعیت آسانسور
passengers = []

def addPassenger(origin, destination, status, waitTime=0):
    color = (255,0,0) if origin < destination else (0,200,0)
    passengers.append({
        "origin": origin,
        "destination": destination,
        "waitTime": waitTime,
        "status": status,
        "color": color
    })

# نمونه مسافرها
addPassenger(1, 0, PassengerStatus.IN_HALL)
#addPassenger(2, 5, PassengerStatus.IN_HALL)
#addPassenger(2, 6, PassengerStatus.IN_HALL)
addPassenger(4, 0, PassengerStatus.IN_HALL)
addPassenger(6, 0, PassengerStatus.IN_HALL)
addPassenger(7, 2, PassengerStatus.IN_HALL)

# ---------- توابع رسم ----------
def draw_rect(x, y, w, h, color, text=None):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, color, rect)
    if text:
        font = pygame.font.SysFont("Arial", 14)
        label = font.render(text, True, (0,0,0))
        screen.blit(label, (x+2, y+2))

def draw_line(x1, y1, x2, y2, color=(0,0,0), width=2):
    pygame.draw.line(screen, color, (x1,y1), (x2,y2), width)

def draw_passenger(passenger, order, cabin_pos):
    if passenger["status"] == PassengerStatus.IN_HALL:
        y = HEIGHT - (passenger["origin"]+1) * FloorHeight
        x = ElevatorX + ElevatorWidth + 10 + (PassengerWidth+5) * order
        draw_rect(x, y, PassengerWidth, FloorHeight-10, passenger["color"],
                  f"{passenger['origin']}→{passenger['destination']}")

    elif passenger["status"] == PassengerStatus.IN_CABIN:
        y = HEIGHT - (cabin_pos+1) * FloorHeight
        x = ElevatorX + 5 + (PassengerWidth+5) * order
        draw_rect(x, y+5, PassengerWidth, FloorHeight-10, passenger["color"],
                  f"{passenger['origin']}→{passenger['destination']}")

    elif passenger["status"] == PassengerStatus.ARRIVED:
        y = HEIGHT - (passenger["destination"]+1) * FloorHeight
        x =  5 + (PassengerWidth+5) * order
        draw_rect(x, y+5, PassengerWidth, FloorHeight-10, passenger["color"],
                  f"{passenger['origin']}→{passenger['destination']}")

def draw_positions():
    font = pygame.font.SysFont("Arial", 16)
    x, y = SimulatorWidth+50, HEIGHT//2

    label = font.render(f"Total: {TotalMove}", True, (0,0,0))
    screen.blit(label, (x, HEIGHT//2 - 50))

    label = font.render(f"reward: {reward}", True, (0,0,0))
    screen.blit(label, (x, HEIGHT//2 - 100))


    # تقسیم آرایه به تکه‌های 10تایی
    chunk_size = 10
    for i in range(0, len(elevator_positions), chunk_size):
        chunk = elevator_positions[i:i+chunk_size]
        text = " → ".join(map(str, chunk))
        label = font.render(text, True, (0,0,0))
        screen.blit(label, (x , y))
        y += 20   # برو خط بعدی
        
def draw_passenger_table():
    font = pygame.font.SysFont("Arial", 16)
    x, y = SimulatorWidth+50, 20
    table_width = 4 * 80      # چون 4 ستون داری
    row_height = 20
    table_height = (len(passengers)+1) * row_height + 10

    # بک‌گراند سفید جدول
    pygame.draw.rect(screen, (255,255,255), (x-5, y-5, table_width+10, table_height+10))
    pygame.draw.rect(screen, (0,0,0), (x-5, y-5, table_width+10, table_height+10), 2)  # خط حاشیه مشکی

    # هدر جدول
    header = ["Origin", "Dest", "Status", "Wait"]
    for i, title in enumerate(header):
        label = font.render(title, True, (0,0,0))
        screen.blit(label, (x + i*80, y))

    # ردیف‌های جدول
    for row, p in enumerate(passengers):
        vals = [p["origin"], p["destination"], p["status"].name, str(p["waitTime"])]
        for i, val in enumerate(vals):
            label = font.render(str(val), True, (0,0,0))
            screen.blit(label, (x + i*80, y + 20 + row*20))

#------------------------- Ai Def
def compute_reward(CabinPos, passengers, last_pos):
    reward = 0

    # 1️⃣ هزینه حرکت (حرکت اضافی منفی)
    reward -= abs(CabinPos - last_pos)

    # 2️⃣ پاداش برای رسیدن مسافر به مقصد (فقط یک بار)
    for p in passengers:
        if p["status"] == PassengerStatus.ARRIVED:
            if not p.get("rewarded", False):
                reward += 10  # پاداش رسیدن به مقصد
                p["rewarded"] = True  # دیگر پاداش داده نشود
        # 3️⃣ هزینه انتظار مسافرها
        else:
            reward -= 0.1  # هر واحد زمان که مسافر نرسیده، منفی

    return reward

# ---------- حلقه اصلی ----------
running = True
direction = 1  # 1 بالا -1 پایین
timer_event = pygame.USEREVENT+1
pygame.time.set_timer(timer_event, 1000)  # هر 1 ثانیه حرکت کنه

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == timer_event:
            CabinPos = CabinPos
           #CabinPos += direction
            #if CabinPos >= NumOfFloor-1:
                #direction = -1
           #elif CabinPos <= 0:
                #direction = 1
                            
        if event.type == pygame.KEYDOWN:
            if pygame.K_0 <= event.key <= pygame.K_9:   # اعداد
                num = event.key - pygame.K_0
                if num < NumOfFloor:
                    target_floor = num
                    Move = abs( CabinPos - target_floor )
                    TotalMove += Move

                    lastCabinPos = CabinPos
                    reward = compute_reward( target_floor , passengers , lastCabinPos  )

                    CabinPos = target_floor   # مستقیم پرش به اون طبقه
                    elevator_positions.append(CabinPos)

                    for p in passengers:
                        if p["status"] != PassengerStatus.ARRIVED:
                            p["waitTime"] += Move

                    # تغییر وضعیت مسافرها
                    for p in passengers:
                        if p["origin"] == CabinPos and p["status"] == PassengerStatus.IN_HALL:
                            p["status"] = PassengerStatus.IN_CABIN
                        if p["destination"] == CabinPos and p["status"] == PassengerStatus.IN_CABIN:
                            p["status"] = PassengerStatus.ARRIVED
                            p["color"] = "gray"
 

    # ----- رسم -----
    screen.fill((230,230,230))

    # خطوط طبقات
    for i in range(NumOfFloor):
        y = HEIGHT - (i+1)*FloorHeight
        draw_line(0, y, SimulatorWidth, y, (200,0,0))

    # بدنه آسانسور
    for i in range(NumOfFloor):
        y = HEIGHT - (i+1)*FloorHeight
        draw_rect(ElevatorX, y, ElevatorWidth, FloorHeight, (0,0,200))

    # موقعیت کابین
    y = HEIGHT - (CabinPos+1)*FloorHeight
    draw_rect(ElevatorX, y, ElevatorWidth, FloorHeight, (0,0,0))

    # مسافرها
    arived_order = [0]*NumOfFloor
    hall_order = [0]*NumOfFloor
    cabin_order = 0
    for p in passengers:
        if p["status"] == PassengerStatus.IN_HALL:
            draw_passenger(p, hall_order[p["origin"]], CabinPos)
            hall_order[p["origin"]] += 1
        elif p["status"] == PassengerStatus.IN_CABIN:
            draw_passenger(p, cabin_order, CabinPos)
            cabin_order += 1
        elif p["status"] == PassengerStatus.ARRIVED:
            draw_passenger(p, arived_order[p["destination"]], CabinPos)
            arived_order[p["destination"] ] += 1

    draw_passenger_table()
    draw_positions()
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()

#------------------------

import numpy as np

def get_state(CabinPos, passengers, NumOfFloor=10):
    """
    بازگرداندن وضعیت فعلی آسانسور به عنوان ورودی شبکه عصبی
    ورودی: 
        CabinPos: موقعیت فعلی کابین (int)
        passengers: لیست دیکشنری مسافرها
        NumOfFloor: تعداد طبقات
    خروجی:
        state: آرایه 1 بعدی با اندازه 30 (10+10+10)
    """
    state = []

    # ---------- 1. موقعیت کابین (one-hot) ----------
    cabin_one_hot = [0]*NumOfFloor
    cabin_one_hot[CabinPos] = 1
    state.extend(cabin_one_hot)

    # ---------- 2. شاسی‌های طبقات (hall calls) ----------
    hall_calls = [0]*NumOfFloor
    for p in passengers:
        if p["status"] == PassengerStatus.IN_HALL:
            hall_calls[p["origin"]] = 1
    state.extend(hall_calls)

    # ---------- 3. شاسی‌های داخل کابین (cabin calls) ----------
    cabin_calls = [0]*NumOfFloor
    for p in passengers:
        if p["status"] == PassengerStatus.IN_CABIN:
            cabin_calls[p["destination"]] = 1
    state.extend(cabin_calls)

    return np.array(state, dtype=np.float32)