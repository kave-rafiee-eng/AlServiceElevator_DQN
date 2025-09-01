# Simulator.py
import pygame
from enum import Enum
import numpy as np

pygame.init()

class PassengerStatus(Enum):
    IN_HALL = 1
    IN_CABIN = 2
    ARRIVED = 3

class ElevatorSimulator:
    def __init__(self, width=1000, height=800, num_floors=10):
        # تنظیمات صفحه
        self.WIDTH = width
        self.HEIGHT = height
        self.SimulatorWidth = self.WIDTH / 2
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Elevator Simulation")

        # آسانسور و مسافرها
        self.NumOfFloor = num_floors
        self.FloorHeight = self.HEIGHT // self.NumOfFloor
        self.PassengerWidth = 40
        self.ElevatorX = 200
        self.ElevatorWidth = self.PassengerWidth * 5
        self.CabinPos = 0

        self.clock = pygame.time.Clock()
        self.passengers = []
        self.elevator_positions = []
        self.TotalMove = 0
        self.reward = 0
        self.totalReward = 0
        self.change = False
    # --------- مدیریت مسافرها ---------
    def add_passenger(self, origin, destination, status=PassengerStatus.IN_HALL):
        color = (255,0,0) if origin < destination else (0,200,0)
        self.passengers.append({
            "origin": origin,
            "destination": destination,
            "Hall_WT": 0,
            "Cabin_WT": 0,
            "status": status,
            "color": color,
            "rewarded": 0
        })

    # --------- محاسبه پاداش ---------
    def compute_reward(self, last_pos):
        reward = 0


        if self.CabinPos != last_pos:
            reward -= abs( self.CabinPos - last_pos ) * 10

        TotalHall = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_HALL:
                TotalHall+=1
        reward -= 0.2 * TotalHall  

        TotalCabin = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN:
                TotalCabin+=1

        if not self.change :
            reward -= 10
        elif  TotalCabin > 1 : 
            reward += TotalCabin * 3

        # ---- جریمه نهایی بر اساس کل حرکت‌ها ----
        if all(p["status"] == PassengerStatus.ARRIVED for p in self.passengers):
            reward -= 2 * self.TotalMove

        # ------------------ پاداش منفی حرکت نکردن ------------------
        if self.CabinPos == last_pos:
            reward += -40  # گیر کردن روی یک طبقه

        # ------------------ پاداش رسیدن مسافر ------------------
        arrived = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.ARRIVED and not p.get("rewarded", False):
                arrived += 1
                p["rewarded"] = True

        if arrived == 1 :
            reward = 2
        elif arrived > 1 :
            #reward =  arrived * ( 3 + arrived * 2  )
            reward =  arrived * 4

        # ------------------ مجازات زمان انتظار ------------------
        """
        total_wait = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_HALL:
                total_wait += p["Hall_WT"]  # جمع زمان انتظار
        reward -= 0.08 * total_wait  # مجازات بر اساس مجموع waitTime
       
        total_wait = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN:
                total_wait += p["Cabin_WT"]  # جمع زمان انتظار
        reward -= 0.1 * total_wait  # مجازات بر اساس مجموع waitTime
         """

        # ------------------ پاداش جهت حرکت ------------------
        
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN and p.get("Cabin_WT", 0) > 0:

                # جهت حرکت آسانسور
                CabinDir = last_pos - self.CabinPos # مثبت: بالا، منفی: پایین
                # جهت مقصد مسافر
                PassDir = p.get("origin", 0) - p.get("destination", 0) 

                #print(f"origin :{p['origin']}, destination :{p['destination']} , PassDir :{PassDir} , CabinDir :{CabinDir}" )

                # اگر حرکت آسانسور در جهت مقصد مسافر باشه
                #if CabinDir * PassDir > 0:
                    #reward += 0.02 * abs( CabinDir )  

                if CabinDir * PassDir < 0 :
                    reward -= 9 


        self.reward = reward
        self.totalReward += reward

        #print(f"reward :{self.reward}" )

        return reward

    # --------- تغییر موقعیت کابین ---------
    def move_to_floor(self, target_floor):
        Move = abs(self.CabinPos - target_floor) + 1
        self.TotalMove += Move
        lastCabinPos = self.CabinPos

        self.CabinPos = target_floor
        self.elevator_positions.append(self.CabinPos)

        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_HALL:
                p["Hall_WT"] += Move

        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN:
                p["Cabin_WT"] += Move

        self.change = False
        # تغییر وضعیت مسافرها
        for p in self.passengers:
            if p["origin"] == self.CabinPos and p["status"] == PassengerStatus.IN_HALL:
                p["status"] = PassengerStatus.IN_CABIN
                self.change = True
            if p["destination"] == self.CabinPos and p["status"] == PassengerStatus.IN_CABIN:
                p["status"] = PassengerStatus.ARRIVED
                p["color"] = "gray"
                self.change = True  

    # --------- وضعیت شبکه عصبی ---------
    def get_state(self):
        state = []

        # ---------- 1. موقعیت کابین ----------
        cabin_one_hot = [0]*self.NumOfFloor
        cabin_one_hot[self.CabinPos] = 1
        state.extend(cabin_one_hot)

        # ---------- 2. شاسی‌های طبقات ----------
        hall_calls = [0]*self.NumOfFloor
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_HALL:
                hall_calls[p["origin"]] = 1
        state.extend(hall_calls)

        # ---------- 3. شاسی‌های کابین ----------
        cabin_calls = [0]*self.NumOfFloor
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN:
                cabin_calls[p["destination"]] = 1
        state.extend(cabin_calls)

        return np.array(state, dtype=np.float32)

    # --------- توابع رسم ---------
    def draw_rect(self, x, y, w, h, color, text=None):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, color, rect)
        if text:
            font = pygame.font.SysFont("Arial", 14)
            label = font.render(text, True, (0,0,0))
            self.screen.blit(label, (x+2, y+2))

    def draw_line(self, x1, y1, x2, y2, color=(0,0,0), width=2):
        pygame.draw.line(self.screen, color, (x1,y1), (x2,y2), width)

    def draw_passenger(self, passenger, order):
        if passenger["status"] == PassengerStatus.IN_HALL:
            y = self.HEIGHT - (passenger["origin"]+1) * self.FloorHeight
            x = self.ElevatorX + self.ElevatorWidth + 10 + (self.PassengerWidth+5) * order
            self.draw_rect(x, y, self.PassengerWidth, self.FloorHeight-10, passenger["color"],
                      f"{passenger['origin']}→{passenger['destination']}")
        elif passenger["status"] == PassengerStatus.IN_CABIN:
            y = self.HEIGHT - (self.CabinPos+1) * self.FloorHeight
            x = self.ElevatorX + 5 + (self.PassengerWidth+5) * order
            self.draw_rect(x, y+5, self.PassengerWidth, self.FloorHeight-10, passenger["color"],
                      f"{passenger['origin']}→{passenger['destination']}")
        elif passenger["status"] == PassengerStatus.ARRIVED:
            y = self.HEIGHT - (passenger["destination"]+1) * self.FloorHeight
            x =  5 + (self.PassengerWidth+5) * order
            self.draw_rect(x, y+5, self.PassengerWidth, self.FloorHeight-10, passenger["color"],
                      f"{passenger['origin']}→{passenger['destination']}")

    def draw_positions(self):
        font = pygame.font.SysFont("Arial", 16)
        x, y = self.SimulatorWidth+50, self.HEIGHT//2

        label = font.render(f"Total: {self.TotalMove}", True, (0,0,0))
        self.screen.blit(label, (x, self.HEIGHT//2 - 50))

        label = font.render(f"totalReward: {self.totalReward}", True, (0,0,0))
        self.screen.blit(label, (x, self.HEIGHT//2 -120))

        label = font.render(f"reward: {self.reward}", True, (0,0,0))
        self.screen.blit(label, (x, self.HEIGHT//2 - 100))

        chunk_size = 10
        for i in range(0, len(self.elevator_positions), chunk_size):
            chunk = self.elevator_positions[i:i+chunk_size]
            text = " → ".join(map(str, chunk))
            label = font.render(text, True, (0,0,0))
            self.screen.blit(label, (x , y))
            y += 20

    def draw_passenger_table(self):
        font = pygame.font.SysFont("Arial", 16)
        x, y = self.SimulatorWidth+50, 20
        table_width = 5 * 80
        row_height = 20
        table_height = (len(self.passengers)+1) * row_height + 10

        pygame.draw.rect(self.screen, (255,255,255), (x-5, y-5, table_width+10, table_height+10))
        pygame.draw.rect(self.screen, (0,0,0), (x-5, y-5, table_width+10, table_height+10), 2)

        header = ["Origin", "Dest", "Status", "H_WT" , "C_WT"]
        for i, title in enumerate(header):
            label = font.render(title, True, (0,0,0))
            self.screen.blit(label, (x + i*80, y))

        for row, p in enumerate(self.passengers):
            vals = [p["origin"], p["destination"], p["status"].name, str(p["Hall_WT"]), str(p["Cabin_WT"])]
            for i, val in enumerate(vals):
                label = font.render(str(val), True, (0,0,0))
                self.screen.blit(label, (x + i*80, y + 20 + row*20))

    def render(self):
        # پاک کردن صفحه با رنگ پس‌زمینه
        self.screen.fill((230,230,230))

        # خطوط طبقات
        for i in range(self.NumOfFloor):
            y = self.HEIGHT - (i+1)*self.FloorHeight
            self.draw_line(0, y, self.SimulatorWidth, y, (200,0,0))

        # بدنه آسانسور
        for i in range(self.NumOfFloor):
            y = self.HEIGHT - (i+1)*self.FloorHeight
            self.draw_rect(self.ElevatorX, y, self.ElevatorWidth, self.FloorHeight, (0,0,200))

        # موقعیت کابین
        y = self.HEIGHT - (self.CabinPos+1)*self.FloorHeight
        self.draw_rect(self.ElevatorX, y, self.ElevatorWidth, self.FloorHeight, (0,0,0))

        # مسافرها
        arived_order = [0]*self.NumOfFloor
        hall_order = [0]*self.NumOfFloor
        cabin_order = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_HALL:
                self.draw_passenger(p, hall_order[p["origin"]])
                hall_order[p["origin"]] += 1
            elif p["status"] == PassengerStatus.IN_CABIN:
                self.draw_passenger(p, cabin_order)
                cabin_order += 1
            elif p["status"] == PassengerStatus.ARRIVED:
                self.draw_passenger(p, arived_order[p["destination"]])
                arived_order[p["destination"]] += 1

        # رسم جدول و موقعیت‌ها
        self.draw_passenger_table()
        self.draw_positions()

