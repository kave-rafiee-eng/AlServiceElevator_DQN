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
    def __init__(self, width=900, height=600, num_floors=10):
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
        self.CabinLastPos = 0

        self.clock = pygame.time.Clock()
        self.passengers = []
        self.elevator_positions = []
        self.TotalMove = 0
        self.reward = 0
        self.totalReward = 0
        self.change = False
        self.step = 0

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
            "rewarded": False,
            "boarded_rewarded":False
        })

    # --------- محاسبه پاداش ---------
    def compute_reward(self, last_pos):

        reward = 0.0

        # ------------------ 5. جریمه جهت اشتباه ------------------
        TotalCabin = 0
        wrongDir = False
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN:
                TotalCabin += 1
                cabin_dir = self.CabinPos - last_pos   # مثبت = بالا، منفی = پایین
                pass_dir = p["destination"] - p["origin"]
                if cabin_dir * pass_dir < 0:  # خلاف جهت
                    reward -= 1.5
                    wrongDir = True

        # ------------------ 2. جریمه توقف بیهوده ------------------
        if self.CabinPos == last_pos:
            reward += -0.9

        if not self.change :
            reward += -0.9
        else :
            unique_floors = len(set(
                [p["origin"] for p in self.passengers] +
                [p["destination"] for p in self.passengers]
            ))
                        
            reward += (1 / unique_floors) * 1 

            if not wrongDir :
                reward += (1 / unique_floors) * TotalCabin

        # ------------------ 7. پایان اپیزود ------------------
        if all(p["status"] == PassengerStatus.ARRIVED for p in self.passengers):
            
            unique_floors = len(set(
                [p["origin"] for p in self.passengers] +
                [p["destination"] for p in self.passengers]
            ))
            print(f" unique_floors={unique_floors}, steps={self.step} ")
            if unique_floors == self.step :
                reward += 5
            #print(f"All passengers arrived! , totalReward={self.totalReward:.2f}")

        """

                # ------------------ 4. پاداش رسیدن مسافر به مقصد ------------------
        for p in self.passengers:
            if p["status"] == PassengerStatus.ARRIVED and not p.get("arrived_rewarded", False):
                total_passengers = len(self.passengers)
                reward = (1 / total_passengers) * 2   
                p["arrived_rewarded"] = True

                # ------------------ 3. پاداش برای سوار کردن مسافر ------------------
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN and not p.get("boarded_rewarded", False):
                reward += 0.1
                p["boarded_rewarded"] = True

        TotalCabin = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN:
                TotalCabin+=1

        if not self.change :
            reward += -0.3
        #else :
            #reward += 0.2 * TotalCabin
        
        # ------------------ پاداش منفی حرکت نکردن ------------------
        if self.CabinPos == last_pos:
            reward += -0.3  # گیر کردن روی یک طبقه
        

        unique_origins = len(set(p["origin"] for p in self.passengers))
        unique_destinations = len(set(p["destination"] for p in self.passengers))
        unique_total = unique_origins + unique_destinations
        if self.step  >  unique_total :
            reward += -0.3


        # ------------------ پاداش رسیدن مسافر ------------------
        arrived = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.ARRIVED and not p.get("rewarded", False):
                arrived += 1
                p["rewarded"] = True

        if arrived >= 1:
            total_passengers = len(self.passengers)
            reward = (arrived / total_passengers) * 2   

        # ----------------- reward Cabin Passenger Direction------------------
        wrongDir = False
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_CABIN and p.get("Cabin_WT", 0) > 0:

                CabinDir = last_pos - self.CabinPos # مثبت: بالا، منفی: پایین
                PassDir = p.get("origin", 0) - p.get("destination", 0) 

                #print(f"origin :{p['origin']}, destination :{p['destination']} , PassDir :{PassDir} , CabinDir :{CabinDir}" )

                if CabinDir * PassDir < 0 :
                    reward += -0.5
                    wrongDir = True


        if  self.change and not wrongDir :
            reward += 0.1 * TotalCabin

        if all(p["status"] == PassengerStatus.ARRIVED for p in self.passengers):
            
            print(f"All passengers arrived! : { self.step } ")

            unique_origins = len(set(p["origin"] for p in self.passengers))
            unique_destinations = len(set(p["destination"] for p in self.passengers))
            unique_total = unique_origins + unique_destinations

            #reward += ( unique_total / self.step ) * 1 
            #print("unique_total", unique_total)


        #reward -= self.TotalMove 


                    if unique_total == self.step :
                reward += 200
            else :
            
        if self.CabinPos != last_pos:
            #reward -= abs( self.CabinPos - last_pos ) * 10

        TotalHall = 0
        for p in self.passengers:
            if p["status"] == PassengerStatus.IN_HALL:
                TotalHall+=1
        reward -= 0.2 * TotalHall  
        
        if not self.change :
            reward -= 10
        elif  TotalCabin > 1 : 
            reward += TotalCabin * 3

        # ---- جریمه نهایی بر اساس کل حرکت‌ها ----
        if all(p["status"] == PassengerStatus.ARRIVED for p in self.passengers):
            reward -= 2 * self.TotalMove
        '''



        # ------------------ مجازات زمان انتظار ------------------

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

        self.reward = reward
        self.totalReward += reward

        #print(f"reward :{self.reward}" )

        return reward

    # --------- تغییر موقعیت کابین ---------
    def move_to_floor(self, target_floor):

        self.CabinLastPos = self.CabinPos
        self.CabinPos = target_floor
        self.step += 1
        Move = abs( self.CabinLastPos - self.CabinPos ) + 1
        self.TotalMove += Move
        
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

        # ---------- ----------
        cabin_dir_onehot = [0, 0, 0]  # [down, stay, up]

        if self.CabinLastPos < self.CabinPos:
            cabin_dir_onehot[2] = 1  # بالا
        elif self.CabinLastPos > self.CabinPos:
            cabin_dir_onehot[0] = 1  # پایین
        else:
            cabin_dir_onehot[1] = 1  # ایستاده

        state.extend(cabin_dir_onehot)

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

        label = font.render(f"Move: {self.TotalMove} , Step :{ self.step}", True, (0,0,0))
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

