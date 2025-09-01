import pygame, torch
from Simulator import ElevatorSimulator, PassengerStatus
import torch.nn as nn
import random
import torch



def add_random_passengers(sim, num_passengers):
    for _ in range(num_passengers):
        start = random.randint(0, sim.NumOfFloor - 1)   # اصلاح شد
        dest = random.randint(0, sim.NumOfFloor - 1)
        while dest == start:
            dest = random.randint(0, sim.NumOfFloor - 1)
        sim.add_passenger(start, dest)

def reset_simulator():
    sim = ElevatorSimulator()

    #-----------------------
    #add_random_passengers( sim , random.randint(1, 8))

    #-----------------------
    #for i in range( 1, 9 ): 
       #sim.add_passenger(i, 0)
    
    #-----------------------
    #sim.add_passenger(8, 0)

    #-----------------------
    passenger_scenario = []
    num_passengers = random.randint(1, 8)

    for _ in range(num_passengers):
        origin = random.randint(1, sim.NumOfFloor - 1)  # مبدا نمی‌تونه 0 باشه
        destination = random.randint(0, origin - 1)          # مقصد حتماً کوچکتر از مبدا
        passenger_scenario.append((origin, destination))
    
    for origin, destination in passenger_scenario:
        sim.add_passenger(origin , destination )
    return sim

pygame.init()
sim = reset_simulator()


# بارگذاری مدل

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # استفاده از Sequential برای تعریف لایه‌ها
 
        self.FC = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_size)
        )
        

        # اختیاری: مقداردهی اولیه وزن‌ها با He Initialization
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        return self.FC(x)
    
"""
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
"""

        
STATE_SIZE = 33
ACTION_SIZE = 10  # طبق تعداد طبقات آسانسور

model = DQN(STATE_SIZE, ACTION_SIZE)
model.load_state_dict(torch.load("Outputs/final_weights_3000.pth"))  # فقط وزن‌ها
model.eval()  # حالا درست اجرا میشه



running = True
clock = pygame.time.Clock()
state = sim.get_state()
total_reward = 0

def RunModel( First = False ) :
    
    global state, total_reward

    if First :
        sim.screen.fill((230,230,230))
        sim.render()
        pygame.display.flip()
    else :
        done = all(p["status"] == PassengerStatus.ARRIVED for p in sim.passengers)
        if not done :
            # انتخاب اقدام با استفاده از مدل
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()

            last_pos = sim.CabinPos
            sim.move_to_floor(action)
            reward = sim.compute_reward(last_pos)
            total_reward += reward
            state = sim.get_state()

            # رسم شبیه‌ساز
            sim.screen.fill((230,230,230))
            sim.render()
            pygame.display.flip()
            clock.tick(2)

            if done:
                print(f"Test finished, total reward: {total_reward:.2f}")
                # مکث برای دیدن آخرین وضعیت
                running = False

Reset = True
while running:

    if Reset :
        Reset = False
        RunModel( First=1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN  :
            if pygame.K_UP == event.key :
                sim = reset_simulator()
                Reset = True

            elif pygame.K_DOWN == event.key  :
                RunModel()

pygame.quit()