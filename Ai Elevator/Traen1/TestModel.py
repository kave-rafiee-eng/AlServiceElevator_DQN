import pygame, torch
from Simulator import ElevatorSimulator, PassengerStatus
import torch.nn as nn

import torch



pygame.init()
sim = ElevatorSimulator()

for i in range(1,10):  # مثلاً 5 مسافر ثابت
    sim.add_passenger(i, 0)

sim.add_passenger(8, 0)
sim.add_passenger(7, 0)

sim.add_passenger(3, 0)
sim.add_passenger(4, 0)
sim.add_passenger(1, 0)

# بارگذاری مدل

# تعریف مدل مشابه مدل آموزش
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)  # لایه سوم مخفی
        self.fc4 = nn.Linear(64, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # فعال‌سازی لایه سوم
        return self.fc4(x)

STATE_SIZE = 30
ACTION_SIZE = 10  # طبق تعداد طبقات آسانسور

model = DQN(STATE_SIZE, ACTION_SIZE)
model.load_state_dict(torch.load("elevator_dqn.pth"))  # فقط وزن‌ها
model.eval()  # حالا درست اجرا میشه



running = True
clock = pygame.time.Clock()
state = sim.get_state()
total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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
        clock.tick(5)

        if done:
            print(f"Test finished, total reward: {total_reward:.2f}")
            # مکث برای دیدن آخرین وضعیت
            running = False

pygame.quit()