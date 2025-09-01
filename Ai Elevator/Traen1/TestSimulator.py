import pygame, sys
from Simulator import ElevatorSimulator, PassengerStatus
import random

pygame.init()

def add_random_passengers(sim, num_passengers):
    for _ in range(num_passengers):
        start = random.randint(0, sim.NumOfFloor - 1)   # اصلاح شد
        dest = random.randint(0, sim.NumOfFloor - 1)
        while dest == start:
            dest = random.randint(0, sim.NumOfFloor - 1)
        sim.add_passenger(start, dest)

def reset_simulator():
    sim = ElevatorSimulator()

    passenger_scenario = []

    num_passengers = random.randint(1, 8)

    for _ in range(num_passengers):
        origin = random.randint(1, sim.NumOfFloor - 1)  # مبدا نمی‌تونه 0 باشه
        destination = random.randint(0, origin - 1)          # مقصد حتماً کوچکتر از مبدا
        passenger_scenario.append((origin, destination))
    
    for origin, destination in passenger_scenario:
        sim.add_passenger(origin , destination )
        
    #add_random_passengers( sim , random.randint(1, 8))
    #for i in range(1, 9):  # مثلاً 5 مسافر ثابت
    #    sim.add_passenger(i, 0)

    #sim.add_passenger(5, 0)

    return sim

sim = reset_simulator()

running = True
clock = pygame.time.Clock()
max_steps_per_episode = 50

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # ---------- دریافت ورودی کاربر ----------
        
        if event.type == pygame.KEYDOWN:
            if pygame.K_UP == event.key :
                sim = reset_simulator()

            if pygame.K_0 <= event.key <= pygame.K_9:   # اعداد
                num = event.key - pygame.K_0
                if num < sim.NumOfFloor:
                    last_pos = sim.CabinPos
                    sim.move_to_floor(num)
                    sim.compute_reward(last_pos)
                    state = sim.get_state()
                    print(state)
                    for i, p in enumerate(sim.passengers):
                        print(f"Passenger {i+1}: status={p['status']}, wait_time={p.get('Cabin_WT',0)}")

                    #print(f"Moved to floor {num}, Reward: {reward:.2f}")

    # ---------- بررسی پایان ----------
    done = all(p["status"] == PassengerStatus.ARRIVED for p in sim.passengers)
    if done:
        pass
        #print("All passengers arrived!")

    # ---------- رسم ----------
    sim.screen.fill((230,230,230))
    sim.render()
    
    pygame.display.flip()

