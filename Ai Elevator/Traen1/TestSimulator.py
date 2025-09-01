import pygame, sys
from Simulator import ElevatorSimulator, PassengerStatus

pygame.init()
sim = ElevatorSimulator()
#for i in range(1, 5):  # شروع ساده‌تر: ۴ مسافر
   # sim.add_passenger(i, 0)


for i in range(1, 9):  # مثلاً 5 مسافر ثابت
    sim.add_passenger(i, 0)

#sim.add_passenger(5, 0)


running = True
clock = pygame.time.Clock()
max_steps_per_episode = 50

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # ---------- دریافت ورودی کاربر ----------
        if event.type == pygame.KEYDOWN:
            if pygame.K_0 <= event.key <= pygame.K_9:   # اعداد
                num = event.key - pygame.K_0
                if num < sim.NumOfFloor:
                    last_pos = sim.CabinPos
                    sim.move_to_floor(num)
                    sim.compute_reward(last_pos)
                    for i, p in enumerate(sim.passengers):
                        print(f"Passenger {i+1}: status={p['status']}, wait_time={p.get('Cabin_WT',0)}")

                    #print(f"Moved to floor {num}, Reward: {reward:.2f}")

    # ---------- بررسی پایان ----------
    done = all(p["status"] == PassengerStatus.ARRIVED for p in sim.passengers)
    if done:
        print("All passengers arrived!")

    # ---------- رسم ----------
    sim.screen.fill((230,230,230))
    sim.render()
    
    pygame.display.flip()

