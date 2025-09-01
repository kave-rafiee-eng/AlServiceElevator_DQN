import pygame, sys, random, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from Simulator import ElevatorSimulator, PassengerStatus
import random

MODEL_PATH = "elevator_dqn.pth"

# انتخاب device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- DQN ----------
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

# ---------- تنظیمات ----------
STATE_SIZE = 30                # دقت کن طول state با این یکی باشد
ACTION_SIZE = ElevatorSimulator().NumOfFloor

model = DQN(STATE_SIZE, ACTION_SIZE).to(device)

model.load_state_dict(torch.load( MODEL_PATH , map_location=device))

target_model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ---------- 
MODEL_MEMORY_SIZE = 200
models_memory = deque(maxlen=MODEL_MEMORY_SIZE)  # نگهداری آخرین 200 مدل به همراه reward و episode


buffer = deque(maxlen=10000)
BATCH_SIZE = 256
GAMMA = 0.99

# اپسیلون خطی
EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS = 1.0, 0.05, 30000
def get_epsilon(step):
    frac = max(0.0, 1 - step / EPSILON_DECAY_STEPS)
    return EPSILON_END + (EPSILON_START - EPSILON_END) * frac


def add_random_passengers(sim, num_passengers):
    for _ in range(num_passengers):
        start = random.randint(0, sim.NumOfFloor - 1)   # اصلاح شد
        dest = random.randint(0, sim.NumOfFloor - 1)
        while dest == start:
            dest = random.randint(0, sim.NumOfFloor - 1)
        sim.add_passenger(start, dest)


def reset_simulator(current_episode):
    sim = ElevatorSimulator()

    for i in range(1, 9):  # مثلاً 5 مسافر ثابت
       sim.add_passenger(i, 0)

    """
    if current_episode < 500:
        # --- مرحله اول: سناریوی ساده و تکراری ---
        for i in range(1, 6):  # مثلاً 5 مسافر ثابت
            sim.add_passenger(i, 0)

    elif current_episode < 1500:
        # --- مرحله دوم: ترکیبی (بیشتر ثابت) ---
        if random.random() < 0.7:
            for i in range(1, 6):
                sim.add_passenger(i, 0)
        else:
            add_random_passengers( sim , random.randint(3, 8))

    elif current_episode < 3000:
        # --- مرحله سوم: ترکیبی (بیشتر رندوم) ---
        if random.random() < 0.3:
            for i in range(1, 6):
                sim.add_passenger(i, 0)
        else:
            add_random_passengers( sim , random.randint(3, 8))

    else:
        # --- مرحله چهارم: کاملاً رندوم ---
        add_random_passengers( sim , random.randint(3, 8))

    """
    return sim


# ---------- شبیه‌ساز ----------
pygame.init()
sim = reset_simulator(0)

TARGET_UPDATE_STEPS = 200
REPLAY_WARMUP = 200
global_step = 0

# اپیزودها
MAX_EPISODES = 10000
max_steps_per_episode = 300
episode_rewards = []




state = sim.get_state()
total_reward = 0
step_count = 0
current_episode = 1
running = True

# ---------- حلقه اصلی ----------
while running and current_episode <= MAX_EPISODES:
    global_step += 1
    step_count += 1
    EPSILON = get_epsilon(global_step)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # انتخاب اکشن
    if random.random() < EPSILON:
        action = random.randint(0, sim.NumOfFloor - 1)
    else:
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32).to(device))
            max_q = q_values.max().item()
            candidates = (q_values == max_q).nonzero(as_tuple=False).view(-1).tolist()
            action = random.choice(candidates)

    last_pos = sim.CabinPos
    sim.move_to_floor(action)

    reward = sim.compute_reward(last_pos)


    next_state = sim.get_state()
    done = all(p["status"] == PassengerStatus.ARRIVED for p in sim.passengers)

    # ذخیره تجربه
    buffer.append((state, action, reward, next_state, done))
    state = next_state
    total_reward += reward

    # آموزش
    if len(buffer) >= max(BATCH_SIZE, REPLAY_WARMUP):
        batch = random.sample(buffer, BATCH_SIZE)
        states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

        '''
        states_b      = torch.tensor(states_b, dtype=torch.float32).to(device)
        actions_b     = torch.tensor(actions_b, dtype=torch.long).to(device)
        rewards_b     = torch.tensor(rewards_b, dtype=torch.float32).to(device)
        next_states_b = torch.tensor(next_states_b, dtype=torch.float32).to(device)
        dones_b       = torch.tensor(dones_b, dtype=torch.float32).to(device)
        '''

        states_b = torch.tensor(np.array(states_b), dtype=torch.float32).to(device)
        actions_b = torch.tensor(np.array(actions_b), dtype=torch.long).to(device)
        rewards_b = torch.tensor(np.array(rewards_b), dtype=torch.float32).to(device)
        next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32).to(device)
        dones_b = torch.tensor(np.array(dones_b), dtype=torch.float32).to(device)

        q_values_b = model(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = model(next_states_b).argmax(dim=1)
            next_q_values = target_model(next_states_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_b + GAMMA * next_q_values * (1 - dones_b)

        loss = criterion(q_values_b, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if global_step % TARGET_UPDATE_STEPS == 0:
            target_model.load_state_dict(model.state_dict())

    # پایان اپیزود
    if done or step_count >= max_steps_per_episode:
        print(f"Episode {current_episode} | reward {total_reward:.2f} | eps {EPSILON:.3f}")
        episode_rewards.append(total_reward)

        state_dict_clone = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        models_memory.append({
            "episode": current_episode,
            "reward": total_reward,
            "state_dict": state_dict_clone
        })

        current_episode += 1

        # ریست
        sim = reset_simulator(current_episode)

       # for i, p in enumerate(sim.passengers):
           # print(f"Passenger {i+1}: status={p['status']}, or={p.get('origin',0)}")

        state = sim.get_state()
        total_reward = 0
        step_count = 0

# ---------- ذخیره ----------

#torch.save(model.state_dict(), MODEL_PATH)
#print(f"Model saved to {MODEL_PATH}")

# ---------- انتخاب بهترین مدل از آخرین 200 مدل و ذخیره آن ----------
if len(models_memory) > 0:
    # پیدا کردن ایتم با بیشترین reward
    best_item = max(models_memory, key=lambda x: x["reward"])
    best_episode = best_item["episode"]
    best_reward = best_item["reward"]
    best_state_dict = best_item["state_dict"]

    # ذخیره بهترین مدل
    BEST_MODEL_PATH = "elevator_dqn.pth"
    # توجه: state_dict قبلاً کلون شده؛ مستقیم ذخیره می‌کنیم
    torch.save(best_state_dict, BEST_MODEL_PATH)
    print(f"Best model from episode {best_episode} with reward {best_reward:.2f} saved to {BEST_MODEL_PATH}")
else:
    print("No models in memory to select best from.")

# ---------- رسم نمودار ----------
plt.figure(figsize=(10,5))

# فقط 10 درصد آخر
n = len(episode_rewards)
last_part = max(1, int(n * 0.1))  # تعداد داده‌ها (حداقل 1)

plt.plot(
    range(n - last_part + 1, n + 1),
    episode_rewards[-last_part:],
    marker='o'
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.show()
