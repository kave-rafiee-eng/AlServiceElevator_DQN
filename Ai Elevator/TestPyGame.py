import pygame
import sys

# تنظیمات اولیه
pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# آسانسور
elevator_width, elevator_height = 60, 60
elevator_x = WIDTH // 2 - elevator_width // 2
elevator_y = HEIGHT - elevator_height
speed = 5

# 10 طبقه
floors = 10
floor_height = HEIGHT // floors

# تعریف رویداد تایمر
MOVE_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(MOVE_EVENT, 1000)  # هر 1000ms = یک ثانیه

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOVE_EVENT:
            elevator_y -= 60   # هر ثانیه یک طبقه بالا (اندازه آسانسور = 60px)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and elevator_y > 0:
        elevator_y -= speed
    if keys[pygame.K_DOWN] and elevator_y < HEIGHT - elevator_height:
        elevator_y += speed

    # رسم صفحه
    screen.fill((200, 200, 200))

    # رسم خطوط طبقات
    for i in range(floors):
        pygame.draw.line(screen, (0, 0, 0), (0, i * floor_height), (WIDTH, i * floor_height), 2)

    # رسم آسانسور
    pygame.draw.rect(screen, (100, 100, 255), (elevator_x, elevator_y, elevator_width, elevator_height))

    pygame.display.flip()
    clock.tick(30)