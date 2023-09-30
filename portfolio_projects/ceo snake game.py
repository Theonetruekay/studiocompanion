import pygame
import time
import random

pygame.init()

# Screen dimensions
width, height = 400, 300
win = pygame.display.set_mode((width, height))

# Initialize game variables
snake_block = 10
snake_speed = 30
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 50)
small_font = pygame.font.SysFont(None, 20)

def snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(win, (0, 0, 128), [x[0], x[1], snake_block, snake_block])
    ceo_label = small_font.render("CEO", True, (255, 255, 255))
    win.blit(ceo_label, (snake_list[-1][0], snake_list[-1][1] - 10))

def gameLoop():
    game_over = False

    # Snake initial position
    x1, y1 = width / 2, height / 2
    dx, dy = 0, 0

    snake_list = []
    length_of_snake = 1

    # Food position
    foodx = round(random.randrange(0, width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, height - snake_block) / 10.0) * 10.0

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and dx == 0:
                    dx = -snake_block
                    dy = 0
                elif event.key == pygame.K_RIGHT and dx == 0:
                    dx = snake_block
                    dy = 0
                elif event.key == pygame.K_UP and dy == 0:
                    dy = -snake_block
                    dx = 0
                elif event.key == pygame.K_DOWN and dy == 0:
                    dy = snake_block
                    dx = 0

        x1 += dx
        y1 += dy

        # Infinite walls logic
        if x1 >= width:
            x1 = 0
        if x1 < 0:
            x1 = width
        if y1 >= height:
            y1 = 0
        if y1 < 0:
            y1 = height

        win.fill((0, 0, 0))
        food = small_font.render("$", True, (255, 0, 0))
        win.blit(food, (foodx, foody))
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)

        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_head:
                game_over = True

        snake(snake_block, snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, height - snake_block) / 10.0) * 10.0
            length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

gameLoop()
