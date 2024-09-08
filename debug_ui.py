import pygame
from gomoku import Checkerboard
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = 600
GRID_SIZE = 15
CELL_SIZE = SCREEN_SIZE // (GRID_SIZE - 1)
LINE_COLOR = (0, 0, 0)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
BG_COLOR = (205, 133, 63)
BOLD_DOT_COLOR = (0, 0, 0)
BOLD_DOT_RADIUS = 5

# Button dimensions and colors
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + BUTTON_HEIGHT + 10))
pygame.display.set_caption("Gomoku")

# Create the checkerboard
board = Checkerboard(GRID_SIZE, GRID_SIZE, 3)

def draw_board():
    screen.fill(BG_COLOR)
    for x in range(GRID_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, SCREEN_SIZE))
        pygame.draw.line(screen, LINE_COLOR, (0, x * CELL_SIZE), (SCREEN_SIZE, x * CELL_SIZE))

    # Draw bold dots
    bold_dots = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
    for (x, y) in bold_dots:
        pygame.draw.circle(screen, BOLD_DOT_COLOR, (x * CELL_SIZE, y * CELL_SIZE), BOLD_DOT_RADIUS)

    # Draw stones
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if board.board[y][x] == board.black:
                pygame.draw.circle(screen, BLACK_COLOR, (x * CELL_SIZE, y * CELL_SIZE), CELL_SIZE // 2 - 2)
            elif board.board[y][x] == board.white:
                pygame.draw.circle(screen, WHITE_COLOR, (x * CELL_SIZE, y * CELL_SIZE), CELL_SIZE // 2 - 2)

def draw_button(x, y, width, height, text, hover=False):
    button_color = BUTTON_HOVER_COLOR if hover else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, (x, y, width, height))
    font = pygame.font.Font(None, 36)
    text_surf = font.render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surf, text_rect)

def main():
    running = True
    game_over = False
    reset_button_rect = pygame.Rect((SCREEN_SIZE // 2 - BUTTON_WIDTH // 2, SCREEN_SIZE + 10), (BUTTON_WIDTH, BUTTON_HEIGHT))

    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_hover = reset_button_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if reset_button_rect.collidepoint(event.pos):
                    board.reset()
                    game_over = False
                    print("The board has been reset!")
                elif not game_over:
                    mouse_x, mouse_y = event.pos
                    if mouse_y < SCREEN_SIZE:  # Ensure clicks are within the board area
                        grid_x = round(mouse_x / CELL_SIZE)
                        grid_y = round(mouse_y / CELL_SIZE)

                        if board.board[grid_y][grid_x] == 0:
                            result, winner = board.step(grid_y, grid_x)
                            print(board.get_state()[0])
                            if result:
                                game_over = True
                                print(f"Player {'Black' if winner == board.black else 'White'} wins!")

        draw_board()
        draw_button(reset_button_rect.x, reset_button_rect.y, reset_button_rect.width, reset_button_rect.height, "Reset", mouse_hover)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

# Running the game
# We cannot execute the Pygame window in this environment. This code is intended to be run locally.
main()
