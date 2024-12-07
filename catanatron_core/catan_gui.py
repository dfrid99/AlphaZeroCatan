import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1024, 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mini Catan Game")

# Define colors and resources
BACKGROUND = (200, 200, 200)
HEXAGON_COLORS = {
    'forest': (34, 139, 34),
    'fields': (255, 215, 0),
    'mountains': (169, 169, 169),
    'hills': (255, 0, 0),
    'pasture': (144, 238, 144),
    'desert': (210, 180, 140)
}

RESOURCE_MAP = {
    'forest': 'wood',
    'hills': 'brick',
    'mountains': 'ore',
    'fields': 'grain',
    'pasture': 'wool'
}

ROAD_COST = {'wood': 1, 'brick': 1}
SETTLEMENT_COST = {'wood': 1, 'brick': 1, 'grain': 1, 'wool': 1}
CITY_COST = {'ore': 3, 'grain': 2}
# Font setup
FONT = pygame.font.Font(None, 36)
SMALL_FONT = pygame.font.Font(None, 24)

class Hexagon:
    def __init__(self, center, size, resource, number):
        self.center = center
        self.size = size
        self.resource = resource
        self.number = number
        self.vertices = self.calculate_vertices()

    def calculate_vertices(self):
        vertices = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            x = self.center[0] + self.size * math.cos(angle_rad)
            y = self.center[1] + self.size * math.sin(angle_rad)
            vertices.append((int(x), int(y)))
        return vertices

    def draw(self, surface):
        pygame.draw.polygon(surface, HEXAGON_COLORS[self.resource], self.vertices)
        pygame.draw.polygon(surface, (0, 0, 0), self.vertices, 2)  # Black border

        if self.number:
            text = FONT.render(str(self.number), True, (0, 0, 0))
            text_rect = text.get_rect(center=self.center)
            surface.blit(text, text_rect)

class CatanBoard:
    def __init__(self, center, hex_size):
        self.center = center
        self.hex_size = hex_size
        self.hexagons = []
        self.vertices = set()
        self.edges = set()
        self.setup_board()

    def get_adjacent_hexagons(self, vertex):
        return [hexagon for hexagon in self.hexagons if vertex in hexagon.vertices]
    
    def setup_board(self):
        horiz_spacing = self.hex_size * math.sqrt(3)
        vert_spacing = self.hex_size * 1.5

        positions = [
            self.center,
            (self.center[0] - horiz_spacing, self.center[1]),
            (self.center[0] - horiz_spacing/2, self.center[1] - vert_spacing),
            (self.center[0] + horiz_spacing/2, self.center[1] - vert_spacing),
            (self.center[0] + horiz_spacing, self.center[1]),
            (self.center[0] + horiz_spacing/2, self.center[1] + vert_spacing),
            (self.center[0] - horiz_spacing/2, self.center[1] + vert_spacing),
        ]

        resources = ['forest', 'fields', 'mountains', 'hills', 'pasture', 'desert']
        resources = resources * 2
        resources = resources[:7]
        random.shuffle(resources)

        numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        random.shuffle(numbers)

        for i, pos in enumerate(positions):
            resource = resources[i]
            number = numbers[i] if resource != 'desert' else None
            hexagon = Hexagon(pos, self.hex_size, resource, number)
            self.hexagons.append(hexagon)
            self.vertices.update(hexagon.vertices)

        self.generate_edges()

    def generate_edges(self):
        for hexagon in self.hexagons:
            for i in range(6):
                v1 = hexagon.vertices[i]
                v2 = hexagon.vertices[(i + 1) % 6]
                self.edges.add((min(v1, v2), max(v1, v2)))

    def draw(self, surface):
        for hexagon in self.hexagons:
            hexagon.draw(surface)

    def get_nearest_vertex(self, pos):
        return min(self.vertices, key=lambda v: math.hypot(v[0] - pos[0], v[1] - pos[1]))

    def get_nearest_edge(self, pos):
        return min(self.edges, key=lambda e: point_line_distance(pos, e[0], e[1]))

class Button:
    def __init__(self, x, y, width, height, text, color, text_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = FONT.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Player:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.resources = {
            'wood': 0,
            'brick': 0,
            'ore': 0,
            'grain': 0,
            'wool': 0
        }
        self.settlements = []
        self.roads = []
        self.cities = []

    def add_settlement(self, pos):
        self.settlements.append(pos)

    def add_city(self, pos):
        if pos in self.settlements:
            self.settlements.remove(pos)
            self.cities.append(pos)

    def add_road(self, edge):
        self.roads.append(edge)

    def can_afford(self, cost):
        return all(self.resources.get(resource, 0) >= amount for resource, amount in cost.items())

    def pay_cost(self, cost):
        for resource, amount in cost.items():
            self.resources[resource] -= amount

    def add_resources(self, resources):
        for resource, amount in resources.items():
            self.resources[resource] += amount

    def draw_settlements(self, surface):
        for settlement in self.settlements:
            pygame.draw.circle(surface, self.color, settlement, 10)

    def draw_cities(self, surface):
        for city in self.cities:
            pygame.draw.rect(surface, self.color, (city[0] - 10, city[1] - 10, 20, 20))

    def draw_roads(self, surface):
        for road in self.roads:
            pygame.draw.line(surface, self.color, road[0], road[1], 4)
    
    def add_resource(self, resource, amount=1):
        if resource in self.resources:
            self.resources[resource] += amount

    def get_connected_vertices(self):
        connected = set()
        for settlement in self.settlements:
            connected.add(settlement)
        for road in self.roads:
            connected.add(road[0])
            connected.add(road[1])
        return connected

    def add_resource(self, resource, amount=1):
        if resource in self.resources:
            self.resources[resource] += amount

    def draw_resources(self, surface, x, y):
        pygame.draw.rect(surface, (255, 255, 255), (x, y, 180, 150))
        text = SMALL_FONT.render(f"{self.name}'s Resources", True, (0, 0, 0))
        surface.blit(text, (x + 5, y + 5))
        
        y_offset = 30
        for resource, amount in self.resources.items():
            text = SMALL_FONT.render(f"{resource}: {amount}", True, (0, 0, 0))
            surface.blit(text, (x + 5, y + y_offset))
            y_offset += 20

    def add_settlement(self, pos):
        self.settlements.append(pos)

    def add_road(self, edge):
        self.roads.append(edge)

    def draw_settlements(self, surface):
        for settlement in self.settlements:
            pygame.draw.circle(surface, self.color, settlement, 10)

    def draw_roads(self, surface):
        for road in self.roads:
            pygame.draw.line(surface, self.color, road[0], road[1], 4)

    def can_trade_with_bank(self):
        return any(amount >= 4 for amount in self.resources.values())

    def trade_with_bank(self, give_resource, receive_resource):
        if self.resources[give_resource] >= 4:
            self.resources[give_resource] -= 4
            self.resources[receive_resource] += 1
            return True
        return False

class BankTradeMenu:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.give_resource = None
        self.receive_resource = None
        self.resources = ['wood', 'brick', 'ore', 'grain', 'wool']
        self.button_rects = []

    def draw(self, surface):
        pygame.draw.rect(surface, (200, 200, 200), self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        
        font = pygame.font.Font(None, 24)
        title = font.render("Bank Trade (4:1)", True, (0, 0, 0))
        surface.blit(title, (self.rect.x + 10, self.rect.y + 10))

        y_offset = 40
        self.button_rects = []
        for resource in self.resources:
            give_rect = pygame.Rect(self.rect.x + 10, self.rect.y + y_offset, 100, 30)
            receive_rect = pygame.Rect(self.rect.x + 120, self.rect.y + y_offset, 100, 30)
            
            pygame.draw.rect(surface, (150, 150, 150), give_rect)
            pygame.draw.rect(surface, (150, 150, 150), receive_rect)
            
            give_text = font.render(f"Give {resource}", True, (0, 0, 0))
            receive_text = font.render(f"Receive {resource}", True, (0, 0, 0))
            
            surface.blit(give_text, (give_rect.x + 5, give_rect.y + 5))
            surface.blit(receive_text, (receive_rect.x + 5, receive_rect.y + 5))
            
            self.button_rects.append((give_rect, f"give_{resource}"))
            self.button_rects.append((receive_rect, f"receive_{resource}"))
            
            y_offset += 40

        trade_button = pygame.Rect(self.rect.x + 10, self.rect.y + y_offset, 210, 30)
        pygame.draw.rect(surface, (0, 255, 0), trade_button)
        trade_text = font.render("Trade", True, (0, 0, 0))
        surface.blit(trade_text, (trade_button.x + 85, trade_button.y + 5))
        self.button_rects.append((trade_button, "trade"))

    def handle_click(self, pos):
        for rect, action in self.button_rects:
            if rect.collidepoint(pos):
                if action.startswith("give_"):
                    self.give_resource = action.split("_")[1]
                elif action.startswith("receive_"):
                    self.receive_resource = action.split("_")[1]
                elif action == "trade":
                    return self.give_resource, self.receive_resource
        return None
    
def point_line_distance(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
    den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    
    return num / den if den != 0 else 0


def main():
    clock = pygame.time.Clock()
    board = CatanBoard((WIDTH // 2, HEIGHT // 2), 60)
    
    players = [Player("Red", (255, 0, 0)), Player("Blue", (0, 0, 255))]
    current_player = 0
    game_phase = "setup"
    setup_turns = 0
    setup_stage = "settlement"
    
    roll_button = Button(WIDTH - 280, HEIGHT - 60, 120, 50, "Roll Dice", (0, 255, 0), (0, 0, 0))
    end_turn_button = Button(WIDTH - 150, HEIGHT - 60, 120, 50, "End Turn", (255, 165, 0), (0, 0, 0))
    dice_result = None
    dice_rolled = False

    def valid_road_placement(player, edge):
        connected_vertices = player.get_connected_vertices()
        is_valid = edge[0] in connected_vertices or edge[1] in connected_vertices
        
        print(f"Checking road placement for {player.name}:")
        print(f"  Edge: {edge}")
        print(f"  Connected vertices: {connected_vertices}")
        print(f"  Is valid: {is_valid}")
        
        return is_valid
    
    def get_nearest_valid_edge(player, pos):
        valid_edges = [edge for edge in board.edges if valid_road_placement(player, edge)]
        if not valid_edges:
            print(f"No valid edges found for {player.name}")
            return None
        return min(valid_edges, key=lambda e: point_line_distance(pos, e[0], e[1]))

    def distribute_resources(roll):
        for hexagon in board.hexagons:
            if hexagon.number == roll and hexagon.resource != 'desert':
                for player in players:
                    for settlement in player.settlements:
                        if settlement in hexagon.vertices:
                            player_resource = RESOURCE_MAP[hexagon.resource]
                            player.add_resource(player_resource)
                            print(f"{player.name} received 1 {player_resource}")  # Debug print

    buy_road_button = Button(10, HEIGHT - 60, 120, 50, "Buy Road", (100, 100, 100), (255, 255, 255))
    buy_settlement_button = Button(140, HEIGHT - 60, 120, 50, "Buy Settlement", (100, 100, 100), (255, 255, 255))
    buy_city_button = Button(270, HEIGHT - 60, 120, 50, "Buy City", (100, 100, 100), (255, 255, 255))
    bank_trade_button = Button(400, HEIGHT - 60, 120, 50, "Bank Trade", (100, 100, 100), (255, 255, 255))

    building_type = None    
    bank_trade_menu = None
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_phase == "setup":
                    if setup_stage == "settlement":
                        vertex = board.get_nearest_vertex(event.pos)
                        players[current_player].add_settlement(vertex)
                        print(f"{players[current_player].name} placed a settlement at {vertex}")
                        setup_stage = "road"
                    elif setup_stage == "road":
                        nearest_edge = get_nearest_valid_edge(players[current_player], event.pos)
                        if nearest_edge:
                            players[current_player].add_road(nearest_edge)
                            print(f"{players[current_player].name} placed a road at {nearest_edge}")
                            
                            setup_turns += 1
                            if setup_turns < 2:
                                current_player = 1  # Switch to Player 2 after Player 1's first turn
                            elif setup_turns == 2:
                                current_player = 1  # Player 2 goes again
                            elif setup_turns == 3:
                                current_player = 0  # Back to Player 1 for the last turn
                            
                            if setup_turns == 4:
                                game_phase = "play"
                                current_player = 0  # Start with Player 1 in play phase
                            else:
                                setup_stage = "settlement"
                            
                            print(f"Setup Turn: {setup_turns}, Current Player: {players[current_player].name}")
                        else:
                            print(f"Invalid road placement for {players[current_player].name}")
                elif game_phase == "play":
                    if bank_trade_menu:
                        trade_result = bank_trade_menu.handle_click(event.pos)
                        if trade_result:
                            give_resource, receive_resource = trade_result
                            if players[current_player].trade_with_bank(give_resource, receive_resource):
                                print(f"{players[current_player].name} traded 4 {give_resource} for 1 {receive_resource}")
                            bank_trade_menu = None
                    elif roll_button.is_clicked(event.pos) and not dice_rolled:
                        dice_result = random.randint(1, 6) + random.randint(1, 6)
                        dice_rolled = True
                        distribute_resources(dice_result)
                        print(f"Dice roll: {dice_result}")
                        for player in players:
                            print(f"{player.name} resources: {player.resources}")
                    elif end_turn_button.is_clicked(event.pos) and dice_rolled:
                        current_player = (current_player + 1) % 2
                        dice_rolled = False
                        dice_result = None
                        building_type = None
                    elif buy_road_button.is_clicked(event.pos) and dice_rolled:
                        if players[current_player].can_afford(ROAD_COST):
                            building_type = "road"
                    elif buy_settlement_button.is_clicked(event.pos) and dice_rolled:
                        if players[current_player].can_afford(SETTLEMENT_COST):
                            building_type = "settlement"
                    elif buy_city_button.is_clicked(event.pos) and dice_rolled:
                        if players[current_player].can_afford(CITY_COST):
                            building_type = "city"
                    elif bank_trade_button.is_clicked(event.pos) and dice_rolled:
                        if players[current_player].can_trade_with_bank():
                            bank_trade_menu = BankTradeMenu(WIDTH // 2 - 150, HEIGHT // 2 - 150, 300, 300)

        screen.fill(BACKGROUND)
        board.draw(screen)

        # Draw player resources with adjusted positions
        players[0].draw_resources(screen, 10, 10)  # Player 1's resources on the left
        players[1].draw_resources(screen, WIDTH - 190, 10)  # Player 2's resources on the right

        for player in players:
            player.draw_settlements(screen)
            player.draw_cities(screen)
            player.draw_roads(screen)

        if game_phase == "setup":
            action = "place a settlement" if setup_stage == "settlement" else "place a road"
            text = FONT.render(f"{players[current_player].name}, {action}", True, (0, 0, 0))
            screen.blit(text, (WIDTH // 2 - 150, HEIGHT - 40))
        else:
            text = FONT.render(f"{players[current_player].name}'s Turn", True, (0, 0, 0))
            screen.blit(text, (WIDTH // 2 - 70, 10))
            
            if not dice_rolled:
                roll_button.draw(screen)
            else:
                end_turn_button.draw(screen)
                buy_road_button.draw(screen)
                buy_settlement_button.draw(screen)
                buy_city_button.draw(screen)
                bank_trade_button.draw(screen)
            
            if dice_result:
                dice_text = FONT.render(f"Dice Roll: {dice_result}", True, (0, 0, 0))
                screen.blit(dice_text, (WIDTH // 2 - 50, HEIGHT - 100))

            if building_type:
                instruction_text = FONT.render(f"Click to place {building_type}", True, (0, 0, 0))
                screen.blit(instruction_text, (WIDTH // 2 - 100, HEIGHT - 150))

            if bank_trade_menu:
                bank_trade_menu.draw(screen)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()