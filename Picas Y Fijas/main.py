import random
from itertools import permutations
from typing import override


### --------- UTILS --------- ###

def generate_secret():
    """Genera un numero secreto de 4 dígitos no repetidos."""
    digits = random.sample("0123456789", 4)
    return ''.join(digits)

def generate_all_numbers():
    """Genera todos los posibles números 4 dígitos no repetidos."""
    return [''.join(p) for p in permutations("0123456789", 4)]

def validate_guess(guess: str) -> bool:
    """Valída que la conjetura sea de 4 dígitos únicos."""
    return len(guess) == 4 and len(set(guess)) == 4 and guess.isdigit()

def calculate_picas_fijas(candidate: str, guess: str) -> tuple[int, int]:
    """Calcula picas y fijas entre un candidato y una conjetura."""
    picas = sum((d in candidate) and (guess[i] != candidate[i]) for i, d in enumerate(guess))
    fijas = sum(guess[i] == candidate[i] for i in range(4))
    return picas, fijas

def validate_winner(perception: str):
    if ',' in perception:
        picas, fijas = perception.split(',')
        return fijas == '4'
    return False

### --------- CLASES --------- ###

class Agent:

    def __init__(self):
        self.number = generate_secret()  # Número secreto del agente

    def compute(self, perception: str) -> str:
        """Procesa una percepción y retorna una acción."""

        if perception in "BN":
            return 'L'

        if perception == "L":
            return self.send_guess()

        elif perception.isdigit() and len(perception) == 4:
            # Es un número adivinado por el oponente, calcular picas y fijas
            picas, fijas = calculate_picas_fijas(self.number, perception)
            return f"{picas},{fijas}"

        elif ',' in perception:
            # Es feedback (picas, fijas) del oponente
            picas, fijas = map(int, perception.split(','))
            self.process_feedback(picas, fijas)
            return 'L'

        else:
            raise ValueError(f"Percepción inválida: {perception}")

    def send_guess(self) -> str:
        pass

    def process_feedback(self, picas: int, fijas: int):
        pass

class AgentImp(Agent):
    def __init__(self):
        super().__init__()
        self.last_guess = None  # Última conjetura hecha por este agente
        self.possible_numbers = generate_all_numbers()  # Posibles números del oponente


    @override
    def send_guess(self) -> str:
        """Envía una conjetura (número de 4 dígitos)."""
        if not self.possible_numbers:
            raise ValueError("No hay números posibles restantes")
        self.last_guess = random.choice(self.possible_numbers)  # Podría optimizarse
        return self.last_guess

    @override
    def process_feedback(self, picas: int, fijas: int) -> None:
        """Filtra los números posibles según el feedback."""
        new_possible = []
        for num in self.possible_numbers:
            computed_p, computed_f = calculate_picas_fijas(num, self.last_guess)
            if computed_p == picas and computed_f == fijas:
                new_possible.append(num)
        self.possible_numbers = new_possible


class Environment:
    def __init__(self, agent_b, agent_n):
        self.agent_b = agent_b  # Agente Blanco (empieza primero)
        self.agent_n = agent_n  # Agente Negro
        self.current_turn = 'B'  # Empieza Blanco
        self.max_turns = 20
        self.winner = None
        self.initialized = False
        self.action = "B"

    def play_turn(self) -> bool:
        """Ejecuta un turno y retorna True si el juego terminó."""

        last_turn = self.current_turn

        if not self.initialized:
            self.action = agent_b.compute('B')
            self.action = agent_n.compute('N')
            self.initialized = True

        elif self.current_turn == 'B':
            self.action = agent_b.compute(self.action)
            self.current_turn = 'N'

        elif self.current_turn == 'N':
            self.action = agent_n.compute(self.action)
            self.current_turn = 'B'
        if self.action != 'L':
            print(f"\n--- Turno ({last_turn}) ---")
            print(self.action)
        return validate_winner(self.action)

    def run_game(self) -> str:
        """Ejecuta el juego hasta que haya un ganador o se alcance el máximo de turnos."""
        game_over = False
        while not game_over:
            game_over = self.play_turn()
            if game_over:
                self.winner = self.current_turn
                print(f"¡Agente {self.winner} ha ganado!")
                return self.winner
        print("¡Empate! Nadie adivinó el número.")
        return "Draw"

class DummyAgent(Agent):

    @override
    def send_guess(self):
        return input("Ingresa un número de 4 dígitos: ")

    @override
    def process_feedback(self, picas, fijas):
        pass

# Ejemplo de uso:
if __name__ == "__main__":
    agent_b = AgentImp()  # Agente Blanco (inicia primero)
    agent_n = AgentImp()  # Agente Negro
    print(f"Dummy number: {agent_b.number}")
    print(f"Agent number: {agent_n.number}")
    env = Environment(agent_b, agent_n)
    winner = env.run_game()
    print(f"El ganador es: {winner}")