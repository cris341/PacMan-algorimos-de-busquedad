"""
Pac-Man para algoritmos de búsqueda.

  - El mapa es una lista de strings dentro del código.
  - Pac-Man lo controla el jugador.
  - Cada fantasma persigue a Pac-Man con un algoritmo distinto.
    Activos ahora: A*, BCU, BFS, Greedy y DFS. Para sumar otro, agrégalo en ALGORITMOS.
  - Los fantasmas solo replanifican si Pac-Man cambió de casilla desde la última búsqueda.
  - Capas de visualización para el camino y los nodos expandidos.

Requiere:  pip install pygame
Ejecutar:  python pacman_visual.py
"""
import heapq
import math
import random
import sys
from collections import deque, namedtuple

import pygame

# =====================================================================
# MAPA
# =====================================================================
MAPA = [
    "###################",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#o##..~~...~~..##o#",
    "#....##.#.#.##....#",
    "####.####D####.####",
    "T.....#12345#.....T",
    "####.#########.####",
    "#....##.#.#.##....#",
    "#o##.....P.....##o#",
    "#.##.###.#.###.##.#",
    "#........#........#",
    "###################",
]

assert all(len(f) == len(MAPA[0]) for f in MAPA), "Todas las filas deben medir lo mismo"

COSTOS = {'.': 1, 'o': 1, '~': 3, 'T': 1, 'P': 1, 'D': 1,
          '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

FILAS, COLS = len(MAPA), len(MAPA[0])

# =====================================================================
# GRAFO
# =====================================================================
def construir_grafo(mapa, es_fantasma=False):
    """Devuelve {celda: {vecino: costo_de_entrar}}."""
    filas, cols = len(mapa), len(mapa[0])

    def transitable(i, j):
        c = mapa[i][j]
        if c == '#':
            return False
        if c == 'D' and not es_fantasma:
            return False
        return True

    # Túneles: la 'T' del borde izquierdo conecta con la del borde derecho
    tuneles = {}
    for i in range(filas):
        if mapa[i][0] == 'T' and mapa[i][cols - 1] == 'T':
            tuneles[(i, 0)] = (i, cols - 1)
            tuneles[(i, cols - 1)] = (i, 0)

    grafo = {}
    for i in range(filas):
        for j in range(cols):
            if not transitable(i, j):
                continue
            vecinos = {}
            for di, dj in DIRS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < filas and 0 <= nj < cols):
                    if (i, j) in tuneles and di == 0:
                        ni, nj = tuneles[(i, j)]
                    else:
                        continue
                if transitable(ni, nj):
                    vecinos[(ni, nj)] = COSTOS[mapa[ni][nj]]
            grafo[(i, j)] = vecinos
    return grafo


def buscar_posiciones(mapa):
    """Encuentra la posición de Pac-Man ('P') y de cada fantasma ('1'-'5')."""
    pos = {}
    for i, fila in enumerate(mapa):
        for j, c in enumerate(fila):
            if c in 'P12345':
                pos[c] = (i, j)
    return pos


def vecino_en_direccion(grafo, pos, d):
    """Celda vecina de `pos` en la dirección `d` (considera el túnel), o None."""
    ni, nj = pos[0] + d[0], pos[1] + d[1]
    for cand in ((ni, nj), (ni, nj % COLS)):
        if cand in grafo[pos]:
            return cand
    return None


def celdas_alcanzables(grafo, inicio):
    """Validación del mapa: celdas a las que se puede llegar desde `inicio`."""
    vistas, pendientes = {inicio}, [inicio]
    while pendientes:
        actual = pendientes.pop()
        for v in grafo[actual]:
            if v not in vistas:
                vistas.add(v)
                pendientes.append(v)
    return vistas


# =====================================================================
# ALGORITMOS DE BÚSQUEDA
# Cada uno recibe (grafo, inicio, objetivo) y devuelve un Resultado con:
#   camino:     celdas desde inicio hasta objetivo (o None si no hay)
#   expandidos: celdas expandidas, en el orden en que se expandieron
#   generados:  cantidad de celdas descubiertas (expandidas + en la frontera)
#   costo:      costo total del camino (o None si no hay)
# =====================================================================
Resultado = namedtuple("Resultado", "camino expandidos generados costo")


def reconstruir(predecesor, nodo):
    camino = []
    while nodo is not None:
        camino.append(nodo)
        nodo = predecesor[nodo]
    camino.reverse()
    return camino


def costo_camino(grafo, camino):
    return sum(grafo[a][b] for a, b in zip(camino, camino[1:]))


def _resultado(grafo, predecesor, nodo, expandidos, generados):
    camino = reconstruir(predecesor, nodo)
    return Resultado(camino, expandidos, generados, costo_camino(grafo, camino))


def bfs(grafo, inicio, objetivo):
    cola = deque([inicio])
    visitados = {inicio}
    predecesor = {inicio: None}
    expandidos = []
    while cola:
        nodo = cola.popleft()
        expandidos.append(nodo)
        if nodo == objetivo:
            return _resultado(grafo, predecesor, nodo, expandidos, len(visitados))
        for vecino in grafo[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                predecesor[vecino] = nodo
                cola.append(vecino)
    return Resultado(None, expandidos, len(visitados), None)


def dfs(grafo, inicio, objetivo):
    """DFS clásico: el nodo se marca al EXPANDIRLO (al sacarlo de la pila)."""
    pila = [(inicio, None)]          # (nodo, padre con el que se descubrió)
    predecesor = {}                  # solo contiene nodos ya expandidos
    descubiertos = {inicio}
    expandidos = []
    while pila:
        nodo, padre = pila.pop()
        if nodo in predecesor:
            continue
        predecesor[nodo] = padre
        expandidos.append(nodo)
        if nodo == objetivo:
            return _resultado(grafo, predecesor, nodo, expandidos, len(descubiertos))
        for vecino in grafo[nodo]:
            if vecino not in predecesor:
                descubiertos.add(vecino)
                pila.append((vecino, nodo))
    return Resultado(None, expandidos, len(descubiertos), None)


def bcu(grafo, inicio, objetivo):
    dist = {inicio: 0}
    predecesor = {inicio: None}
    frontera = [(0, inicio)]
    expandidos = []
    while frontera:
        costo, nodo = heapq.heappop(frontera)
        if costo > dist[nodo]:       # entrada vieja de la cola de prioridad
            continue
        expandidos.append(nodo)
        if nodo == objetivo:
            return _resultado(grafo, predecesor, nodo, expandidos, len(dist))
        for vecino, c in grafo[nodo].items():
            nuevo = costo + c
            if nuevo < dist.get(vecino, float("inf")):
                dist[vecino] = nuevo
                predecesor[vecino] = nodo
                heapq.heappush(frontera, (nuevo, vecino))
    return Resultado(None, expandidos, len(dist), None)


# ---------------------------------------------------------------------
# Heurística (usada por A* y Greedy)
# ---------------------------------------------------------------------
def _pares_tunel(mapa):
    cols = len(mapa[0])
    return [((i, 0), (i, cols - 1)) for i in range(len(mapa))
            if mapa[i][0] == 'T' and mapa[i][cols - 1] == 'T']


PARES_TUNEL = _pares_tunel(MAPA)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristica(celda, meta):
    """Estimación del costo restante: distancia Manhattan.

    Es admisible porque cada casilla cuesta al menos 1. Con túnel, la ruta directa
    puede no ser la más corta, así que se toma el mínimo entre ir directo y cruzar
    por el túnel (ir a una boca, 1 paso de túnel, y de la otra boca a la meta).
    """
    h = manhattan(celda, meta)
    for a, b in PARES_TUNEL:
        h = min(h,
                manhattan(celda, a) + 1 + manhattan(b, meta),
                manhattan(celda, b) + 1 + manhattan(a, meta))
    return h


def astar(grafo, inicio, objetivo):
    g = {inicio: 0}
    predecesor = {inicio: None}
    h0 = heuristica(inicio, objetivo)
    frontera = [(h0, h0, 0, inicio)]          # (f, h, g, celda); a igual f gana la de menor h
    expandidos = []
    while frontera:
        _, _, costo, nodo = heapq.heappop(frontera)
        if costo > g[nodo]:                   # entrada vieja de la cola de prioridad
            continue
        expandidos.append(nodo)
        if nodo == objetivo:
            return _resultado(grafo, predecesor, nodo, expandidos, len(g))
        for vecino, c in grafo[nodo].items():
            nuevo = costo + c
            if nuevo < g.get(vecino, float("inf")):
                g[vecino] = nuevo
                predecesor[vecino] = nodo
                h = heuristica(vecino, objetivo)
                heapq.heappush(frontera, (nuevo + h, h, nuevo, vecino))
    return Resultado(None, expandidos, len(g), None)


def greedy(grafo, inicio, objetivo):
    """Búsqueda voraz: la frontera se ordena solo por h (ignora el costo acumulado).

    Es rápida, pero no garantiza el camino óptimo (por ejemplo, atraviesa el barro
    sin "darse cuenta" de que cuesta 3).
    """
    frontera = [(heuristica(inicio, objetivo), inicio)]
    predecesor = {inicio: None}        # también sirve como conjunto de descubiertos
    cerrados = set()
    expandidos = []
    while frontera:
        _, nodo = heapq.heappop(frontera)
        if nodo in cerrados:
            continue
        cerrados.add(nodo)
        expandidos.append(nodo)
        if nodo == objetivo:
            return _resultado(grafo, predecesor, nodo, expandidos, len(predecesor))
        for vecino in grafo[nodo]:
            if vecino not in predecesor:
                predecesor[vecino] = nodo
                heapq.heappush(frontera, (heuristica(vecino, objetivo), vecino))
    return Resultado(None, expandidos, len(predecesor), None)


# =====================================================================
# CONFIGURACIÓN VISUAL
# =====================================================================
CELDA = 36
ANCHO_MAPA = COLS * CELDA
ALTO_MAPA = FILAS * CELDA
ANCHO_PANEL = 280
# El panel necesita más alto que el mapa para que quepan todas sus líneas
ANCHO, ALTO = ANCHO_MAPA + ANCHO_PANEL, max(ALTO_MAPA, 520)

NEGRO = (0, 0, 0)
BLANCO = (240, 240, 240)
GRIS = (150, 150, 165)
AZUL_PARED = (20, 20, 110)
AZUL_BORDE = (70, 70, 255)
CASA = (18, 18, 32)
TUNEL = (12, 12, 45)
BARRO = (92, 64, 38)
BARRO_OSCURO = (66, 44, 26)
PUERTA = (255, 184, 222)
PILDORA = (255, 215, 170)
AMARILLO = (255, 221, 0)
PANEL = (15, 15, 26)

# clave en el mapa -> (algoritmo, color)
FANTASMAS_INFO = {
    '1': ("A*", (255, 60, 60)),
    '2': ("BCU", (255, 150, 220)),
    '3': ("BFS", (60, 220, 255)),
    '4': ("Greedy", (255, 170, 50)),
    '5': ("DFS", (90, 220, 90)),
}

# Algoritmos ya implementados: clave del fantasma en el mapa -> función de búsqueda.
ALGORITMOS = {'1': astar, '2': bcu, '3': bfs, '4': greedy, '5': dfs}
# Cada cuántos pasos replanifica cada fantasma (None = solo cuando termina su camino).
# En todos los casos, solo se replanifica si Pac-Man cambió de casilla desde la última búsqueda.
# DFS no replanifica antes: su camino cambia por completo de una búsqueda a otra
# (depende de desde dónde se busque) y el fantasma se queda oscilando sin avanzar.
REPLANIFICAR_CADA = {'1': 3, '2': 3, '3': 3, '4': 3, '5': None}

FILA_Y0, FILA_DY = 152, 26   # posición y separación de la lista de fantasmas en el panel

PASO_PACMAN_MS = 150
PASO_FANTASMA_MS = 220


# =====================================================================
# ENTIDADES
# =====================================================================
class Fantasma:
    def __init__(self, clave, pos, buscar):
        self.algoritmo, self.color = FANTASMAS_INFO[clave]
        self.buscar = buscar      # función de búsqueda: (grafo, inicio, meta) -> Resultado
        self.activo = True        # se cambia con el check del panel
        self.replan_cada = REPLANIFICAR_CADA.get(clave, 3)
        self.inicio = pos
        self.reiniciar()

    def reiniciar(self):
        self.pos = self.inicio
        self.camino = []          # celdas por recorrer (sin incluir la actual)
        self.visitados = set()    # celdas expandidas en la última búsqueda
        self.nodos_ultima = 0     # nodos expandidos en la última búsqueda
        self.nodos_total = 0      # nodos expandidos acumulados en la partida
        self.pasos_desde_plan = 0
        self.meta_ultima = None   # casilla de Pac-Man en la última planificación

    def planificar(self, grafo, meta):
        """Ejecuta el algoritmo del fantasma hacia `meta` y guarda el resultado."""
        r = self.buscar(grafo, self.pos, meta)
        self.camino = r.camino[1:] if r.camino else []
        self.visitados = set(r.expandidos)
        self.nodos_ultima = len(r.expandidos)
        self.nodos_total += len(r.expandidos)
        self.pasos_desde_plan = 0
        self.meta_ultima = meta

    def avanzar(self):
        """Da un solo paso siguiendo el camino ya planificado."""
        if self.camino:
            self.pos = self.camino.pop(0)
            self.pasos_desde_plan += 1


# =====================================================================
# JUEGO
# =====================================================================
class Juego:
    def __init__(self):
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Pac-Man - Algoritmos de búsqueda")
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont("arial", 15)
        self.fuente_b = pygame.font.SysFont("arial", 17, bold=True)
        self.fuente_g = pygame.font.SysFont("arial", 20, bold=True)
        self.fuente_xl = pygame.font.SysFont("arial", 40, bold=True)

        self.grafo_pacman = construir_grafo(MAPA, es_fantasma=False)
        self.grafo_fantasma = construir_grafo(MAPA, es_fantasma=True)
        pos = buscar_posiciones(MAPA)
        self.pos_inicial_pacman = pos['P']
        self.fantasmas = [Fantasma(k, pos[k], ALGORITMOS[k]) for k in '12345' if k in ALGORITMOS]
        self.alcanzables = celdas_alcanzables(self.grafo_pacman, pos['P'])

        self.ver_caminos = True
        self.ver_visitados = False
        self.ver_coordenadas = False
        self.fuente_c = pygame.font.SysFont("arial", 10)
        self.cache_coord = {}
        self.reiniciar()

    def reiniciar(self):
        self.pac_pos = self.pos_inicial_pacman
        self.dir = (0, 0)
        self.dir_deseada = (0, 0)
        self.mirando = (0, 1)
        self.puntaje = 0
        self.estado = "esperando"    # esperando | jugando | ganaste | perdiste
        self.t_pac = 0
        self.t_fan = 0
        self.pildoras = {}
        for i, fila in enumerate(MAPA):
            for j, c in enumerate(fila):
                if c in '.o' and (i, j) in self.alcanzables:
                    self.pildoras[(i, j)] = 10 if c == '.' else 50
        for f in self.fantasmas:
            f.reiniciar()

    # ------------------------------------------------------------ lógica
    def eventos(self):
        teclas_dir = {
            pygame.K_UP: (-1, 0), pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0), pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1), pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1), pygame.K_d: (0, 1),
        }
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.salir()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for k, f in enumerate(self.fantasmas):
                    if self.rect_fila(k).collidepoint(e.pos):
                        self.alternar_fantasma(f)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.salir()
                elif e.key in teclas_dir:
                    self.dir_deseada = teclas_dir[e.key]
                    if self.estado == "esperando":
                        self.estado = "jugando"
                elif e.key == pygame.K_r:
                    self.reiniciar()
                elif e.key == pygame.K_p:
                    self.ver_caminos = not self.ver_caminos
                elif e.key == pygame.K_v:
                    self.ver_visitados = not self.ver_visitados
                elif e.key == pygame.K_c:
                    self.ver_coordenadas = not self.ver_coordenadas

    def rect_fila(self, k):
        """Zona clicable (fila completa) del fantasma número k en el panel."""
        return pygame.Rect(ANCHO_MAPA + 8, FILA_Y0 + k * FILA_DY - 3, ANCHO_PANEL - 16, FILA_DY - 2)

    def alternar_fantasma(self, f):
        """Activa o desactiva un fantasma. Al cambiar, vuelve a su casa con estadísticas en cero."""
        f.activo = not f.activo
        f.reiniciar()

    def salir(self):
        pygame.quit()
        sys.exit()

    def actualizar(self, dt):
        if self.estado != "jugando":
            return

        self.t_pac += dt
        if self.t_pac >= PASO_PACMAN_MS:
            self.t_pac -= PASO_PACMAN_MS
            self.mover_pacman()
            self.comer()
            self.revisar_colision()

        self.t_fan += dt
        if self.t_fan >= PASO_FANTASMA_MS and self.estado == "jugando":
            self.t_fan -= PASO_FANTASMA_MS
            for f in self.fantasmas:
                if not f.activo:
                    continue
                # Solo se vuelve a buscar si Pac-Man cambió de casilla desde la última búsqueda
                pacman_se_movio = self.pac_pos != f.meta_ultima
                toca_replanificar = (not f.camino or (f.replan_cada is not None
                                                      and f.pasos_desde_plan >= f.replan_cada))
                if pacman_se_movio and toca_replanificar:
                    f.planificar(self.grafo_fantasma, self.pac_pos)
                f.avanzar()
            self.revisar_colision()

    def mover_pacman(self):
        for d in (self.dir_deseada, self.dir):
            if d == (0, 0):
                continue
            destino = vecino_en_direccion(self.grafo_pacman, self.pac_pos, d)
            if destino:
                self.pac_pos = destino
                self.dir = d
                self.mirando = d
                return
        self.dir = (0, 0)

    def comer(self):
        if self.pac_pos in self.pildoras:
            self.puntaje += self.pildoras.pop(self.pac_pos)
            if not self.pildoras:
                self.estado = "ganaste"

    def revisar_colision(self):
        if any(f.pos == self.pac_pos for f in self.fantasmas if f.activo):
            self.estado = "perdiste"

    # ------------------------------------------------------------ dibujo
    def centro(self, celda):
        i, j = celda
        return (j * CELDA + CELDA // 2, i * CELDA + CELDA // 2)

    def es_pared(self, i, j):
        return 0 <= i < FILAS and 0 <= j < COLS and MAPA[i][j] == '#'

    def dibujar_mapa(self):
        pan = self.pantalla
        for i, fila in enumerate(MAPA):
            for j, c in enumerate(fila):
                x, y = j * CELDA, i * CELDA
                rect = pygame.Rect(x, y, CELDA, CELDA)
                if c == '#':
                    pygame.draw.rect(pan, AZUL_PARED, rect)
                    # borde brillante solo en los lados que dan a un pasillo
                    lados = {
                        (-1, 0): ((x, y), (x + CELDA, y)),
                        (1, 0): ((x, y + CELDA), (x + CELDA, y + CELDA)),
                        (0, -1): ((x, y), (x, y + CELDA)),
                        (0, 1): ((x + CELDA, y), (x + CELDA, y + CELDA)),
                    }
                    for (di, dj), (p1, p2) in lados.items():
                        if not self.es_pared(i + di, j + dj):
                            pygame.draw.line(pan, AZUL_BORDE, p1, p2, 3)
                    continue

                pygame.draw.rect(pan, NEGRO, rect)
                if c == '~':
                    pygame.draw.rect(pan, BARRO, rect)
                    rnd = random.Random(i * 100 + j)
                    for _ in range(5):
                        px = x + rnd.randint(5, CELDA - 5)
                        py = y + rnd.randint(5, CELDA - 5)
                        pygame.draw.circle(pan, BARRO_OSCURO, (px, py), rnd.randint(2, 4))
                elif c in '12345':
                    pygame.draw.rect(pan, CASA, rect)
                elif c == 'D':
                    pygame.draw.rect(pan, PUERTA, (x + 2, y + CELDA // 2 - 3, CELDA - 4, 6))
                elif c == 'T':
                    pygame.draw.rect(pan, TUNEL, rect)
                    m = CELDA // 2
                    if j == 0:
                        pts = [(x + 8, y + m), (x + 22, y + m - 8), (x + 22, y + m + 8)]
                    else:
                        pts = [(x + CELDA - 8, y + m), (x + CELDA - 22, y + m - 8),
                               (x + CELDA - 22, y + m + 8)]
                    pygame.draw.polygon(pan, (120, 120, 255), pts)

    def dibujar_capas_busqueda(self):
        if self.ver_visitados:
            for f in self.fantasmas:
                if not f.activo or not f.visitados:
                    continue
                capa = pygame.Surface((CELDA, CELDA), pygame.SRCALPHA)
                capa.fill((*f.color, 55))
                for (i, j) in f.visitados:
                    self.pantalla.blit(capa, (j * CELDA, i * CELDA))
        if self.ver_caminos:
            for f in self.fantasmas:
                if not f.activo or not f.camino:
                    continue
                celdas = [f.pos] + f.camino
                for a, b in zip(celdas, celdas[1:]):
                    if abs(a[1] - b[1]) <= 1:      # no dibujar la línea que cruza el túnel
                        pygame.draw.line(self.pantalla, f.color, self.centro(a), self.centro(b), 3)
                for c in f.camino:
                    pygame.draw.circle(self.pantalla, f.color, self.centro(c), 5)

    def dibujar_coordenadas(self):
        """Escribe (fila,columna) en la esquina de cada casilla transitable."""
        if not self.ver_coordenadas:
            return
        for i, fila in enumerate(MAPA):
            for j, c in enumerate(fila):
                if c == '#':
                    continue
                img = self.cache_coord.get((i, j))
                if img is None:
                    img = self.fuente_c.render(f"{i},{j}", True, (225, 225, 240))
                    self.cache_coord[(i, j)] = img
                self.pantalla.blit(img, (j * CELDA + 2, i * CELDA + 1))

    def dibujar_pildoras(self):
        for celda, valor in self.pildoras.items():
            cx, cy = self.centro(celda)
            radio = 3 if valor == 10 else 9
            pygame.draw.circle(self.pantalla, PILDORA, (cx, cy), radio)

    def dibujar_pacman(self):
        cx, cy = self.centro(self.pac_pos)
        r = CELDA // 2 - 4
        pygame.draw.circle(self.pantalla, AMARILLO, (cx, cy), r)
        angulo = math.atan2(self.mirando[0], self.mirando[1])
        boca = math.radians(30)   # boca fija, sin animación
        pts = [(cx, cy),
               (cx + 2 * r * math.cos(angulo - boca), cy + 2 * r * math.sin(angulo - boca)),
               (cx + 2 * r * math.cos(angulo + boca), cy + 2 * r * math.sin(angulo + boca))]
        # la boca se dibuja en negro; se recorta al círculo con una máscara simple
        capa = pygame.Surface((CELDA, CELDA), pygame.SRCALPHA)
        pygame.draw.polygon(capa, (0, 0, 0, 255),
                            [(p[0] - (cx - CELDA // 2), p[1] - (cy - CELDA // 2)) for p in pts])
        self.pantalla.blit(capa, (cx - CELDA // 2, cy - CELDA // 2))

    def dibujar_fantasma(self, f):
        cx, cy = self.centro(f.pos)
        r = CELDA // 2 - 4
        pygame.draw.circle(self.pantalla, f.color, (cx, cy - 2), r)
        pygame.draw.rect(self.pantalla, f.color, (cx - r, cy - 2, 2 * r, r))
        for k in range(3):   # "pies" ondulados
            px = cx - r + r // 3 + k * (2 * r // 3)
            pygame.draw.circle(self.pantalla, f.color, (px, cy + r - 2), r // 3)
        for dx in (-5, 5):   # ojos
            pygame.draw.circle(self.pantalla, BLANCO, (cx + dx, cy - 4), 4)
            pygame.draw.circle(self.pantalla, (30, 30, 200), (cx + dx + 1, cy - 4), 2)

    def texto(self, s, x, y, color=BLANCO, fuente=None):
        self.pantalla.blit((fuente or self.fuente).render(s, True, color), (x, y))

    def dibujar_panel(self):
        x0 = ANCHO_MAPA
        pygame.draw.rect(self.pantalla, PANEL, (x0, 0, ANCHO_PANEL, ALTO))
        x = x0 + 16
        self.texto("PAC-MAN · BÚSQUEDAS", x, 12, AMARILLO, self.fuente_g)
        self.texto(f"Puntaje: {self.puntaje}", x, 50)
        self.texto(f"Píldoras restantes: {len(self.pildoras)}", x, 72)
        self.texto(f"Estado: {self.estado}", x, 94)

        self.texto("Fantasmas (clic para activar)", x, 126, GRIS, self.fuente_b)
        for k, f in enumerate(self.fantasmas):
            y = FILA_Y0 + k * FILA_DY
            # casilla (check)
            caja = pygame.Rect(x, y, 18, 18)
            pygame.draw.rect(self.pantalla, BLANCO if f.activo else GRIS, caja, 2, border_radius=3)
            if f.activo:
                pygame.draw.lines(self.pantalla, (90, 220, 90), False,
                                  [(x + 4, y + 10), (x + 8, y + 14), (x + 15, y + 5)], 3)
            color = f.color if f.activo else (80, 80, 90)
            pygame.draw.circle(self.pantalla, color, (x + 34, y + 9), 8)
            self.texto(f.algoritmo, x + 48, y, BLANCO if f.activo else GRIS, self.fuente_b)
            if f.activo:
                self.texto(f"últ {f.nodos_ultima} · tot {f.nodos_total}", x + 112, y + 1, GRIS)
            else:
                self.texto("desactivado", x + 112, y + 1, (100, 100, 110))
        y = FILA_Y0 + len(self.fantasmas) * FILA_DY
        self.texto("(nodos expandidos por búsqueda)", x, y, GRIS)

        y += 22
        pygame.draw.rect(self.pantalla, BARRO, (x, y, 16, 16))
        self.texto("Barro: costo 3 (resto: costo 1)", x + 26, y, GRIS)
        self.texto("A*/BCU/BFS/Greedy replanifican cada 3 pasos", x, y + 22, GRIS)
        self.texto("DFS replanifica al terminar su camino", x, y + 40, GRIS)
        self.texto("(solo si Pac-Man cambió de casilla)", x, y + 58, GRIS)

        y += 84
        self.texto("Controles", x, y, GRIS, self.fuente_b)
        onoff = lambda b: "ON" if b else "OFF"
        lineas = [
            "Flechas / WASD: mover",
            f"P: caminos [{onoff(self.ver_caminos)}]",
            f"V: nodos visitados [{onoff(self.ver_visitados)}]",
            f"C: coordenadas (fila,col) [{onoff(self.ver_coordenadas)}]",
            "R: reiniciar     ESC: salir",
        ]
        for k, linea in enumerate(lineas):
            self.texto(linea, x, y + 24 + k * 19)

    def dibujar_mensaje_final(self):
        if self.estado == "esperando":
            sub = self.fuente_b.render("Presiona una flecha para empezar", True, BLANCO)
            r = sub.get_rect(center=(ANCHO_MAPA // 2, ALTO_MAPA // 2 + 60))
            pygame.draw.rect(self.pantalla, NEGRO, r.inflate(24, 14), border_radius=8)
            self.pantalla.blit(sub, r)
            return
        if self.estado == "jugando":
            return
        velo = pygame.Surface((ANCHO_MAPA, ALTO_MAPA), pygame.SRCALPHA)
        velo.fill((0, 0, 0, 170))
        self.pantalla.blit(velo, (0, 0))
        msg, color = (("¡GANASTE!", AMARILLO) if self.estado == "ganaste"
                      else ("PERDISTE", (255, 70, 70)))
        img = self.fuente_xl.render(msg, True, color)
        self.pantalla.blit(img, img.get_rect(center=(ANCHO_MAPA // 2, ALTO_MAPA // 2 - 14)))
        sub = self.fuente.render("Presiona R para reiniciar", True, BLANCO)
        self.pantalla.blit(sub, sub.get_rect(center=(ANCHO_MAPA // 2, ALTO_MAPA // 2 + 24)))

    def dibujar(self):
        self.pantalla.fill(NEGRO)
        self.dibujar_mapa()
        self.dibujar_capas_busqueda()
        self.dibujar_coordenadas()
        self.dibujar_pildoras()
        self.dibujar_pacman()
        for f in self.fantasmas:
            if f.activo:
                self.dibujar_fantasma(f)
        self.dibujar_panel()
        self.dibujar_mensaje_final()

    # ------------------------------------------------------------ bucle
    def ejecutar(self):
        while True:
            dt = self.reloj.tick(60)
            self.eventos()
            self.actualizar(dt)
            self.dibujar()
            pygame.display.flip()


if __name__ == "__main__":
    Juego().ejecutar()