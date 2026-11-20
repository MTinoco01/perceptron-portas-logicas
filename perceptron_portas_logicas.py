"""
Trabalho P2 | Perceptron: Portas Lógicas AND, OR e NAND
Disciplina: Inteligência Artificial 
Professor: Alexandre Augusto Angelo de Souza

Integrantes:
    - Matheus Tinoco Costa
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt

PASTA = "resultados"
os.makedirs(PASTA, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
CASAS = 10


class PortaLogica:
    """Guarda apenas os dados de uma porta lógica ."""

    ENTRADAS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    def __init__(self, nome, saidas):
        self.nome = nome
        self.X = PortaLogica.ENTRADAS.copy()
        self.y = np.array(saidas)

    def tabela(self):
        """Devolve a tabela-verdade como DataFrame."""
        return pd.DataFrame({"x1": self.X[:, 0], "x2": self.X[:, 1], self.nome: self.y})


class Perceptron:
    """Neurônio de Rosenblatt com função de ativação degrau."""

    def __init__(self, w1, w2, b, eta):
        self.w = np.array([w1, w2], dtype=float)
        self.b = float(b)
        self.eta = eta
        self.historico = []   
        self.retas = []      
        self.epocas = 0

    def potencial(self, x):
        """u = w1*x1 + w2*x2 + b (arredondado para evitar erro de ponto flutuante)."""
        return round(float(np.dot(self.w, x) + self.b), CASAS)

    def ativacao(self, u):
        """Degrau: 1 se u >= 0, senão 0."""
        return 1 if u >= 0 else 0

    def prever(self, X):
        """Aplica potencial + ativação a uma ou várias entradas."""
        X = np.atleast_2d(X)
        return np.array([self.ativacao(self.potencial(x)) for x in X])

    def atualizar(self, x, y):
        """Processa uma amostra: calcula u, ŷ e erro; corrige pesos/bias se errou."""
        u = self.potencial(x)
        y_hat = self.ativacao(u)
        erro = y - y_hat
        if erro != 0:
            self.w = np.round(self.w + self.eta * erro * np.array(x), CASAS)
            self.b = round(self.b + self.eta * erro, CASAS)
        return u, y_hat, erro

    def treinar(self, porta, max_epocas=100):
        """Repete épocas até uma época sem erros (ou até max_epocas)."""
        for epoca in range(1, max_epocas + 1):
            self.retas.append(("início", epoca, self.w[0], self.w[1], self.b))
            erros_epoca = 0
            for x, y in zip(porta.X, porta.y):
                w1, w2, b = self.w[0], self.w[1], self.b 
                u, y_hat, erro = self.atualizar(x, y)
                erros_epoca += abs(erro)
                self.historico.append({
                    "época": epoca, "x1": x[0], "x2": x[1], "y": y,
                    "w1": w1, "w2": w2, "b": b, "u": u, "ŷ": y_hat, "erro": erro,
                    "novo w1": self.w[0], "novo w2": self.w[1], "novo b": self.b,
                })
            self.retas.append(("fim", epoca, self.w[0], self.w[1], self.b))
            self.epocas = epoca
            if erros_epoca == 0:  
                break
        return self

    def reta_decisao(self):
        """Devolve a função x2(x1) da reta w1*x1 + w2*x2 + b = 0.
        Se w2 == 0 a reta é vertical (x1 = -b/w1) e não pode ser escrita como x2(x1);
        nesse caso devolve None (o gráfico trata esse caso)."""
        return funcao_reta(self.w[0], self.w[1], self.b)

    def tabela_treino(self):
        return pd.DataFrame(self.historico)


# ----------------------------- Funções auxiliares -----------------------------

def funcao_reta(w1, w2, b):
    if w2 == 0:
        return None
    return lambda x1: -(w1 * x1 + b) / w2


def desenhar_reta(ax, w1, w2, b, **estilo):
    """Desenha a reta w1*x1 + w2*x2 + b = 0 (inclusive o caso vertical)."""
    xs = np.linspace(-0.5, 1.5, 100)
    f = funcao_reta(w1, w2, b)
    if f is not None:
        ax.plot(xs, f(xs), **estilo)
    elif w1 != 0:
        ax.axvline(-b / w1, **estilo)


def desenhar_pontos(ax, porta):
    """Pontos da tabela-verdade: saída 1 em azul, saída 0 em vermelho."""
    for valor, cor, marcador in [(0, "red", "o"), (1, "blue", "s")]:
        pts = porta.X[porta.y == valor]
        ax.scatter(pts[:, 0], pts[:, 1], c=cor, marker=marcador, s=120,
                   edgecolors="k", zorder=3, label=f"saída {valor}")


def formatar_eixos(ax, titulo):
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)


def grafico_porta(porta, w1, w2, b, titulo, arquivo):
    fig, ax = plt.subplots(figsize=(7, 5))
    desenhar_pontos(ax, porta)
    desenhar_reta(ax, w1, w2, b, color="green", label="reta de decisão")
    formatar_eixos(ax, titulo)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, arquivo), dpi=120)
    plt.close(fig)


def grafico_evolucao_and(porta, modelo):
    """Posição da reta no início e ao fim de cada época (porta AND)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    desenhar_pontos(ax, porta)
    cores = plt.cm.viridis(np.linspace(0, 0.9, len(modelo.retas)))
    for (momento, epoca, w1, w2, b), cor in zip(modelo.retas, cores):
        estilo = "--" if momento == "início" else "-"
        desenhar_reta(ax, w1, w2, b, color=cor, linestyle=estilo,
                      label=f"época {epoca} ({momento})")
    formatar_eixos(ax, "AND – evolução da reta de decisão por época")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "AND_evolucao_retas.png"), dpi=120)
    plt.close(fig)


# ------------------------------ Execução Parte 1 ------------------------------

PORTAS = {
    "AND": PortaLogica("AND", [0, 0, 0, 1]),
    "OR": PortaLogica("OR", [0, 1, 1, 1]),
    "NAND": PortaLogica("NAND", [1, 1, 1, 0]),
}

INICIAIS = {
    "AND": dict(w1=0.3, w2=0.4, b=0.1, eta=0.2),
    "OR": dict(w1=0.4, w2=-0.1, b=0.2, eta=0.3),
    "NAND": dict(w1=0.1, w2=0.1, b=0.1, eta=0.2),
}

print("=" * 90)
print("PARTE 1 - IMPLEMENTAÇÃO PRÓPRIA")
print("=" * 90)

modelos_p1 = {}
for nome, porta in PORTAS.items():
    print(f"\n--- Tabela-verdade {nome} ---")
    print(porta.tabela().to_string(index=False))

    p = Perceptron(**INICIAIS[nome]).treinar(porta)
    modelos_p1[nome] = p

    tabela = p.tabela_treino()
    tabela.to_csv(os.path.join(PASTA, f"{nome}_treinamento.csv"), index=False)
    print(f"\n--- Treinamento {nome} (valores iniciais: {INICIAIS[nome]}) ---")
    print(tabela.to_string(index=False))

    prev = p.prever(porta.X)
    print(f"\nPesos finais: w1 = {p.w[0]}, w2 = {p.w[1]}, b = {p.b}")
    print(f"Épocas necessárias: {p.epocas}")
    print(f"Previsões: {prev}  | Desejado: {porta.y}  | "
          f"Todas corretas? {'SIM' if np.array_equal(prev, porta.y) else 'NÃO'}")

    grafico_porta(porta, p.w[0], p.w[1], p.b,
                  f"{nome} – Parte 1 (w1={p.w[0]}, w2={p.w[1]}, b={p.b})",
                  f"{nome}_parte1.png")

grafico_evolucao_and(PORTAS["AND"], modelos_p1["AND"])

p = modelos_p1["AND"]
assert p.epocas == 3 and (p.w[0], p.w[1], p.b) == (0.3, 0.2, -0.5), \
    "AND não reproduziu o resultado de sala (w1=0.3, w2=0.2, b=-0.5 em 3 épocas)"
print("\n[OK] AND reproduz a resolução de sala: w1=0.3, w2=0.2, b=-0.5 em 3 épocas.")

# PARTE 2 - MODELO PRONTO (scikit-learn)
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.exceptions import ConvergenceWarning

print("\n" + "=" * 90)
print("PARTE 2 - SCIKIT-LEARN")
print("=" * 90)

comparacao = []
for nome, porta in PORTAS.items():
    modelo = SkPerceptron(eta0=0.2, max_iter=100, tol=None,
                          shuffle=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(porta.X, porta.y)

    w1, w2 = modelo.coef_[0]
    b = modelo.intercept_[0]
    prev_sk = modelo.predict(porta.X)
    print(f"\n{nome}: coef_ = {modelo.coef_}, intercept_ = {modelo.intercept_}, "
          f"predict = {prev_sk}, épocas (n_iter_) = {modelo.n_iter_}")

    grafico_porta(porta, w1, w2, b,
                  f"{nome} – scikit-learn (w1={w1:g}, w2={w2:g}, b={b:g})",
                  f"{nome}_sklearn.png")

    p = modelos_p1[nome]
    prev_p1 = p.prever(porta.X)
    comparacao.append({
        "porta": nome,
        "P1 w1": p.w[0], "P1 w2": p.w[1], "P1 b": p.b,
        "SK w1": round(w1, CASAS), "SK w2": round(w2, CASAS), "SK b": round(b, CASAS),
        "previsões P1": str(prev_p1), "previsões SK": str(prev_sk),
        "coincidem?": "SIM" if np.array_equal(prev_p1, prev_sk) else "NÃO",
    })

tabela_comp = pd.DataFrame(comparacao)
tabela_comp.to_csv(os.path.join(PASTA, "comparacao_parte1_parte2.csv"), index=False)
print("\n--- Comparação Parte 1 x Parte 2 ---")
print(tabela_comp.to_string(index=False))

print(f"Gráficos e tabelas CSV salvos na pasta '{PASTA}/'.")
