import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Palavras e suas dimensões (valores fictícios de 0 a 10)
palavras = ["Guerreiro", "Filósofo", "Artista", "Cientista", "Líder"]

# Cada linha é uma palavra, cada coluna é uma dimensão (5D)
# [Agressividade, Inteligência, Criatividade, Sociabilidade, Liderança]
dados = [
    [9, 3, 2, 4, 7],   # Guerreiro
    [2, 9, 5, 8, 4],   # Filósofo
    [3, 6, 9, 5, 6],   # Artista
    [1, 10, 4, 9, 5],  # Cientista
    [7, 7, 6, 6, 9]    # Líder
]

# --- Escolhendo 3 dimensões (exemplo: 0=agressividade, 1=inteligência, 4=liderança)
x = [p[0] for p in dados]  # Agressividade
y = [p[1] for p in dados]  # Inteligência
z = [p[4] for p in dados]  # Liderança

# Criar gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, s=100, c='red')

# Adicionar labels das palavras
for i, palavra in enumerate(palavras):
    ax.text(x[i]+0.1, y[i]+0.1, z[i]+0.1, palavra)

# Nome dos eixos
ax.set_xlabel("Agressividade")
ax.set_ylabel("Inteligência")
ax.set_zlabel("Liderança")
ax.set_title("Visualização 3D de Palavras em 5 Dimensões (reduzido para 3D)")

plt.show()
