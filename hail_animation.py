# hail_animation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================
# Animación de altura vs tiempo (solo weak)
# =============================================
def animate_hailstone(traj, save_path=None, interval=50):
    """
    Animación de la trayectoria de un grano de granizo (altura vs tiempo) filtrando solo fase 'weak'.
    """
    x = traj['time']
    y = traj['height']
    phase = traj.get('phase', ['weak']*len(x))  # por defecto 'weak'

    # Filtramos solo los puntos weak
    x_weak = [xi for xi, p in zip(x, phase) if p == 'weak']
    y_weak = [yi for yi, p in zip(y, phase) if p == 'weak']

    if not x_weak:
        print("⚠️ No hay puntos de fase 'weak' para graficar altura.")
        return

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(min(x_weak), max(x_weak))
    ax.set_ylim(0, max(y_weak)*1.1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Trayectoria del granizo — Fase weak")
    line, = ax.plot([], [], 'b-o')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x_weak[:frame], y_weak[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(x_weak), init_func=init,
                        blit=True, interval=50)

    if save_path:
        ani.save(save_path, writer='pillow', dpi=80)
    else:
        plt.show()


# =============================================
# Animación de radio vs tiempo (solo weak)
# =============================================
def animate_hail_growth(traj, save_path=None, interval=100):
    """
    Animación del crecimiento del granizo (radio vs tiempo),
    usando únicamente la fase 'weak'.
    
    Parámetros
    ----------
    traj : dict
        Diccionario con claves 'time', 'radius' y 'phase'.
    save_path : str, opcional
        Ruta para guardar la animación (GIF o MP4).
    interval : int
        Milisegundos entre cuadros.
    """
    t = traj['time']
    R = np.array(traj['radius']) * 1000.0  # convertir a mm
    phase = traj.get('phase', ['weak']*len(t))

    # Filtramos solo weak
    t_weak = [ti for ti, p in zip(t, phase) if p == 'weak']
    R_weak = [Ri for Ri, p in zip(R, phase) if p == 'weak']

    if not t_weak:
        print("⚠️ No hay puntos de fase 'weak' para graficar radio.")
        return

    # Color único para weak
    phase_colors = ['tab:blue'] * len(t_weak)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xlim(min(t_weak), max(t_weak))
    ax.set_ylim(0, max(R_weak)*1.1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Radio (mm)")
    ax.set_title("Evolución del radio del granizo — Fase weak")

    line, = ax.plot([], [], lw=2)
    phase_text = ax.text(0.7, 0.9, '', transform=ax.transAxes, fontsize=12, color='k')

    def init():
        line.set_data([], [])
        phase_text.set_text('')
        return line, phase_text

    def update(frame):
        line.set_data(t_weak[:frame], R_weak[:frame])
        phase_text.set_text("Fase: Weak")
        return line, phase_text

    ani = FuncAnimation(fig, update, frames=len(t_weak), init_func=init,
                        blit=True, interval=50)

    if save_path:
        ani.save(save_path, writer='pillow', dpi=80)
    else:
        plt.show()
