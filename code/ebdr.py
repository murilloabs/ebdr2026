import sys
sys.path.append(r"C:\Users\M\Desktop\ROSS\ross")

import numpy as np
import ross as rs
import matplotlib.pyplot as plt

# ==========================================
# 1. Parâmetros e Materiais
# ==========================================
steel = rs.materials.steel

# ==========================================
# 2. Configuração do Rotor 1 (Motriz)
# ==========================================
shaft_r1 = [
    rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel),
    rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel)
]

gear1 = rs.GearElement.from_geometry(
    n=1, 
    material=steel, 
    width=0.04,          
    i_d=0.05,            
    o_d=0.15,            
    n_teeth=30,          
    pr_angle=np.radians(20),  
    helix_angle=np.radians(25) 
)

bearing1_r1 = rs.BearingElement(n=0, kxx=1e7, kyy=1.5e7, kzz=1e7, cxx=5e2, cyy=8e2, czz=5e2)
bearing2_r1 = rs.BearingElement(n=2, kxx=1e7, kyy=1.5e7, kzz=1e7, cxx=5e2, cyy=8e2, czz=5e2)

rotor1 = rs.Rotor(shaft_r1, [gear1], [bearing1_r1, bearing2_r1], tag="Rotor Motriz")

# ==========================================
# 3. Configuração do Rotor 2 (Movido)
# ==========================================
shaft_r2 = [
    rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel),
    rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel)
]

gear2 = rs.GearElement.from_geometry(
    n=1, 
    material=steel, 
    width=0.04, 
    i_d=0.05, 
    o_d=0.30, 
    n_teeth=60, 
    pr_angle=np.radians(20), 
    helix_angle=np.radians(25) 
)

bearing1_r2 = rs.BearingElement(n=0, kxx=1e7, kyy=1.5e7, kzz=1e7, cxx=5e2, cyy=8e2, czz=5e2)
bearing2_r2 = rs.BearingElement(n=2, kxx=1e7, kyy=1.5e7, kzz=1e7, cxx=5e2, cyy=8e2, czz=5e2)

rotor2 = rs.Rotor(shaft_r2, [gear2], [bearing1_r2, bearing2_r2], tag="Rotor Movido")

# ==========================================
# 4. Montagem do MultiRotor
# ==========================================
multi_rotor = rs.MultiRotor(
    driving_rotor=rotor1,
    driven_rotor=rotor2,
    coupled_nodes=(1, 1), 
    orientation_angle=0.0 
)

# ==========================================
# 5. Análise de Resposta no Tempo
# ==========================================
time_array = np.linspace(0, 0.5, 1000) 
speed_rads = 300.0 

unbalance_magnitude = 0.005 
F_unbalance = multi_rotor.unbalance_force_over_time(
    node=[1], 
    magnitude=[unbalance_magnitude], 
    phase=[0.0], 
    omega=speed_rads, 
    t=time_array
)

response = multi_rotor.run_time_response(speed=speed_rads, F=F_unbalance.T, t=time_array)

# ==========================================
# 6. Extração dos Graus de Liberdade (DOFs)
# ==========================================
idx_g1_start = 1 * 6
idx_g1_end   = idx_g1_start + 6

offset_r2 = rotor1.ndof
idx_g2_start = offset_r2 + (1 * 6)
idx_g2_end   = idx_g2_start + 6

dofs_gear1 = response.yout[:, idx_g1_start:idx_g1_end]
dofs_gear2 = response.yout[:, idx_g2_start:idx_g2_end]

# --- CORREÇÃO 1: Fatiando os arrays para pegar apenas o Regime Permanente (após 0.35s) ---
mask_steady = time_array >= 0.35
t_steady = time_array[mask_steady]
dofs_gear1_steady = dofs_gear1[mask_steady, :]
dofs_gear2_steady = dofs_gear2[mask_steady, :]

# ==========================================
# 7. Plotagem dos Resultados
# ==========================================
labels = ['Translacional X (m)', 'Translacional Y (m)', 'Translacional Z (m)', 
          'Rotacional Rx (rad)', 'Rotacional Ry (rad)', 'Rotacional Rz/Torsional (rad)']

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Regime Permanente nos Graus de Liberdade - Acoplamento Helicoidal", fontsize=16)

for i, ax in enumerate(axs.flatten()):
    # Engrenagem 1 plotada no Eixo Y da Esquerda (Azul)
    line1 = ax.plot(t_steady, dofs_gear1_steady[:, i], color='blue', label='Eng. 1 (Escala Esq.)')
    ax.set_ylabel('Amp. Eng 1', color='blue', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True)
    ax.set_title(labels[i])
    ax.set_xlabel('Tempo (s)')
    
    # --- CORREÇÃO 2: Criação de Eixo Y Secundário para a Engrenagem 2 (Vermelho) ---
    ax2 = ax.twinx()
    line2 = ax2.plot(t_steady, dofs_gear2_steady[:, i], color='red', alpha=0.8, label='Eng. 2 (Escala Dir.)')
    ax2.set_ylabel('Amp. Eng 2', color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Juntando as legendas dos dois eixos para ficar num quadro só
    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc='upper right', fontsize=8)

plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.5, wspace=0.35)
plt.show()