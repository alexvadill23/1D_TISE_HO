import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn import SchrodingerPINN
import os
import time
from scipy.stats import linregress

L = 6.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fija(
    n,
    plot_result=True,
):
    print(f"\n=== ENTRENAR ESTADO n={n} (MALLA FIJA) ===")
    torch.manual_seed(42)
    np.random.seed(42)
    num_points = 1000 

    MARGEN_TOLERANCIA = 0.02
    EPOCAS_ESTABLES_CONSECUTIVAS = 1000
    VENTANA_ESTABILIDAD = 1000

    model = SchrodingerPINN().to(device)
    model.E.data = torch.tensor([0], dtype=torch.float32, device=device)  # Inicialización cerca del valor teórico
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x_train = torch.linspace(-L, L, num_points, device=device).view(-1, 1)

    E_history = []
    loss_history = []
    contador_estable = 0

    start = time.time()

    for epoch in range(100000):
        optimizer.zero_grad()
        x_train.requires_grad_(True)
        psi = model(x_train)
        dpsi_dx = torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_train, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]
        H_psi = -0.5 * d2psi_dx2 + 0.5 * x_train**2 * psi
        loss_pde = torch.mean((H_psi - model.E * psi)**2)
        x_bc = torch.tensor([[-L], [L]], dtype=torch.float32, device=device)
        psi_bc = model(x_bc)
        loss_bc = torch.mean(psi_bc**2)
        prob_density = psi.squeeze()**2
        integral = torch.trapz(prob_density, x_train.squeeze())
        loss_norm = (integral - 1)**2
        psi_neg = model(-x_train)
        if n % 2 == 0:
            loss_sym = torch.mean((psi - psi_neg)**2)
        else:
            loss_sym = torch.mean((psi + psi_neg)**2)
        loss_proj = torch.tensor(0.0, device=device)
        for k in range(n):
            try:
                x_k = np.load(f"x_vals_n{k}_fija.npy")
                psi_k = np.load(f"psi_pred_n{k}_fija.npy")
                x_current = x_train.detach().cpu().numpy().squeeze()
                psi_k_interp = np.interp(x_current, x_k, psi_k)
                psi_k_tensor = torch.tensor(psi_k_interp, dtype=torch.float32, device=device).view(-1, 1)
                overlap = torch.trapz(psi_k_tensor.squeeze() * psi.squeeze(), x_train.squeeze())
                loss_proj += overlap**2
            except:
                continue

        total_loss = (
            0.3 * loss_pde +
            1.0 * loss_bc +
            3 * loss_norm +
            5 * loss_sym +
            10.0 * loss_proj
        )
        if len(E_history) >= 10:
            recent_E = torch.tensor(E_history[-10:], device=device)
            energy_var = torch.var(recent_E)
            total_loss += 0.03 * energy_var
        total_loss.backward()
        optimizer.step()

        current_energy = model.E.item()
        E_history.append(current_energy)
        loss_history.append(total_loss.item())

        # Criterio de parada: estabilidad + pendiente
        if len(E_history) >= VENTANA_ESTABILIDAD:
            energias_recientes = E_history[-VENTANA_ESTABILIDAD:]
            energia_media = np.mean(energias_recientes)
            desviacion = np.std(energias_recientes)
            variacion_relativa = desviacion / abs(energia_media) if abs(energia_media) > 1e-8 else float('inf')
            # Nuevo: calcula la pendiente (slope)
            epochs_window = np.arange(VENTANA_ESTABILIDAD)
            slope, _, _, _, _ = linregress(epochs_window, energias_recientes)
            if variacion_relativa < MARGEN_TOLERANCIA and abs(slope) < 1e-5:
                contador_estable += 1
            else:
                contador_estable = 0
            if contador_estable >= EPOCAS_ESTABLES_CONSECUTIVAS:
                print(f"\nCriterio de estabilidad alcanzado: {contador_estable} épocas estables consecutivas (variación relativa={variacion_relativa:.6f}, pendiente={slope:.2e})")
                break

        # Print cada 1000 épocas con debugging de cada término de la loss
        if epoch % 1000 == 0:
            if len(E_history) >= VENTANA_ESTABILIDAD:
                energias_recientes = E_history[-VENTANA_ESTABILIDAD:]
                energia_media = np.mean(energias_recientes)
                variacion = np.std(energias_recientes) / abs(energia_media) if abs(energia_media) > 1e-8 else float('inf')
                epochs_window = np.arange(VENTANA_ESTABILIDAD)
                slope, _, _, _, _ = linregress(epochs_window, energias_recientes)
                print(f"Época {epoch}: Loss={total_loss.item():.4f}, E={current_energy:.6f}, Puntos={x_train.shape[0]}, Var={variacion:.4f}, Estable={contador_estable}, Pendiente={slope:.2e}")
                print(f"  [DEBUG] loss_pde={loss_pde.item():.4e}, loss_bc={loss_bc.item():.4e}, loss_norm={loss_norm.item():.4e}, loss_sym={loss_sym.item():.4e}, loss_proj={loss_proj.item():.4e}")
            else:
                print(f"Época {epoch}: Loss={total_loss.item():.4f}, E={current_energy:.6f}, Puntos={x_train.shape[0]}")
                print(f"  [DEBUG] loss_pde={loss_pde.item():.4e}, loss_bc={loss_bc.item():.4e}, loss_norm={loss_norm.item():.4e}, loss_sym={loss_sym.item():.4e}, loss_proj={loss_proj.item():.4e}")

    end = time.time()

    # Guardar resultados
    torch.save(model.state_dict(), f"modelo_n{n}_fija.pth")
    training_info = {
    'training_time': end - start,
    # puedes añadir más info si quieres
    }
    model.eval()
    with torch.no_grad():
        psi_pred = model(x_train).detach().cpu().numpy()
    x_np = x_train.detach().cpu().numpy().squeeze()
    norm = np.sqrt(np.trapezoid(np.abs(psi_pred.squeeze())**2, x_np))
    psi_pred_norm = psi_pred / norm
    np.save(f"x_vals_n{n}_fija.npy", x_np)
    np.save(f"psi_pred_n{n}_fija.npy", psi_pred_norm.squeeze())
    np.save(f"energia_history_n{n}_fija.npy", np.array(E_history))
    np.save(f"loss_history_n{n}_fija.npy", np.array(loss_history))
    np.save(f"training_info_n{n}_fija.npy", training_info)  

    # Evaluación final
    energy_theoretical = n + 0.5
    energy_obtained = model.E.item()
    error_rel = abs(energy_obtained - energy_theoretical) / energy_theoretical * 100
    print(f"\nRESULTADOS FINALES (n={n}):")
    print(f"  Energía teórica: {energy_theoretical:.3f}")
    print(f"  Energía obtenida: {energy_obtained:.6f}")
    print(f"  Error relativo: {error_rel:.2f}%")
    print(f"  Puntos utilizados: {x_train.shape[0]}")
    print(f"  Tiempo total: {end-start:.2f} segundos")
    print(f"  Épocas completadas: {epoch+1}")

    # Visualización de la función de onda y la energía
    if plot_result:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(np.arange(len(E_history)), E_history, label="Energía PINN")
        plt.axhline(energy_theoretical, color='r', linestyle='--', label="Energía teórica")
        plt.xlabel("Época")
        plt.ylabel("Energía")
        plt.legend()
        plt.title("Energía durante el entrenamiento")


if __name__ == "__main__":
    n = int(input("Introduce el número de estado (n=0,1,2,...): "))
    train_fija(n)