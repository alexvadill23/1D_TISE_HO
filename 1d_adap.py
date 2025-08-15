import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn import SchrodingerPINN
import os
import time
from scipy.stats import linregress

L = 6.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_adap(
    n,
    plot_result=True,
):
    print(f"\n=== ENTRENAR ESTADO n={n} (MALLA ADAPTATIVA) ===")
    torch.manual_seed(42)
    np.random.seed(42)
    puntos_iniciales = 100
    puntos_por_refinamiento = 50
    frecuencia_refinamiento = 1000

    MARGEN_TOLERANCIA = 0.02
    EPOCAS_ESTABLES_CONSECUTIVAS = 1000
    VENTANA_ESTABILIDAD = 1000

    max_total_points = 1000  # Límite de puntos de colocación
    lr_switched = False

    model = SchrodingerPINN().to(device)
    model.E.data = torch.tensor([0], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    x_train = torch.linspace(-L, L, puntos_iniciales, device=device).view(-1, 1)
    E_history = []
    loss_history = []

    # Guardar las componentes de la loss
    loss_pde_history = []
    loss_bc_history = []
    loss_norm_history = []
    loss_sym_history = []
    loss_proj_history = []

    contador_estable = 0
    epoch = 0

    start = time.time()

    while True:
        for _ in range(frecuencia_refinamiento):
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
                    x_k = np.load(f"x_vals_n{k}_adap.npy")
                    psi_k = np.load(f"psi_pred_n{k}_adap.npy")
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
            total_loss.backward(retain_graph=True)
            optimizer.step()

            current_energy = model.E.item()
            E_history.append(current_energy)
            loss_history.append(total_loss.item())

            # Guardar las componentes de la loss
            loss_pde_history.append(loss_pde.item())
            loss_bc_history.append(loss_bc.item())
            loss_norm_history.append(loss_norm.item())
            loss_sym_history.append(loss_sym.item())
            loss_proj_history.append(loss_proj.item())

            # Criterio de parada: estabilidad + pendiente
            if len(E_history) >= VENTANA_ESTABILIDAD:
                energias_recientes = E_history[-VENTANA_ESTABILIDAD:]
                energia_media = np.mean(energias_recientes)
                desviacion = np.std(energias_recientes)
                variacion_relativa = desviacion / abs(energia_media) if abs(energia_media) > 1e-8 else float('inf')
                epochs_window = np.arange(VENTANA_ESTABILIDAD)
                slope, _, _, _, _ = linregress(epochs_window, energias_recientes)
                if variacion_relativa < MARGEN_TOLERANCIA and abs(slope) < 1e-5:
                    contador_estable += 1
                else:
                    contador_estable = 0
                if contador_estable >= EPOCAS_ESTABLES_CONSECUTIVAS:
                    print(f"\nCriterio de estabilidad alcanzado: {contador_estable} épocas estables consecutivas (variación relativa={variacion_relativa:.6f}, pendiente={slope:.2e})")
                    break

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
            epoch += 1
        # Refinamiento adaptativo
        if x_train.shape[0] < max_total_points:
            x_dense = torch.linspace(-L, L, 2000, device=device, requires_grad=True).view(-1, 1)
            psi_dense = model(x_dense)
            dpsi_dx_dense = torch.autograd.grad(psi_dense, x_dense, grad_outputs=torch.ones_like(psi_dense), create_graph=True)[0]
            d2psi_dx2_dense = torch.autograd.grad(dpsi_dx_dense, x_dense, grad_outputs=torch.ones_like(dpsi_dx_dense), create_graph=True)[0]
            H_psi_dense = -0.5 * d2psi_dx2_dense + 0.5 * x_dense**2 * psi_dense
            residuo = torch.abs(H_psi_dense - model.E * psi_dense).detach().cpu().numpy().squeeze()
            idx_top = np.argsort(residuo)[-puntos_por_refinamiento:]
            nuevos_puntos = x_dense[idx_top]
            x_train = torch.cat([x_train, nuevos_puntos], dim=0)
            x_train, _ = torch.sort(x_train, dim=0)
            print(f"Refinamiento: +{puntos_por_refinamiento} puntos añadidos, total: {x_train.shape[0]}")
        elif not lr_switched:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4  # O el valor que prefieras
            lr_switched = True
            print(f"Learning rate cambiado a 1e-4 (malla completa alcanzada)")

        # Si se cumple el criterio de parada, sal del bucle principal
        if contador_estable >= EPOCAS_ESTABLES_CONSECUTIVAS:
            break

    end = time.time()

    # Guardar resultados
    torch.save(model.state_dict(), f"modelo_n{n}_adap.pth")
    training_info = {
        'training_time': end - start,
    }
    model.eval()
    with torch.no_grad():
        psi_pred = model(x_train).detach().cpu().numpy()
    x_np = x_train.detach().cpu().numpy().squeeze()
    norm = np.sqrt(np.trapezoid(np.abs(psi_pred.squeeze())**2, x_np))
    psi_pred_norm = psi_pred / norm
    np.save(f"x_vals_n{n}_adap.npy", x_np)
    np.save(f"psi_pred_n{n}_adap.npy", psi_pred_norm.squeeze())
    np.save(f"energia_history_n{n}_adap.npy", np.array(E_history))
    np.save(f"loss_history_n{n}_adap.npy", np.array(loss_history))
    np.save(f"training_info_n{n}_adap.npy", training_info)

    # Guardar las componentes de la loss
    np.save(f"loss_pde_history_n{n}_adap.npy", np.array(loss_pde_history))
    np.save(f"loss_bc_history_n{n}_adap.npy", np.array(loss_bc_history))
    np.save(f"loss_norm_history_n{n}_adap.npy", np.array(loss_norm_history))
    np.save(f"loss_sym_history_n{n}_adap.npy", np.array(loss_sym_history))
    np.save(f"loss_proj_history_n{n}_adap.npy", np.array(loss_proj_history))

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

    # Visualización de las distintas componentes de la loss
    plt.figure(figsize=(10,4))
    plt.plot(loss_pde_history, label="PDE")
    plt.plot(loss_bc_history, label="BC")
    plt.plot(loss_norm_history, label="Norm")
    plt.plot(loss_sym_history, label="Sym")
    plt.plot(loss_proj_history, label="Proj")
    plt.yscale('log')
    plt.xlabel("Época")
    plt.ylabel("Valor de cada término")
    plt.legend()
    plt.title("Evolución de las componentes de la loss")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n = int(input("Introduce el número de estado (n=0,1,2,...): "))
    train_adap(n)