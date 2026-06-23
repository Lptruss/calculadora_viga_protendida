# -*- coding: utf-8 -*-
"""
Calculadora de Viga Protendida - MEF 3D (Sólidos + Treliça)
Versão Limpa (Sem exportação 3D ParaView)
"""

import numpy as np
import pandas as pd

def gerar_nos_solido(L, bw, h, f, z_offset, div_x, div_y):
    dx = L / div_x
    dy = bw / div_y
    z_inf_max = z_offset - f
    dz_inf = z_inf_max / (div_x / 2)
    z_layers_inf = [round(i * dz_inf, 6) for i in range(int(div_x / 2) + 1)]
    dz_sup = (h - z_offset) / (div_x / 2)
    z_layers_sup = [round(z_offset + i * dz_sup, 6) for i in range(1, int(div_x / 2) + 1)]
    z_layers_parabolica = []
    for i in range(div_x + 1):
        x = i * dx
        z_trelica = round((-4 * f * x * (L - x)) / (L ** 2) + z_offset, 6)
        if z_trelica >= z_inf_max - 1e-6 and z_trelica <= z_offset + 1e-6:
            z_layers_parabolica.append(z_trelica)
    z_layers = sorted(set(z_layers_inf + z_layers_parabolica + z_layers_sup))
    if abs(z_layers[-1] - h) > 1e-6:
        z_layers.append(round(h, 6))
    coords, node_id, node_index = [], {}, 0
    for k, z in enumerate(z_layers):
        for j in range(div_y + 1):
            y = j * dy
            for i in range(div_x + 1):
                x = i * dx
                coords.append((x, y, z))
                node_id[(i, j, k)] = node_index
                node_index += 1
    return coords, node_id, z_layers

def gerar_elementos_solido(div_x, div_y, z_layers, node_id):
    elementos = []
    nz = len(z_layers) - 1
    for k in range(nz):
        for j in range(div_y):
            for i in range(div_x):
                n0 = node_id[(i,     j,     k)]
                n1 = node_id[(i + 1, j,     k)]
                n2 = node_id[(i + 1, j + 1, k)]
                n3 = node_id[(i,     j + 1, k)]
                n4 = node_id[(i,     j,     k + 1)]
                n5 = node_id[(i + 1, j,     k + 1)]
                n6 = node_id[(i + 1, j + 1, k + 1)]
                n7 = node_id[(i,     j + 1, k + 1)]
                elementos.append([n0, n1, n2, n3, n4, n5, n6, n7])
    return elementos

def gerar_nos_trelica_indices(L, f, div_x, z_offset, faixa_y, bw, div_y, z_layers, node_id):
    dx, dy, j = L / div_x, bw / div_y, faixa_y
    indices_trelica, coordenadas_trelica = [], []
    for i in range(div_x + 1):
        x, y = i * dx, j * dy
        z = round((-4 * f * x * (L - x)) / (L ** 2) + z_offset, 6)
        k = next((kk for kk, zz in enumerate(z_layers) if abs(zz - z) < 1e-6), None)
        if k is None: raise ValueError(f"z exato {z} não encontrado em z_layers.")
        indices_trelica.append(node_id[(i, j, k)])
        coordenadas_trelica.append((x, y, z))
    return indices_trelica, coordenadas_trelica

def formar_matriz_B_hexa(dN_glob):
    B = np.zeros((6, 24))
    for i in range(8):
        B[0, i*3] = dN_glob[i, 0]; B[1, i*3+1] = dN_glob[i, 1]; B[2, i*3+2] = dN_glob[i, 2]
        B[3, i*3] = dN_glob[i, 1]; B[3, i*3+1] = dN_glob[i, 0]
        B[4, i*3+1] = dN_glob[i, 2]; B[4, i*3+2] = dN_glob[i, 1]
        B[5, i*3] = dN_glob[i, 2]; B[5, i*3+2] = dN_glob[i, 0]
    return B

def calcular_tensoes_deformacoes_solidos(elementos_solidos, XYZ_original, U_total, E_solido, v_solido, pG, B_matrix_hex_func, formar_B_func):
    XYZ_final = XYZ_original + U_total.reshape((-1, 3))
    c1 = E_solido / ((1 + v_solido) * (1 - 2 * v_solido))
    D_solid = c1 * np.array([
        [1-v_solido, v_solido, v_solido, 0, 0, 0],
        [v_solido, 1-v_solido, v_solido, 0, 0, 0],
        [v_solido, v_solido, 1-v_solido, 0, 0, 0],
        [0, 0, 0, (1-2*v_solido)/2, 0, 0],
        [0, 0, 0, 0, (1-2*v_solido)/2, 0],
        [0, 0, 0, 0, 0, (1-2*v_solido)/2]
    ])
    stresses_all_elements, strains_all_elements = [], []
    U_total_reshaped = U_total.reshape((-1,3))
    for e_idx, nos_indices_globais in enumerate(elementos_solidos):
        XYZ_e_final = XYZ_final[nos_indices_globais]
        u_e_flat = U_total_reshaped[nos_indices_globais].ravel()
        stresses_gauss_points, strains_gauss_points = [], []
        for xi, eta, zeta in pG:
            dN_nat = B_matrix_hex_func(xi, eta, zeta)
            J = dN_nat.T @ XYZ_e_final
            try:
                if np.linalg.det(J) <= 1e-9: raise ValueError
                invJ = np.linalg.inv(J)
            except:
                strains_gauss_points.append(np.full(6, np.nan))
                stresses_gauss_points.append(np.full(6, np.nan))
                continue
            epsilon_gp = formar_B_func(dN_nat @ invJ) @ u_e_flat
            strains_gauss_points.append(epsilon_gp)
            stresses_gauss_points.append(D_solid @ epsilon_gp)
        stresses_all_elements.append(stresses_gauss_points)
        strains_all_elements.append(strains_gauss_points)
    return stresses_all_elements, strains_all_elements

def calcular_esforcos_trelica(IE_trelica, XYZ_original, U_total, E_trelica, A_trelica):
    XYZ_final = XYZ_original + U_total.reshape((-1, 3))
    epsilons_trel, sigmas_trel, forces_trel = [], [], []
    for n_i, n_j in IE_trelica:
        L0 = np.linalg.norm(XYZ_original[n_j] - XYZ_original[n_i])
        Lf = np.linalg.norm(XYZ_final[n_j] - XYZ_final[n_i])
        epsilon = (Lf - L0) / L0 if L0 >= 1e-9 else 0.0
        sigma = E_trelica * epsilon
        epsilons_trel.append(epsilon)
        sigmas_trel.append(sigma)
        forces_trel.append(sigma * A_trelica)
    return epsilons_trel, sigmas_trel, forces_trel

def cosdirL(e, IE, XYZ):
    ni, nj = IE[e]
    dx, dy, dz = XYZ[nj] - XYZ[ni]
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    return dx/L, dy/L, dz/L, L

def rigidezElemento(E, A, L, cx, cy, cz):
    T = np.array([cx, cy, cz])
    OT = np.outer(T, T)
    return (E * A / L) * np.block([[OT, -OT], [-OT, OT]])

def B_matrix_hex(xi, eta, zeta):
    return np.array([
        [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
        [ (1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
        [ (1+eta)*(1-zeta),  (1+xi)*(1-zeta), -(1+xi)*(1+eta)],
        [-(1+eta)*(1-zeta),  (1-xi)*(1-zeta), -(1-xi)*(1+eta)],
        [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta),  (1-xi)*(1-eta)],
        [ (1-eta)*(1+zeta), -(1+xi)*(1+zeta),  (1+xi)*(1-eta)],
        [ (1+eta)*(1+zeta),  (1+xi)*(1+zeta),  (1+xi)*(1+eta)],
        [-(1+eta)*(1+zeta),  (1-xi)*(1+zeta),  (1-xi)*(1+eta)]
    ]) / 8.0

def rigidez_hexaedro(E, v, XYZ_e, pG, pesos):
    D = E / ((1 + v) * (1 - 2*v)) * np.array([
        [1-v, v, v, 0, 0, 0], [v, 1-v, v, 0, 0, 0], [v, v, 1-v, 0, 0, 0],
        [0, 0, 0, (1-2*v)/2, 0, 0], [0, 0, 0, 0, (1-2*v)/2, 0], [0, 0, 0, 0, 0, (1-2*v)/2]
    ])
    ke = np.zeros((24,24))
    for (xi, eta, zeta), w in zip(pG, pesos):
        dN_nat = B_matrix_hex(xi, eta, zeta)
        J = dN_nat.T @ XYZ_e
        detJ = np.linalg.det(J)
        if detJ <= 0: continue
        dN_glob = dN_nat @ np.linalg.inv(J)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, i*3] = dN_glob[i, 0]; B[1, i*3+1] = dN_glob[i, 1]; B[2, i*3+2] = dN_glob[i, 2]
            B[3, i*3] = dN_glob[i, 1]; B[3, i*3+1] = dN_glob[i, 0]
            B[4, i*3+1] = dN_glob[i, 2]; B[4, i*3+2] = dN_glob[i, 1]
            B[5, i*3] = dN_glob[i, 2]; B[5, i*3+2] = dN_glob[i, 0]
        ke += B.T @ D @ B * detJ * w
    return ke

def calcular_volume_hexaedro(XYZ_e, pG, pesos):
    return sum(np.linalg.det(B_matrix_hex(xi, eta, zeta).T @ XYZ_e) * w for (xi, eta, zeta), w in zip(pG, pesos))

def calcular_forcas_protensao_com_perdas(IE_trelica, XYZ, sigma_p0, A_trelica, E_trelica, E_solido, L, f, bw, h, z_offset, params_perdas):
    detalhamento_perdas = {k: [] for k in ['Posicao_x_cm', 'Perda_Tensao_Atrito_kN_cm2', 'Perda_Tensao_Ancoragem_kN_cm2', 'Perda_Tensao_CS_Flu_kN_cm2', 'Perda_Tensao_Relaxacao_kN_cm2']}
    P0 = sigma_p0 * A_trelica
    forcas_finais = np.zeros(len(IE_trelica))
    Ac, Ic, alpha_e = bw * h, (bw * h**3) / 12, E_trelica / E_solido
    
    g_ancoragem, xr, delta_P_max_ancoragem = params_perdas['g_ancoragem'], 0, 0
    if g_ancoragem > 0:
        delta_p_inicial = P0 * (params_perdas['mu'] * (8 * f / L**2) + params_perdas['k_cm'])
        if delta_p_inicial > 1e-9:
            xr = np.sqrt((g_ancoragem