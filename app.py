# -*- coding: utf-8 -*-
"""
Calculadora de Viga Protendida - MEF 3D (Sólidos + Treliça)
Versão Segura e Descompactada
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

    coords = []
    node_id = {}
    node_index = 0
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
    dx = L / div_x
    dy = bw / div_y
    j = faixa_y
    indices_trelica = []
    coordenadas_trelica = []
    for i in range(div_x + 1):
        x = i * dx
        y = j * dy
        z = round((-4 * f * x * (L - x)) / (L ** 2) + z_offset, 6)
        k = None
        for kk, zz in enumerate(z_layers):
            if abs(zz - z) < 1e-6:
                k = kk
                break
        if k is None:
            raise ValueError(f"z exato {z} não encontrado em z_layers.")
        indices_trelica.append(node_id[(i, j, k)])
        coordenadas_trelica.append((x, y, z))
    return indices_trelica, coordenadas_trelica

def formar_matriz_B_hexa(dN_glob):
    B = np.zeros((6, 24))
    for i in range(8):
        B[0, i*3] = dN_glob[i, 0]
        B[1, i*3+1] = dN_glob[i, 1]
        B[2, i*3+2] = dN_glob[i, 2]
        B[3, i*3] = dN_glob[i, 1]
        B[3, i*3+1] = dN_glob[i, 0]
        B[4, i*3+1] = dN_glob[i, 2]
        B[4, i*3+2] = dN_glob[i, 1]
        B[5, i*3] = dN_glob[i, 2]
        B[5, i*3+2] = dN_glob[i, 0]
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
    stresses_all_elements = []
    strains_all_elements = []
    U_total_reshaped = U_total.reshape((-1,3))
    for e_idx, nos_indices_globais in enumerate(elementos_solidos):
        XYZ_e_final = XYZ_final[nos_indices_globais]
        u_e_flat = U_total_reshaped[nos_indices_globais].ravel()
        stresses_gauss_points = []
        strains_gauss_points = []
        for xi, eta, zeta in pG:
            dN_nat = B_matrix_hex_func(xi, eta, zeta)
            J = dN_nat.T @ XYZ_e_final
            try:
                if np.linalg.det(J) <= 1e-9:
                    raise ValueError
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
    epsilons_trel = []
    sigmas_trel = []
    forces_trel = []
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
        [1-v, v, v, 0, 0, 0], 
        [v, 1-v, v, 0, 0, 0], 
        [v, v, 1-v, 0, 0, 0],
        [0, 0, 0, (1-2*v)/2, 0, 0], 
        [0, 0, 0, 0, (1-2*v)/2, 0], 
        [0, 0, 0, 0, 0, (1-2*v)/2]
    ])
    ke = np.zeros((24,24))
    for (xi, eta, zeta), w in zip(pG, pesos):
        dN_nat = B_matrix_hex(xi, eta, zeta)
        J = dN_nat.T @ XYZ_e
        detJ = np.linalg.det(J)
        if detJ <= 0: 
            continue
        dN_glob = dN_nat @ np.linalg.inv(J)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, i*3] = dN_glob[i, 0]
            B[1, i*3+1] = dN_glob[i, 1]
            B[2, i*3+2] = dN_glob[i, 2]
            B[3, i*3] = dN_glob[i, 1]
            B[3, i*3+1] = dN_glob[i, 0]
            B[4, i*3+1] = dN_glob[i, 2]
            B[4, i*3+2] = dN_glob[i, 1]
            B[5, i*3] = dN_glob[i, 2]
            B[5, i*3+2] = dN_glob[i, 0]
        ke += B.T @ D @ B * detJ * w
    return ke

def calcular_volume_hexaedro(XYZ_e, pG, pesos):
    volume = 0.0
    for (xi, eta, zeta), w in zip(pG, pesos):
        J = B_matrix_hex(xi, eta, zeta).T @ XYZ_e
        volume += np.linalg.det(J) * w
    return volume

def calcular_forcas_protensao_com_perdas(IE_trelica, XYZ, sigma_p0, A_trelica, E_trelica, E_solido, L, f, bw, h, z_offset, params_perdas):
    detalhamento_perdas = {
        'Posicao_x_cm': [], 
        'Perda_Tensao_Atrito_kN_cm2': [], 
        'Perda_Tensao_Ancoragem_kN_cm2': [], 
        'Perda_Tensao_CS_Flu_kN_cm2': [], 
        'Perda_Tensao_Relaxacao_kN_cm2': []
    }
    P0 = sigma_p0 * A_trelica
    forcas_finais = np.zeros(len(IE_trelica))
    Ac = bw * h
    Ic = (bw * h**3) / 12
    alpha_e = E_trelica / E_solido
    
    g_ancoragem = params_perdas['g_ancoragem']
    xr = 0
    delta_P_max_ancoragem = 0
    if g_ancoragem > 0:
        delta_p_inicial = P0 * (params_perdas['mu'] * (8 * f / L**2) + params_perdas['k_cm'])
        if delta_p_inicial > 1e-9:
            xr = np.sqrt((g_ancoragem * E_trelica * A_trelica) / delta_p_inicial)
            delta_P_max_ancoragem = 2 * delta_p_inicial * xr

    for e_idx, (no_i, no_j) in enumerate(IE_trelica):
        x_medio = (XYZ[no_i, 0] + XYZ[no_j, 0]) / 2.0
        
        expoente = params_perdas['mu'] * (8 * f / (L**2)) * x_medio + params_perdas['k_cm'] * x_medio
        forca_apos_atrito = P0 * np.exp(-expoente)
        perda_forca_atrito = P0 - forca_apos_atrito

        perda_ancoragem_neste_ponto = 0.0
        if g_ancoragem > 0 and x_medio < xr:
            perda_ancoragem_neste_ponto = delta_P_max_ancoragem * (1 - x_medio / xr)
            
        forca_apos_perdas_imediatas = forca_apos_atrito - perda_ancoragem_neste_ponto

        z_cabo = (-4*f/L**2)*x_medio**2 + (4*f/L)*x_medio + z_offset
        e_x = abs((h / 2) - z_cabo)
        
        peso_solido = params_perdas.get('peso_especifico_solido', 0)
        pressao = params_perdas['pressao_superficie']
        carga_distribuida = peso_solido * Ac + pressao * bw
        M_cargas_permanentes = (carga_distribuida * x_medio / 2) * (L - x_medio)
        
        F_pontual = params_perdas['Forca_pontual']
        if x_medio <= L / 2:
            M_cargas_permanentes += (F_pontual * x_medio) / 2
        else:
            M_cargas_permanentes += (F_pontual * (L - x_medio)) / 2
        
        W_cp_x = abs(Ic / 1e-9) if e_x < 1e-9 else abs(Ic / e_x)
        k_x = alpha_e * A_trelica * (1/Ac + e_x**2 / Ic)

        termo_tensao_concreto = (forca_apos_perdas_imediatas / Ac) + ((M_cargas_permanentes - (forca_apos_perdas_imediatas * e_x)) / W_cp_x)
        numerador_cs = (termo_tensao_concreto * (params_perdas['phi_inf'] / E_solido)) + params_perdas['eps_cs_inf']
        denominador_cs = 1 + k_x * (1 + 0.5 * params_perdas['phi_inf'])
        delta_sigma_p_cs = (numerador_cs / denominador_cs) * E_trelica
        
        sigma_pg2 = forca_apos_perdas_imediatas / A_trelica
        delta_sigma_relaxacao = params_perdas['psi_inf_final'] * (sigma_pg2 - 2 * delta_sigma_p_cs)
        
        forca_final = forca_apos_perdas_imediatas - (delta_sigma_p_cs + delta_sigma_relaxacao) * A_trelica
        forcas_finais[e_idx] = max(0, forca_final)

        detalhamento_perdas['Posicao_x_cm'].append(x_medio)
        detalhamento_perdas['Perda_Tensao_Atrito_kN_cm2'].append(perda_forca_atrito / A_trelica)
        detalhamento_perdas['Perda_Tensao_Ancoragem_kN_cm2'].append(perda_ancoragem_neste_ponto / A_trelica)
        detalhamento_perdas['Perda_Tensao_CS_Flu_kN_cm2'].append(delta_sigma_p_cs)
        detalhamento_perdas['Perda_Tensao_Relaxacao_kN_cm2'].append(delta_sigma_relaxacao)

    df_detalhes_perdas = pd.DataFrame(detalhamento_perdas)
    df_detalhes_perdas.index.name = 'Elemento_Treliça_ID'
    return {"forcas_finais": forcas_finais, "detalhamento_perdas": df_detalhes_perdas}

def executar_analise(L=600, bw=20, h=60, f=8, z_offset=31, div_x=30, div_y=2, 
                     sigma_protensao=150, A_trelica=8, E_trelica=20000, E_solido=3067, 
                     v_solido=0.2, peso_especifico_solido=25e-6, pressao_superficie=0.0005, 
                     Forca_pontual=50, faixa_y=1):
    if div_x % 2 != 0: 
        div_x += 1
        
    params_perdas = {
        'mu': 0.05, 
        'k_cm': 3e-6, 
        'g_ancoragem': 0.6, 
        'peso_especifico_solido': peso_especifico_solido, 
        'phi_inf': 2.2, 
        'eps_cs_inf': 0.00033, 
        'psi_inf_final': 0.0733975, 
        'pressao_superficie': pressao_superficie, 
        'Forca_pontual': Forca_pontual
    }

    coords, node_id, z_layers = gerar_nos_solido(L, bw, h, f, z_offset, div_x, div_y)
    elementos_solidos = gerar_elementos_solido(div_x, div_y, z_layers, node_id)
    indices_trelica, nos_trelica = gerar_nos_trelica_indices(L, f, div_x, z_offset, faixa_y, bw, div_y, z_layers, node_id)
    
    IE_trelica = []
    for i in range(len(indices_trelica) - 1):
        IE_trelica.append([indices_trelica[i], indices_trelica[i + 1]])
    IE_trelica = np.array(IE_trelica)
    
    XYZ = np.array(coords)
    nno = XYZ.shape[0]
    ndof = 3 * nno

    F3_pontual = np.zeros(ndof)
    distancias = np.linalg.norm(XYZ - np.array([L / 2.0, bw / 2.0, h]), axis=1)
    no_central_superior = np.argmin(distancias)
    F3_pontual[no_central_superior * 3 + 2] = -Forca_pontual

    dofs_restritos = []
    for i_node in range(nno):
        if XYZ[i_node, 0] == 0 and XYZ[i_node, 2] == 0:
            dofs_restritos.extend([i_node*3, i_node*3+1, i_node*3+2])
        elif XYZ[i_node, 2] == 0 and (XYZ[i_node, 0] == 0 or XYZ[i_node, 0] == L):
            dofs_restritos.extend([i_node*3+1, i_node*3+2])
    dofs_restritos = sorted(list(set(dofs_restritos)))
    dofs_livres = np.setdiff1d(np.arange(ndof), dofs_restritos)

    pG = np.array([
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], 
        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    ]) / np.sqrt(3)
    pesos = np.ones(8)

    F_pp = np.zeros(ndof)
    for nos_element_solido in elementos_solidos:
        volume_e = calcular_volume_hexaedro(XYZ[nos_element_solido], pG, pesos)
        if volume_e > 0:
            forca_nodal = (peso_especifico_solido * volume_e) / 8.0
            for no_global in nos_element_solido: 
                F_pp[no_global * 3 + 2] -= forca_nodal

    F_superficie = np.zeros(ndof)
    if nno > 0:
        z_max = np.max(XYZ[:, 2])
        for nos_globais in elementos_solidos:
            face_sup = [nos_globais[4], nos_globais[5], nos_globais[6], nos_globais[7]]
            todos_no_topo = True
            for no in face_sup:
                if abs(XYZ[no, 2] - z_max) > 1e-6:
                    todos_no_topo = False
                    break
            if todos_no_topo:
                dx_face = XYZ[nos_globais[5], 0] - XYZ[nos_globais[4], 0]
                dy_face = XYZ[nos_globais[7], 1] - XYZ[nos_globais[4], 1]
                area = abs(dx_face * dy_face)
                forca_no = (pressao_superficie * area) / 4.0
                for no in face_sup: 
                    F_superficie[no * 3 + 2] -= forca_no

    resultados_protensao = calcular_forcas_protensao_com_perdas(
        IE_trelica, XYZ, sigma_protensao, A_trelica, E_trelica, E_solido, L, f, bw, h, z_offset, params_perdas
    )
    
    F2 = np.zeros(ndof)
    for e_idx, (no_i, no_j) in enumerate(IE_trelica):
        cx, cy, cz, _ = cosdirL(e_idx, IE_trelica, XYZ)
        F_axial = resultados_protensao["forcas_finais"][e_idx]
        T_vetor = np.array([cx, cy, cz]) * F_axial
        
        forca_nodal = np.concatenate([T_vetor, -T_vetor])
        dofs_globais = [no_i*3, no_i*3+1, no_i*3+2, no_j*3, no_j*3+1, no_j*3+2]
        
        for i in range(6):
            F2[dofs_globais[i]] += forca_nodal[i]

    def solver(XYZ_ref, F_load):
        K = np.zeros((ndof, ndof))
        for e, (ni, nj) in enumerate(IE_trelica):
            cx, cy, cz, L_elem = cosdirL(e, IE_trelica, XYZ_ref)
            ke = rigidezElemento(E_trelica, A_trelica, L_elem, cx, cy, cz)
            dofs = [ni*3, ni*3+1, ni*3+2, nj*3, nj*3+1, nj*3+2]
            for i in range(6):
                for j in range(6): 
                    K[dofs[i], dofs[j]] += ke[i, j]
        for nos_solido in elementos_solidos:
            XYZ_solido = XYZ_ref[nos_solido]
            ke_s = rigidez_hexaedro(E_solido, v_solido, XYZ_solido, pG, pesos)
            dofs_s = []
            for no in nos_solido:
                dofs_s.extend([no*3, no*3+1, no*3+2])
            for i in range(24):
                for j in range(24): 
                    K[dofs_s[i], dofs_s[j]] += ke_s[i, j]
        U = np.zeros(ndof)
        K_livre = K[np.ix_(dofs_livres, dofs_livres)]
        F_livre = F_load[dofs_livres]
        U[dofs_livres] = np.linalg.solve(K_livre, F_livre)
        return U

    U_pp = solver(XYZ, F_pp)
    XYZ_apos_pp = XYZ + U_pp.reshape((-1, 3))
    
    U_prot = solver(XYZ_apos_pp, F2)
    XYZ_apos_prot = XYZ_apos_pp + U_prot.reshape((-1, 3))
    
    U_ext = solver(XYZ_apos_prot, F_superficie + F3_pontual)
    U_total_final = U_pp + U_prot + U_ext

    x_meio_vao = L / 2.0
    indices_meio = np.where(np.isclose(XYZ[:, 0], x_meio_vao))[0]
    flecha_no_centroide = 0.0
    if indices_meio.size > 0:
        coords_yz = XYZ[indices_meio, 1:3]
        distancias_centroide = np.linalg.norm(coords_yz - np.array([bw/2.0, h/2.0]), axis=1)
        idx_central = indices_meio[np.argmin(distancias_centroide)]
        flecha_no_centroide = U_total_final.reshape((-1, 3))[idx_central, 2]

    _, tensoes_trel_final, _ = calcular_esforcos_trelica(IE_trelica, XYZ, U_total_final, E_tre
