# -*- coding: utf-8 -*-
"""
Calculadora de Viga Protendida - MEF 3D (Sólidos + Treliça)
Versão Otimizada com Relatório em PDF (fpdf)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fpdf import FPDF
import os
from datetime import datetime, timedelta, timezone

class RelatorioPDF(FPDF):
    def header(self):
        # 1. Imprime a Página e a Data no topo (em itálico)
        self.set_font('Arial', 'I', 9)
        fuso_br = timezone(timedelta(hours=-3)) # Ajusta para o fuso do Brasil
        data_hora = datetime.now(fuso_br).strftime("%d/%m/%Y %H:%M")
        texto_topo = f"Página {self.page_no()} | Gerado em {data_hora}"
        self.cell(0, 5, texto_topo, border=0, ln=True, align='L')
        self.ln(3) # Dá um pequeno espaço
        
        # 2. Imprime o Título Principal
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Relatorio de Calculo - Viga de Concreto Protendido', border=1, align='C')
        self.ln(15)

def desenhar_esquema(pdf, bw, h, z_offset, tensao_max, tensao_min):
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '5. Esquema da Secao e Diagrama de Tensoes', ln=True)
    
    escala = 0.5
    x_base = 60
    y_base = pdf.get_y() + 10
    
    # 1. Desenha a Viga
    pdf.rect(x_base, y_base, bw * escala, h * escala)
    y_cabo = y_base + (h - z_offset) * escala
    pdf.set_draw_color(200, 0, 0)
    pdf.line(x_base, y_cabo, x_base + (bw * escala), y_cabo)
    pdf.set_draw_color(0, 0, 0)
    
    # 2. Desenha o Diagrama de Tensões (ao lado)
    x_diag = x_base + bw * escala + 30
    pdf.line(x_diag, y_base, x_diag, y_base + h * escala) # Eixo central
    
    # Desenha o triângulo/trapézio de tensões simplificado
    pdf.set_draw_color(0, 0, 200) # Azul para as tensões
    pdf.line(x_diag, y_base, x_diag + 20, y_base) # Topo
    pdf.line(x_diag, y_base + h * escala, x_diag + 20, y_base + h * escala) # Base
    pdf.line(x_diag + 20, y_base, x_diag, y_base + h * escala) # Diagonal de tensão
    
    # Legendas
    pdf.set_font('Arial', '', 8)
    pdf.text(x_diag + 22, y_base + 3, 'Tensao Max')
    pdf.text(x_diag + 22, y_base + h * escala, 'Tensao Min')
    
    pdf.ln(h * escala + 15)

def gerar_pdf(L, bw, h, E_sol, Ap, P0, contraflecha, flecha_total, tensao_efetiva, Msd, Mrd, x, x_d, status, df_perdas):
    pdf = RelatorioPDF()
    pdf.add_page()
    
    # 1. Dados Geométricos e Materiais
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Dados Geometricos e Materiais', ln=True)
    pdf.set_font('Arial', '', 11)
    
    largura_col1 = 80
    largura_col2 = 110
    
    pdf.cell(largura_col1, 8, 'Comprimento (L)', border=1)
    pdf.cell(largura_col2, 8, f'{L} cm', border=1, ln=True)
    pdf.cell(largura_col1, 8, 'Secao (bw x h)', border=1)
    pdf.cell(largura_col2, 8, f'{bw} x {h} cm', border=1, ln=True)
    pdf.cell(largura_col1, 8, 'FCK Concreto (Aproximado)', border=1)
    pdf.cell(largura_col2, 8, f'{E_sol/100:.2f} GPa (E_sol)', border=1, ln=True)
    pdf.cell(largura_col1, 8, 'Area de Aco (Ap)', border=1)
    pdf.cell(largura_col2, 8, f'{Ap} cm2', border=1, ln=True)
    pdf.cell(largura_col1, 8, 'Forca de Protensao Inicial', border=1)
    pdf.cell(largura_col2, 8, f'{P0:.2f} kN', border=1, ln=True)
    
    pdf.ln(10)
    
    # 2. ELS
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Verificacao de Estados Limites de Servico (ELS)', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Contraflecha de Protensao: {contraflecha:.4f} cm', ln=True)
    pdf.cell(0, 8, f'Flecha Total (Cargas + Protensao): {flecha_total:.4f} cm', ln=True)
    pdf.cell(0, 8, f'Tensao Efetiva Media no Cabo: {tensao_efetiva:.2f} kN/cm2', ln=True)
    
    pdf.ln(5)
    
    # 3. ELU
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Verificacao de Estado Limite Ultimo (ELU)', border=0, ln=True, fill=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Momento Solicitante de Calculo (Msd): {Msd:.2f} kN.cm', ln=True)
    pdf.cell(0, 8, f'Momento Resistente de Calculo (Mrd): {Mrd:.2f} kN.cm', ln=True)
    pdf.cell(0, 8, f'Linha Neutra (x): {x:.2f} cm | x/d: {x_d:.3f}', ln=True)
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    if status == "APROVADO":
        pdf.set_text_color(0, 128, 0)
    else:
        pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 10, f'STATUS FINAL: {status}', ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    
    pdf.ln(10)

    # Chama o desenho da viga
    desenhar_esquema(pdf, bw, h, z_offset)
    
    # 4. Detalhamento
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '4. Detalhamento das Perdas de Protensao (kN/cm2)', ln=True)
    
    pdf.set_font('Arial', 'B', 10)
    w = [25, 30, 30, 35, 35, 35]
    headers = ['X (cm)', 'P. Atrito', 'P. Ancora', 'P. Flu/Ret', 'P. Relax', 'T. Final']
    for i in range(len(headers)):
        pdf.cell(w[i], 8, headers[i], border=1, align='C')
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    for index, row in df_perdas.iterrows():
        pdf.cell(w[0], 8, f"{row['Posicao_x_cm']:.1f}", border=1, align='C')
        pdf.cell(w[1], 8, f"{row['Perda_Tensao_Atrito_kN_cm2']:.2f}", border=1, align='C')
        pdf.cell(w[2], 8, f"{row['Perda_Tensao_Ancoragem_kN_cm2']:.2f}", border=1, align='C')
        pdf.cell(w[3], 8, f"{row['Perda_Tensao_CS_Flu_kN_cm2']:.2f}", border=1, align='C')
        pdf.cell(w[4], 8, f"{row['Perda_Tensao_Relaxacao_kN_cm2']:.2f}", border=1, align='C')
        pdf.cell(w[5], 8, f"{row['Tensao_Final_Efetiva_kN_cm2']:.2f}", border=1, align='C')
        pdf.ln()
        
    caminho_pdf = "Relatorio_Viga_Protendida.pdf"
    pdf.output(caminho_pdf)
    return caminho_pdf

def gerar_nos_solido(L, bw, h, f, z_offset, div_x, div_y):
    dx, dy = L / div_x, bw / div_y
    z_inf_max = z_offset - f
    z_layers_inf = [round(i * (z_inf_max / (div_x / 2)), 6) for i in range(int(div_x / 2) + 1)]
    z_layers_sup = [round(z_offset + i * ((h - z_offset) / (div_x / 2)), 6) for i in range(1, int(div_x / 2) + 1)]

    z_layers_parabolica = []
    for i in range(div_x + 1):
        z_trelica = round((-4 * f * (i * dx) * (L - (i * dx))) / (L ** 2) + z_offset, 6)
        if z_inf_max - 1e-6 <= z_trelica <= z_offset + 1e-6:
            z_layers_parabolica.append(z_trelica)

    z_layers = sorted(set(z_layers_inf + z_layers_parabolica + z_layers_sup))
    if abs(z_layers[-1] - h) > 1e-6: z_layers.append(round(h, 6))

    coords, node_id, node_index = [], {}, 0
    for k, z in enumerate(z_layers):
        for j in range(div_y + 1):
            for i in range(div_x + 1):
                coords.append((i * dx, j * dy, z))
                node_id[(i, j, k)] = node_index
                node_index += 1
    return coords, node_id, z_layers

def gerar_elementos_solido(div_x, div_y, z_layers, node_id):
    elementos = []
    for k in range(len(z_layers) - 1):
        for j in range(div_y):
            for i in range(div_x):
                elementos.append([node_id[(i,j,k)], node_id[(i+1,j,k)], node_id[(i+1,j+1,k)], node_id[(i,j+1,k)],
                                  node_id[(i,j,k+1)], node_id[(i+1,j,k+1)], node_id[(i+1,j+1,k+1)], node_id[(i,j+1,k+1)]])
    return elementos

def gerar_nos_trelica_indices(L, f, div_x, z_offset, faixa_y, bw, div_y, z_layers, node_id):
    dx, dy = L / div_x, bw / div_y
    indices_trelica, coordenadas_trelica = [], []
    for i in range(div_x + 1):
        x, z = i * dx, round((-4 * f * (i * dx) * (L - i * dx)) / (L ** 2) + z_offset, 6)
        k = next(kk for kk, zz in enumerate(z_layers) if abs(zz - z) < 1e-6)
        indices_trelica.append(node_id[(i, faixa_y, k)])
        coordenadas_trelica.append((x, faixa_y * dy, z))
    return indices_trelica, coordenadas_trelica

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
        if np.linalg.det(J) <= 0: continue
        dN_glob = dN_nat @ np.linalg.inv(J)
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, i*3], B[1, i*3+1], B[2, i*3+2] = dN_glob[i, 0], dN_glob[i, 1], dN_glob[i, 2]
            B[3, i*3], B[3, i*3+1] = dN_glob[i, 1], dN_glob[i, 0]
            B[4, i*3+1], B[4, i*3+2] = dN_glob[i, 2], dN_glob[i, 1]
            B[5, i*3], B[5, i*3+2] = dN_glob[i, 2], dN_glob[i, 0]
        ke += B.T @ D @ B * np.linalg.det(J) * w
    return ke

def calcular_volume_hexaedro(XYZ_e, pG, pesos):
    return sum(np.linalg.det(B_matrix_hex(xi, eta, zeta).T @ XYZ_e) * w for (xi, eta, zeta), w in zip(pG, pesos))

def calcular_forcas_protensao_com_perdas(IE_trelica, XYZ, sigma_p0, A_trelica, E_trelica, E_solido, L, f, bw, h, z_offset, params_perdas):
    detalhamento_perdas = {'Posicao_x_cm': [], 'Perda_Tensao_Atrito_kN_cm2': [], 'Perda_Tensao_Ancoragem_kN_cm2': [], 'Perda_Tensao_CS_Flu_kN_cm2': [], 'Perda_Tensao_Relaxacao_kN_cm2': []}
    P0 = sigma_p0 * A_trelica
    forcas_finais = np.zeros(len(IE_trelica))
    Ac, Ic, alpha_e = bw * h, (bw * h**3) / 12, E_trelica / E_solido
    
    xr, delta_P_max_ancoragem = 0, 0
    if params_perdas['g_ancoragem'] > 0:
        delta_p_inicial = P0 * (params_perdas['mu'] * (8 * f / L**2) + params_perdas['k_cm'])
        if delta_p_inicial > 1e-9:
            xr = np.sqrt((params_perdas['g_ancoragem'] * E_trelica * A_trelica) / delta_p_inicial)
            delta_P_max_ancoragem = 2 * delta_p_inicial * xr

    for e_idx, (no_i, no_j) in enumerate(IE_trelica):
        x_medio = (XYZ[no_i, 0] + XYZ[no_j, 0]) / 2.0
        forca_apos_atrito = P0 * np.exp(-(params_perdas['mu'] * (8 * f / (L**2)) * x_medio + params_perdas['k_cm'] * x_medio))
        perda_ancoragem = delta_P_max_ancoragem * (1 - x_medio / xr) if params_perdas['g_ancoragem'] > 0 and x_medio < xr else 0.0
        forca_apos_perdas_imediatas = forca_apos_atrito - perda_ancoragem

        e_x = abs((h / 2) - ((-4*f/L**2)*x_medio**2 + (4*f/L)*x_medio + z_offset))
        M_cargas = (((params_perdas['peso_especifico_solido'] * Ac + params_perdas['pressao_superficie'] * bw) * x_medio / 2) * (L - x_medio))
        M_cargas += (params_perdas['Forca_pontual'] * x_medio) / 2 if x_medio <= L / 2 else (params_perdas['Forca_pontual'] * (L - x_medio)) / 2
        
        k_x = alpha_e * A_trelica * (1/Ac + e_x**2 / (abs(Ic / e_x) if e_x > 1e-9 else abs(Ic / 1e-9)))
        termo_tensao_concreto = (forca_apos_perdas_imediatas / Ac) + ((M_cargas - (forca_apos_perdas_imediatas * e_x)) / (abs(Ic / e_x) if e_x > 1e-9 else abs(Ic / 1e-9)))
        
        delta_sigma_p_cs = (((termo_tensao_concreto * (params_perdas['phi_inf'] / E_solido)) + params_perdas['eps_cs_inf']) / (1 + k_x * (1 + 0.5 * params_perdas['phi_inf']))) * E_trelica
        delta_sigma_relaxacao = params_perdas['psi_inf_final'] * ((forca_apos_perdas_imediatas / A_trelica) - 2 * delta_sigma_p_cs)
        
        forcas_finais[e_idx] = max(0, forca_apos_perdas_imediatas - (delta_sigma_p_cs + delta_sigma_relaxacao) * A_trelica)
        detalhamento_perdas['Posicao_x_cm'].append(x_medio)
        detalhamento_perdas['Perda_Tensao_Atrito_kN_cm2'].append((P0 - forca_apos_atrito) / A_trelica)
        detalhamento_perdas['Perda_Tensao_Ancoragem_kN_cm2'].append(perda_ancoragem / A_trelica)
        detalhamento_perdas['Perda_Tensao_CS_Flu_kN_cm2'].append(delta_sigma_p_cs)
        detalhamento_perdas['Perda_Tensao_Relaxacao_kN_cm2'].append(delta_sigma_relaxacao)

    return {"forcas_finais": forcas_finais, "detalhamento_perdas": pd.DataFrame(detalhamento_perdas)}

def estimar_elu(L, bw, h, f, pressao_superficie, Forca_pontual, A_trelica, sigma_protensao, peso_especifico):
    q = (peso_especifico * bw * h) + (pressao_superficie * bw)
    M_g = (q * L**2) / 8
    M_q = (Forca_pontual * L) / 4
    Msd = 1.4 * (M_g + M_q)
    
    fcd = 30 / 1.4 / 10 
    fpyd = sigma_protensao / 1.15
    T = A_trelica * fpyd
    x = T / (0.68 * bw * fcd) if fcd > 0 else 0
    d = h - (h/2 - f)
    x_d = x / d if d > 0 else 0
    Mrd = T * (d - 0.4 * x)
    status = "APROVADO" if Mrd >= Msd else "REPROVADO"
    return Msd, Mrd, x, x_d, status

def executar_analise(L=600, bw=20, h=60, f=8, z_offset=31, div_x=30, div_y=2, 
                     sigma_protensao=150, A_trelica=8, E_trelica=20000, E_solido=3067, 
                     v_solido=0.2, peso_especifico_solido=25e-6, pressao_superficie=0.0005, 
                     Forca_pontual=50, faixa_y=1):
    if div_x % 2 != 0: div_x += 1
    params_perdas = {'mu': 0.05, 'k_cm': 3e-6, 'g_ancoragem': 0.6, 'peso_especifico_solido': peso_especifico_solido, 'phi_inf': 2.2, 'eps_cs_inf': 0.00033, 'psi_inf_final': 0.0733975, 'pressao_superficie': pressao_superficie, 'Forca_pontual': Forca_pontual}

    coords, node_id, z_layers = gerar_nos_solido(L, bw, h, f, z_offset, div_x, div_y)
    elementos_solidos = gerar_elementos_solido(div_x, div_y, z_layers, node_id)
    indices_trelica, _ = gerar_nos_trelica_indices(L, f, div_x, z_offset, faixa_y, bw, div_y, z_layers, node_id)
    IE_trelica = np.array([[indices_trelica[i], indices_trelica[i + 1]] for i in range(len(indices_trelica) - 1)])
    XYZ = np.array(coords)
    ndof = 3 * XYZ.shape[0]

    F3_pontual, F_superficie, F_pp = np.zeros(ndof), np.zeros(ndof), np.zeros(ndof)
    F3_pontual[np.argmin(np.linalg.norm(XYZ - np.array([L / 2.0, bw / 2.0, h]), axis=1)) * 3 + 2] = -Forca_pontual

    dofs_restritos = sorted(list(set(dof for i in range(XYZ.shape[0]) for dof, c in zip([i*3, i*3+1, i*3+2], [XYZ[i,0]==0 and XYZ[i,2]==0, XYZ[i,2]==0 and (XYZ[i,0]==0 or XYZ[i,0]==L), XYZ[i,2]==0 and (XYZ[i,0]==0 or XYZ[i,0]==L)]) if c)))
    dofs_livres = np.setdiff1d(np.arange(ndof), dofs_restritos)
    pG, pesos = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) / np.sqrt(3), np.ones(8)

    for nos in elementos_solidos:
        vol = calcular_volume_hexaedro(XYZ[nos], pG, pesos)
        if vol > 0:
            for no in nos: F_pp[no * 3 + 2] -= (peso_especifico_solido * vol) / 8.0
            face_sup = [nos[4], nos[5], nos[6], nos[7]]
            if all(abs(XYZ[n, 2] - np.max(XYZ[:, 2])) < 1e-6 for n in face_sup):
                area = abs((XYZ[nos[5], 0] - XYZ[nos[4], 0]) * (XYZ[nos[7], 1] - XYZ[nos[4], 1]))
                for no in face_sup: F_superficie[no * 3 + 2] -= (pressao_superficie * area) / 4.0

    resultados_protensao = calcular_forcas_protensao_com_perdas(IE_trelica, XYZ, sigma_protensao, A_trelica, E_trelica, E_solido, L, f, bw, h, z_offset, params_perdas)
    F2 = np.zeros(ndof)
    for e_idx, (no_i, no_j) in enumerate(IE_trelica):
        cx, cy, cz, _ = cosdirL(e_idx, IE_trelica, XYZ)
        T_vetor = np.array([cx, cy, cz]) * resultados_protensao["forcas_finais"][e_idx]
        for i, v in enumerate(np.concatenate([T_vetor, -T_vetor])): F2[[no_i*3, no_i*3+1, no_i*3+2, no_j*3, no_j*3+1, no_j*3+2][i]] += v

    def solver(XYZ_ref, F_load):
        K = sp.lil_matrix((ndof, ndof))
        for e, (ni, nj) in enumerate(IE_trelica):
            cx, cy, cz, L_elem = cosdirL(e, IE_trelica, XYZ_ref)
            ke = rigidezElemento(E_trelica, A_trelica, L_elem, cx, cy, cz)
            dofs = [ni*3, ni*3+1, ni*3+2, nj*3, nj*3+1, nj*3+2]
            for i in range(6):
                for j in range(6): K[dofs[i], dofs[j]] += ke[i, j]
        for nos in elementos_solidos:
            ke_s = rigidez_hexaedro(E_solido, v_solido, XYZ_ref[nos], pG, pesos)
            dofs_s = [no*3+i for no in nos for i in range(3)]
            for i in range(24):
                for j in range(24): K[dofs_s[i], dofs_s[j]] += ke_s[i, j]
        U = np.zeros(ndof)
        U[dofs_livres] = spla.spsolve(K.tocsr()[dofs_livres, :][:, dofs_livres], F_load[dofs_livres])
        return U

    U_pp = solver(XYZ, F_pp)
    U_prot = solver(XYZ + U_pp.reshape((-1, 3)), F2)
    U_ext = solver(XYZ + (U_pp + U_prot).reshape((-1, 3)), F_superficie + F3_pontual)
    U_total = U_pp + U_prot + U_ext

    idx_central = np.where(np.isclose(XYZ[:, 0], L / 2.0))[0]
    idx_central = idx_central[np.argmin(np.linalg.norm(XYZ[idx_central, 1:3] - np.array([bw/2.0, h/2.0]), axis=1))] if idx_central.size > 0 else 0
    flecha_total = U_total.reshape((-1, 3))[idx_central, 2]
    contraflecha = U_prot.reshape((-1, 3))[idx_central, 2]

    df_final = resultados_protensao["detalhamento_perdas"].copy()
    tensoes_trel = [E_trelica * ((np.linalg.norm((XYZ + U_total.reshape((-1, 3)))[nj] - (XYZ + U_total.reshape((-1, 3)))[ni]) - np.linalg.norm(XYZ[nj] - XYZ[ni])) / np.linalg.norm(XYZ[nj] - XYZ[ni])) for ni, nj in IE_trelica]
    df_final['Tensao_Final_Efetiva_kN_cm2'] = sigma_protensao - df_final['Perda_Tensao_Atrito_kN_cm2'] - df_final['Perda_Tensao_Ancoragem_kN_cm2'] - df_final['Perda_Tensao_CS_Flu_kN_cm2'] - df_final['Perda_Tensao_Relaxacao_kN_cm2'] + tensoes_trel
    
    excel_file = 'resultados_viga.xlsx'
    with pd.ExcelWriter(excel_file) as writer:
        pd.DataFrame(np.hstack([XYZ, U_total.reshape((-1, 3))]), columns=['X', 'Y', 'Z', 'Ux', 'Uy', 'Uz']).to_excel(writer, sheet_name='Nos_e_Deslocamentos')
        df_final.to_excel(writer, sheet_name='Resultados_Trelica')

    Msd, Mrd, x, x_d, status = estimar_elu(L, bw, h, f, pressao_superficie, Forca_pontual, A_trelica, sigma_protensao, peso_especifico_solido)
    pdf_file = gerar_pdf(L, bw, h, E_solido, A_trelica, sigma_protensao * A_trelica, contraflecha, flecha_total, df_final['Tensao_Final_Efetiva_kN_cm2'].mean(), Msd, Mrd, x, x_d, status, df_final)

    return flecha_total, excel_file, pdf_file
