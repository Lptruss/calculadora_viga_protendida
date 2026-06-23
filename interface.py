# -*- coding: utf-8 -*-
"""
Interface Web com Download de Planilha e PDF
"""

import streamlit as st
import app as tcc

st.set_page_config(page_title="Calculadora MEF - Viga Protendida", layout="centered")

st.title("🧮 Calculadora de Viga Protendida Biapoiada")
st.markdown("Análise numérica por Elementos Finitos com cálculo automático de perdas.")

st.sidebar.header("📐 Geometria da Viga")
L = st.sidebar.number_input("Comprimento da Viga L (cm)", value=600)
bw = st.sidebar.number_input("Largura da Seção bw (cm)", value=20)
h = st.sidebar.number_input("Altura da Viga h (cm)", value=60)

st.sidebar.header("⛓️ Parâmetros de Protensão")
f = st.sidebar.number_input("Flecha do Cabo f (cm)", value=8)
z_offset = st.sidebar.number_input("Ancoragem z_offset (cm)", value=31)
sigma_protensao = st.sidebar.number_input("Tensão de Protensão σ (kN/cm²)", value=150)
A_trelica = st.sidebar.number_input("Área do Cabo A (cm²)", value=8)

st.sidebar.header("💥 Carregamentos Externos")
pressao_superficie = st.sidebar.number_input("Carga Distribuída (kN/cm²)", value=0.0005, format="%.5f")
Forca_pontual = st.sidebar.number_input("Força Pontual no Meio (kN)", value=50)

st.sidebar.header("🕸️ Refinamento da Malha MEF")
div_x = st.sidebar.slider("Divisões em X (Comprimento)", min_value=10, max_value=60, value=30, step=2)

if st.button("🚀 Executar Análise Estrutural"):
    with st.spinner("Resolvendo equações e gerando relatórios..."):
        try:
            flecha, arquivo_excel, arquivo_pdf = tcc.executar_analise(
                L=L, bw=bw, h=h, f=f, z_offset=z_offset, div_x=div_x,
                sigma_protensao=sigma_protensao, A_trelica=A_trelica,
                pressao_superficie=pressao_superficie, Forca_pontual=Forca_pontual
            )
            
            st.success("🎯 Análise Concluída!")
            st.metric(label="📊 Flecha no Centro da Viga", value=f"{flecha:.4f} cm")
            
            col1, col2 = st.columns(2)
            with col1:
                with open(arquivo_excel, "rb") as f_excel:
                    st.download_button(
                        label="📥 Baixar Planilha (Excel)",
                        data=f_excel,
                        file_name="Resultados_MEF_Viga.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            with col2:
                with open(arquivo_pdf, "rb") as f_pdf:
                    st.download_button(
                        label="📄 Baixar Relatório (PDF)",
                        data=f_pdf,
                        file_name="Relatorio_Viga_Protendida.pdf",
                        mime="application/pdf"
                    )
                
        except Exception as erro:
            st.error(f"Erro numérico: {erro}")
