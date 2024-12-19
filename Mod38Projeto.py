import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from pycaret.classification import *
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from sklearn import metrics

st.set_page_config(page_icon = ':moneybag:', page_title = 'Credit Scoring', layout = 'wide')

# Função para carregar o modelo

@st.cache_resource

# Função para carregar dados

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# Função de pré-processamento dos dados

def preprocess_data(df):

    df['data_ref'] = pd.to_datetime(df['data_ref'])

    df['tempo_emprego'].fillna(-1, inplace=True)

    if 'qt_pessoas_residencia' in df.columns:
        df.drop(columns=['qt_pessoas_residencia'], inplace=True)

    df.tipo_renda.replace({'Bolsista': 'Bolsista/ Serv. Públ.', 'Servidor público': 'Bolsista/ Serv. Públ.'}, inplace=True)
    df.qtd_filhos.replace({4: 'Mais de 4 filhos', 5: 'Mais de 4 filhos', 7: 'Mais de 4 filhos', 14: 'Mais de 4 filhos'}, inplace=True)
    df['qtd_filhos'] = df['qtd_filhos'].astype(str)
    df.educacao.replace({'Fundamental': 'Ensino Básico', 'Médio': 'Ensino Básico', 'Pós graduação': 'Superior completo'}, inplace=True)
    df.estado_civil.replace({'Casado': 'Casado/ União Estável', 'União': 'Casado/ União Estável', 'Separado': 'Separado/ Viúvo', 'Viúvo': 'Separado/ Viúvo'}, inplace=True)
    df.tipo_residencia.replace({'Comunitário': 'Comunit./ Gov.', 'Governamental': 'Comunit./ Gov.'}, inplace=True)

    return df

# Função para avaliar o modelo

def evaluate_model(model, X, y):
    X['score'] = model.predict(X)
    # Acurácia
    X['score'] = (X['score'] > 0.5).astype(int)  
    acc = metrics.accuracy_score(y, X['score'])

    # AUC
    fpr, tpr, thresholds = metrics.roc_curve(y, X['score'])
    auc = metrics.auc(fpr, tpr)

    # Gini
    gini = 2 * auc - 1

    # KS
    ks = ks_2samp(X['score'][y == 1], X['score'][y != 1]).statistic

    st.markdown(f'##### Acurácia: {acc}')
    st.markdown(f'##### AUC: {auc}')
    st.markdown(f'##### GINI: {gini}')
    st.markdown(f'##### KS: {ks}')

# Configurar a interface do Streamlit

def main():

    st.markdown("<h1 style='text-align: center; color: grey;'>:bank: Aplicação para modelagem de Credit Scoring </h1>", unsafe_allow_html=True)

    # st.title(":bank: Aplicação para modelagem de Credit Scoring")
    
    st.markdown("#### :open_file_folder: Faça upload do arquivo CSV para realizar a modelagem:")

    # Upload do arquivo

    uploaded_file = st.file_uploader(type=["csv"], label = '')

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("#### :mag_right: Pré-visualização dos dados:")
            st.dataframe(df.head(5))

            # Pré-processar os dados

            st.markdown("#### :pencil2: Preparando os dados...")
            df_preprocessed = preprocess_data(df)
            st.write("#### :page_facing_up: Dados processados:")
            st.dataframe(df_preprocessed.sample(5))

            # Inicializar o PyCaret e carregar o modelo

            st.markdown("#### :arrows_counterclockwise: Inicializando PyCaret...")
            final_lightgbm = load_model('Final LightGBM Model')


            # Realizar a escoragem

            st.markdown("#### :crystal_ball: Realizando previsões...")

            try:
                
                predict_model(final_lightgbm, data=df_preprocessed);

                predictions = predict_model(final_lightgbm, data=df_preprocessed)

                st.markdown("#### :white_check_mark: Resultados da modelagem:")
                st.dataframe(predictions.sample(5))

                cm = confusion_matrix(predictions['mau'], predictions['prediction_label'])

                col1, col2 = st.columns(2)

                # Matriz de confusão

                with col1:
                    st.markdown("#### :chart_with_upwards_trend: Matriz de Confusão:")
                    plt.figure(figsize=(4, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                    plt.xlabel('Predições')
                    plt.ylabel('Reais')
                    st.pyplot(plt)

                # Avaliação do Modelo   

                with col2:

                    st.markdown('#### :gem: Avaliação do modelo:')

                    if 'mau' in df.columns:  
                        evaluate_model(final_lightgbm, df_preprocessed.drop(columns = 'mau'), df_preprocessed['mau'])

                # Opção para download dos resultados

                    csv = df_preprocessed.to_csv(index=False)
                    st.download_button(label = '### :floppy_disk: Baixar resultados', data=csv, file_name="resultados_modelagem.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Erro ao realizar a escoragem: {e}")

if __name__ == "__main__":
    main()
