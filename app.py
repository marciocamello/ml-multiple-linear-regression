import gradio as gr
import joblib
import pandas as pd

# Carregar modelo
modelo = joblib.load('./modelo_colesterol.pkl')

# Função de predição
def predict(grupo_sanguineo, fumante, nivel_atividade_fisica, idade, peso, altura):
    predicao_individual = {
        'grupo_sanguineo': grupo_sanguineo,
        'fumante': fumante,
        'nivel_atividade_fisica': nivel_atividade_fisica,
        'idade': idade,
        'peso': peso,
        'altura': altura
    }
    predict_df = pd.DataFrame(predicao_individual, index=[1])
    colesterol = modelo.predict(predict_df)
    return colesterol.item(0)

# Criar interface Gradio
colesterol_interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(['O', 'A', 'B', 'AB'], label="Grupo Sanguíneo", value="O"),
        gr.Radio(["Sim", "Não"], label="Fumante", value="Não"),
        gr.Radio(['Alto', 'Moderado', 'Baixo'], label="Nível de Atividade Física", value="Moderado"),
        gr.Slider(20, 80, step=1, label="Idade", value=45),
        gr.Slider(40, 160, step=0.1, label="Peso (kg)", value=94),
        gr.Slider(150, 200, step=1, label="Altura (cm)", value=184)
    ],
    outputs=gr.Number(label="Nível de Colesterol (mg/dL)"),
    title="Predição de Nível de Colesterol",
    description="Insira os dados para prever o nível de colesterol."
)

if __name__ == "__main__":
    colesterol_interface.launch()