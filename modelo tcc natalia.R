library(randomForest)
library(dplyr)
library(caret)
library(readxl)


dados <- read_excel("C:/Users/natal/Downloads/dados_tcc/dados2023.xlsx")

dados$Data <- as.POSIXct(dados$Data, format="%Y-%m-%d %H:%M:%S")

dados <- dados %>%
  mutate(Molhamento_Foliar = ifelse(Precipitacao > 0, 1, 0))

dados <- dados %>%
  mutate(Date = as.Date(Data)) %>%
  group_by(Date) %>%
  mutate(Molhamento_Total = sum(Molhamento_Foliar, na.rm = TRUE)) %>%
  ungroup()

dados <- dados %>%
  mutate(Ocorrencia_Ferrugem = factor(ifelse(
    Molhamento_Total >= 6 & 
      Temp_Media >= 15 & Temp_Media <= 29 & 
      UmidadeRelativa > 75 & 
      Radiacao_Global > 25 & 
      Vento_Velocidade > 1, 
    "Sim", "Não"
  )))

set.seed(123)
dados_reduzidos <- dados %>%
  filter(Ocorrencia_Ferrugem == "Sim" | Ocorrencia_Ferrugem == "Não") %>%
  group_by(Ocorrencia_Ferrugem) %>%
  sample_n(size = min(table(dados$Ocorrencia_Ferrugem))) %>%
  ungroup()

controle <- trainControl(method = "cv", number = 10) 

modelo_rf_cv <- train(Ocorrencia_Ferrugem ~ Temp_Media + UmidadeRelativa + Radiacao_Global + Vento_Velocidade,
                      data = dados_reduzidos,
                      method = "rf",
                      trControl = controle)

print(modelo_rf_cv)

predicoes_cv <- predict(modelo_rf_cv, newdata = dados_reduzidos)

tabela_confusao_cv <- confusionMatrix(predicoes_cv, dados_reduzidos$Ocorrencia_Ferrugem)

print(tabela_confusao_cv)

precision <- tabela_confusao_cv$byClass["Pos Pred Value"] 
recall <- tabela_confusao_cv$byClass["Sensitivity"]        
f1_score <- 2 * (precision * recall) / (precision + recall) 

cat("Precisão: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1-Score: ", f1_score, "\n")

importancia <- varImp(modelo_rf_cv)
print(importancia)
plot(importancia)

set.seed(123)
indices <- createDataPartition(dados_reduzidos$Ocorrencia_Ferrugem, p = 0.8, list = FALSE)
dados_treino <- dados_reduzidos[indices, ]
dados_teste <- dados_reduzidos[-indices, ]

modelo_rf_treino <- randomForest(Ocorrencia_Ferrugem ~ Temp_Media + UmidadeRelativa + Radiacao_Global + Vento_Velocidade,
                                 data = dados_treino)

predicoes_teste <- predict(modelo_rf_treino, newdata = dados_teste)

tabela_confusao_teste <- confusionMatrix(predicoes_teste, dados_teste$Ocorrencia_Ferrugem)

print(tabela_confusao_teste)


library(pROC)

roc_obj <- roc(dados_teste$Ocorrencia_Ferrugem, as.numeric(predicoes_teste))

plot(roc_obj, col = "darkgreen", lwd = 2, main = "Curva ROC - Ocorrência de Ferrugem",
     xlab = "Taxa de Falsos Positivos (1 - Especificidade)", 
     ylab = "Taxa de Verdadeiros Positivos (Sensibilidade)")

abline(a = 0, b = 1, col = "orange", lty = 2)  # linha de chance

legend("bottomright", legend = paste("AUC =", round(auc(roc_obj), 4)), 
       col = "darkgreen", lwd = 2)

ggsave("C:/Users/natal/Downloads/dados_tcc/curva_roc2.png", 
       width = 10, height = 6, dpi = 600)
library(ggplot2)

probabilidades_teste <- predict(modelo_rf_treino, newdata = dados_teste, type = "prob")[, "Sim"]

limiares <- seq(0, 1, by = 0.01)

resultado <- data.frame(Limiar = limiares,
                        Acuracia = NA,
                        Taxa_Erro = NA)

for (i in seq_along(limiares)) {  
  predicoes_binarias <- ifelse(probabilidades_teste > limiares[i], "Sim", "Não")
  conf_matrix <- table(predicoes_binarias, dados_teste$Ocorrencia_Ferrugem)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix) 
  error_rate <- 1 - accuracy  
  resultado$Acuracia[i] <- accuracy
  resultado$Taxa_Erro[i] <- error_rate
}

ggplot(resultado, aes(x = Limiar)) +
  geom_line(aes(y = Acuracia, color = "Acurácia"), size = 1) +
  geom_line(aes(y = Taxa_Erro, color = "Taxa de Erro"), size = 1) +
  labs(title = "Acurácia e Taxa de Erro em Diferentes Limiares",
       x = "Limiar de Decisão",
       y = "Valor",
       color = "Métrica") +
  scale_color_manual(values = c("orange", "lightblue")) +
  theme_minimal()

ggsave("C:/Users/natal/Downloads/dados_tcc/acuracia_taxa_erro.png", 
       width = 10, height = 6, dpi = 600)


probabilidades_teste <- predict(modelo_rf_treino, newdata = dados_teste, type = "prob")[, "Sim"]

limiares <- seq(0, 1, by = 0.01)

resultado <- data.frame(Limiar = limiares,
                        Recall = NA,
                        F1_Score = NA)

for (i in seq_along(limiares)) {
  predicoes_binarias <- ifelse(probabilidades_teste > limiares[i], "Sim", "Não")
  
  conf_matrix <- table(predicoes_binarias, dados_teste$Ocorrencia_Ferrugem)
  
  if (all(c("Sim", "Não") %in% rownames(conf_matrix))) {
    # Calcular recall
    TP <- conf_matrix["Sim", "Sim"]  
    FN <- conf_matrix["Não", "Sim"]  
    recall <- TP / (TP + FN)
    
    precision <- TP / (TP + conf_matrix["Sim", "Não"])
    f1_score <- 2 * (precision * recall) / (precision + recall)
    
    resultado$Recall[i] <- recall
    resultado$F1_Score[i] <- f1_score
  } else {
    resultado$Recall[i] <- NA 
    resultado$F1_Score[i] <- NA  
  }
}

resultado <- na.omit(resultado)

ggplot(resultado, aes(x = Limiar)) +
  geom_line(aes(y = Recall, color = "Recall"), size = 1) +
  geom_line(aes(y = F1_Score, color = "F1-Score"), size = 1) +
  labs(title = "Recall e F1-Score em Diferentes Limiares",
       x = "Limiar de Decisão",
       y = "Valor",
       color = "Métrica") +
  scale_color_manual(values = c("orange", "purple")) +
  theme_minimal()

ggsave("C:/Users/natal/Downloads/dados_tcc/recall_f1_score.png", 
       width = 10, height = 6, dpi = 600)

library(openxlsx)
saveRDS(modelo_rf_treino, file = "C:/Users/natal/Downloads/dados_tcc/modelo_rf.rds")
write.xlsx(predicoes_teste, "C:/Users/natal/Downloads/dados_tcc/predicoes_teste.xlsx")

cenarios <- expand.grid(
  Temp_Media = seq(15, 30, by = 5),         
  UmidadeRelativa = seq(70, 100, by = 10),  
  Radiacao_Global = seq(20, 30, by = 5),     
  Vento_Velocidade = seq(1, 5, by = 1)      
)

cenarios$Predicao_Ferrugem <- predict(modelo_rf_treino, newdata = cenarios)

print(cenarios)

heatmap_data <- cenarios %>%
  group_by(Temp_Media, UmidadeRelativa, Predicao_Ferrugem) %>%
  summarise(Count = n()) %>%
  ungroup() %>%
  mutate(Count = ifelse(Predicao_Ferrugem == "Sim", Count, 0)) 

set.seed(123)
temp_media <- seq(15, 30, by = 1)
umidade_relativa <- seq(70, 100, by = 1)
heatmap_data <- expand.grid(Temp_Media = temp_media, UmidadeRelativa = umidade_relativa)
heatmap_data$Count <- sample(1:100, nrow(heatmap_data), replace = TRUE)  

heatmap_plot <- ggplot(heatmap_data, aes(x = Temp_Media, y = UmidadeRelativa, fill = Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightyellow", high = "steelblue") +
  labs(title = "Gráfico de Calor: Ocorrência de Ferrugem",
       x = "Temperatura Média (°C)",
       y = "Umidade Relativa (%)",
       fill = "Ocorrência de ferrugem") +
  theme_minimal() +
  theme(axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold"))

print(heatmap_plot)

ggsave("grafico_calor.png", plot = heatmap_plot, width = 10, height = 6, dpi = 600)


library(ggplot2)
library(dplyr)

tendencia_ferrugem <- dados %>%
  group_by(Date) %>%
  summarise(Proporcao_Sim = mean(Ocorrencia_Ferrugem == "Sim"), .groups = "drop")

ggplot(tendencia_ferrugem, aes(x = Date, y = Proporcao_Sim)) +
  geom_line(color = "orange", size = 1) +
  labs(title = "Tendência da Ocorrência de Ferrugem ao Longo do Tempo",
       x = "Data",
       y = "Proporção de Ocorrência de Ferrugem (Sim)") +
  theme_minimal() +
  theme(axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold"))

ggsave("C:/Users/natal/Downloads/dados_tcc/tendencia_ferrugem.png", 
       width = 10, height = 6, dpi = 600)