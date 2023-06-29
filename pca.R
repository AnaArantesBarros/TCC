
#1.Carregando bibliotecas e organizando área de tarbalho
library(caret)
library(glmnet)
library(FactoMineR)
library(factoextra)


dados <- read.csv("tabelaNAfix2015_2016 - Area6.csv")

####Organizando dados de referência
##A predição que nos interessa será classificada como 1, as demais como 0.
dados$Classif[dados$Classif == "CANA"]  <- 1
dados$Classif[dados$Classif == "OUTROS"] <- 0
dados$Classif[dados$Classif == "URBANA"]  <- 0
dados$Classif <- factor(dados$Classif)

##Selecionando as colunas que serão utilizadas
#Neste caso os indices em data iniciam na 18 e se repetem a cada 10 até a coluna 185
colunas <- c(
  seq(from = 16, to = 185, by = 10),
  seq(from = 17, to = 185, by = 10),
  seq(from = 18, to = 185, by = 10),
  seq(from = 19, to = 185, by = 10),
  seq(from = 24, to = 185, by = 10),
  seq(from = 25, to = 185, by = 10))
dados1 <- dados[ ,c(colunas)]

#Calcula a matriz de correlação
R <- cor(dados1, method="pearson")

#Realiza o método de componentes principais
pca <- PCA(R, graph=TRUE, ncp=12)
pca

#Verifica os autovalores
autovalores <- get_eigenvalue(pca)
autovalores

#Scree plot - Eigenvalues
fviz_eig(pca, addlabels=TRUE)

#Verifica as coordenadas das variáveis
variaveis <- get_pca_var(pca)
variaveis$coord
variaveis$cor

#Calcula os valores das novas variáveis para cada observação
dados2 <- data.frame(as.matrix(dados1) %*% variaveis$coord)
dados2$Classif <- dados$Classif

#Ajusta o modelo logístico usando os componentes principais
m1 <- glm(Classif~., family=binomial(link="logit"), data=dados2)
summary(m1)

#Matriz de confusão
library(caret)
probs <- predict(m1, newdata=dados2, type="response")
dados2$pdata <- as.factor(ifelse(probs>0.5,"1","0"))
confusionMatrix(dados2$pdata, dados2$Classif, positive="1")

