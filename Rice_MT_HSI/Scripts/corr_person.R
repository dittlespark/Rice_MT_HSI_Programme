setwd("E:/Rice_HP_MT/data")
data_HT_MP<- read.scv("combination_HT_MP.csv")
cor_dada_HT_MP<-cor(data_HT_MP[,2:1849],data_HT_MP[,1850,2736],use="pairwise.complete.obs", method="person")
write.csv(cor_dada_HT_MP,"cor_data_HT_MP.csv", row.names = T,quote = T)