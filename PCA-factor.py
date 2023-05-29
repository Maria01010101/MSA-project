import streamlit as st
from st_pages import show_pages_from_config, add_page_title,Page, Section, show_pages
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.markdown('''
we use PCA to rate the securities. The specific methods are introduced below.

In this method, we consider $p$ indexes of $n$ securities. That is,''')

st.latex(r'''
X=\left[\begin{array}{cccc}
x_{11} & x_{12} & \cdots & x_{1 p} \\
x_{21} & x_{22} & \cdots & x_{2 p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n 1} & x_{n 2} & \cdots & x_{n p}
\end{array}\right]''')

st.markdown('''Firstly, we conduct principal component analysis. Suppose
$\lambda_{1} \geq \lambda_{2} \geq \cdots \lambda_{p} \geq 0$ are $p$ eigenvalues of the correlation matrix. And the corresponding eigenvectors is given by
''')
st.latex(r'''T=\left(\mathbf{t}_{1}, \mathbf{t}_{2}, \cdots, \mathbf{t}_{p}\right)''')


st.markdown('''Then, the contribution rate of $i$-th principal component is ''')
st.latex(r''' \gamma_{i}=\frac{\lambda_{i}}{\sum_{j=1}^{p} \lambda_{j}}, \quad \forall i=1, \cdots, p''')



st.markdown('''The accumulative contribution rate of the first $m$ principal components is given by ''')
st.latex(r'''\Gamma_{m}=\sum_{i=1}^{m} \gamma_{i}=\frac{\sum_{i=1}^{m} \lambda_{i}}{\sum_{i=1}^{p} \lambda_{i}} ''')




st.markdown('''We choose the minimum $m$, which make the accumulative rate of construction reach $85 \%$.

Then, we obtain the principal components ''')
st.latex(r''' 
\left(y_{i 1}, \cdots, y_{i m}\right)^{\top}=\left(\mathbf{t}_{1}^{\top}, \cdots, \mathbf{t}_{m}^{\top}\right)\left(x_{i 1}, \cdots, x_{i p}\right)^{\top}, \quad \forall i=1, \cdots, n
''')


st.markdown('''The correlation coefficient between $x_{i}$ and principal component $y_{j}$ is given by ''')
st.latex(r''' \rho\left(x_{i}, y_{j}\right)=\sqrt{\lambda_{j}} t_{i j}, \quad \forall i=1, \cdots, p, \quad j=1, \cdots, m .''')




st.markdown('''Rate of contribution of $y_{1}, \cdots, y_{m}$ to $x_{i}$ is ''')
st.latex(r''' \rho_{i .1, \cdots, m}^{2}=\sum_{j=1}^{m} \rho^{2}\left(x_{i}, y_{j}\right)=\sum_{j=1}^{m} \lambda_{j} t_{i j}^{2} .''')

st.markdown('''
The estimate of factor loading matrix is as follows: ''')
st.latex(r'''A=\left(\sqrt{\lambda_{1}} \mathbf{t}_{1}, \sqrt{\lambda_{2}} \mathbf{t}_{2}, \cdots, \sqrt{\lambda_{m}} \mathbf{t}_{m}\right) . ''')


st.markdown(''' 
The absolute of this matrix is defined as''')
st.latex(r'''\|A\|=\left(\begin{array}{cccc}
\sqrt{\lambda_{1}}\left|t_{11}\right| & \sqrt{\lambda_{2}}\left|t_{12}\right| & \cdots & \sqrt{\lambda_{m}}\left|t_{1 m}\right| \\
\sqrt{\lambda_{1}}\left|t_{21}\right| & \sqrt{\lambda_{2}}\left|t_{22}\right| & \cdots & \sqrt{\lambda_{m}}\left|t_{2 m}\right| \\
\vdots & \vdots & \ddots & \vdots \\
\sqrt{\lambda_{1}}\left|t_{p 1}\right| & \sqrt{\lambda_{2}}\left|t_{p 2}\right| & \cdots & \sqrt{\lambda_{m}}\left|t_{p m}\right|
\end{array}\right) . ''')



st.markdown('''
We can define the weight of each indexes as follows:
 ''')
st.latex(r'''\omega_{i j}=\frac{\sqrt{\lambda_{j}}\left|t_{i j}\right|}{\sum_{i=1}^{p} \sqrt{\lambda_{j}}\left|t_{i j}\right|}, \quad \forall i=1, \cdots, p, \quad j=1, \cdots, m .
 ''')



st.markdown(''' 
By taking the arithmetic average of the weights, we final obtain the weight of each indexes:''')
st.latex(r'''\omega_{i}=\frac{\sum_{j=1}^{m} \omega_{i j}}{m}, \quad \forall i=1, \cdots, p ''')


st.markdown(''' Composite scores can be calculated via above weights.''')
st.latex(r'''\text { Score }=\sum_{i=1}^{p} \omega_{i} x_{i} ''')


st.latex(r''' 
\begin{equation}
\begin{aligned}
&\text { 表 1: } \mathrm{PCA} \text { 评级准则 }\\
&\begin{array}{ll}
\hline \text { Level } & \text { Score } \\
\hline \text { AAA } & {[7,8]} \\
\text { AA } & {[6,7)} \\
\text { A } & {[5,6)} \\
\text { BBB } & {[4,5)} \\
\text { BB } & {[3,4)} \\
\text { B } & {[2,3)} \\
\text { CCC } & {[0,2)} \\
\hline
\end{array}
\end{aligned}
\end{equation}''')

st.latex(r'''\begin{equation}
\begin{aligned}
&\text { 表 2: Rate of contribution of principal component to variable }\\
&\begin{array}{lcrr}
\hline & \begin{array}{r}
\text { Principal } \\
\text { Component 1 }
\end{array} & \begin{array}{r}
\text { Principal } \\
\text { Component 2 }
\end{array} & \begin{array}{r}
\text { Principal } \\
\text { Component 3 }
\end{array} \\
\hline \begin{array}{l}
\text { 管理实践得分 } \\
\end{array} & 0.9594873 & 0.0017198 & 0.0006090 \\
\begin{array}{l}
\text { 争议事件得分 } \\
\end{array} & 0.0220218 & 0.7844548 & 0.1746905 \\
\begin{array}{l}
\text { 环境维度得分 } \\
\end{array} & 0.4670031 & 0.1100475 & 0.0463217 \\
\begin{array}{l}
\text { 社会维度得分 } \\
\end{array} & 0.4670031 & 0.0097641 & 0.0776353\\
\begin{array}{l}
\text { 治理维度得分 }  \\
\end{array} & 0.3279950 & 0.1429771 & 0.5185134 \\
\hline
\end{array}
\end{aligned}
\end{equation}''')


st.latex(r'''\begin{equation}
\begin{aligned}
&\text { 表 3: 深股 } \mathrm{ESG} \text { 评级与 } \mathrm{PCA} \text { 评级对比 }\\
&\begin{array}{llrrl}
\hline \text { 证券简称 } & \text { Wind ESG 评级 } & \text { Wind ESG 综合得分 } & \text { PCA.score } & \text { PCA.level } \\
\hline \text { 平安银行 } & \mathrm{BBB} & 6.18 & 4.580214 & \mathrm{BBB} \\
\text { 金风科技 } & \mathrm{A} & 7.64 & 5.747949 & \mathrm{~A} \\
\text { 万科 A } & \mathrm{A} & 7.95 & 5.970752 & \mathrm{~A} \\
\text { 华测检测 } & \mathrm{AAA} & 9.26 & 7.004842 & \mathrm{AAA} \\
\text { 元隆雅图 } & \mathrm{BB} & 5.22 & 3.135031 & \mathrm{BB} \\
\text { 天齐锂业 } & \mathrm{BB} & 5.69 & 3.807570 & \mathrm{BB} \\
\hline
\end{array}
\end{aligned}
\end{equation}''')


image = Image.open('c://Users//16539//Documents//SUSTech//COURSES//2023-Spring//多元//input//pca.png')
st.image(image, caption='pca')

st.latex(r'''\begin{equation}
\begin{aligned}
&\text { 表 4: Rate of contribution of principal component to variable }\\
&\begin{array}{|c|c|c|c|}
\hline & \begin{array}{r}
\text { Principal } \\
\text { Component } 1
\end{array} & \begin{array}{r}
\text { Principal } \\
\text { Component } 2
\end{array} & \begin{array}{r}
\text { Principal } \\
\text { Component } 3
\end{array} \\
\hline \text { ESG 管理实践得分 } & 0.9681956 & 0.0053374 & 0.0005779 \\
\hline \text { ESG 争议事件得分 } & 0.0185002 & 0.9429033 & 0.0055561 \\
\hline \text { 环境维度得分 } & 0.5872668 & 0.0332900 & 0.0622426 \\
\hline \text { 社会维度得分 } & 0.6853516 & 0.0287742 & 0.1012809 \\
\hline \text { 治理维度得分 } & 0.3376956 & 0.0105121 & 0.6490494 \\
\hline
\end{array}
\end{aligned}
\end{equation}''')



# import rpy2
# import rpy2.robjects as robjects
# import rpy2.robjects.packages as rpackages
# # import R's utility package
# utils = rpackages.importr('utils')

# # select a mirror for R packages
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# # R package names
# packnames = ('ggplot2', 'readxl', 'tidyverse','knitr','openxlsx')

# # R vector of strings
# from rpy2.robjects.vectors import StrVector

# # Selectively install what needs to be install. We are fancy, just because we can.
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))

# print("download success!!")

# st.header('PCA & Factor Analysis')


# from rpy2.robjects.packages import importr
# ggplot2 = importr('ggplot2')
# readxl = importr('readxl')
# tidyverse = importr('tidyverse')
# openxlsx = importr('openxlsx')

# r = robjects.r


# SZ = r('''
#     SZ <- read.xlsx("c://Users//16539//Documents//SUSTech//COURSES//2023-Spring//多元//input//SZ.xlsx") %>% na.omit()
#     SZ.pr <- SZ[, 5:7] %>% as.data.frame()
# ''')

# st.markdown(SZ)

# st.markdown("AAAAAAAAAAAAAA")

# print("aaaaaaaaaaa")

# SH = r('''
#     SH <- openxlsx.read.xlsx("c://Users//16539//Documents//SUSTech//COURSES//2023-Spring//多元//input//SH.xlsx") %>% na.omit()
# ''')

# st.text(SH)


# library(readxl)
# library(tidyverse)
# # Data of Shenzhen stock
# SZ <- read_xlsx("SZ.xlsx") %>% na.omit()
# SZ.pr <- SZ[, 5:9] %>% as.data.frame()
# colnames(SZ.pr) <- colnames(SZ)[5:9]
# # Data of Shanghai stock
# SH <- read_xlsx("SH.xlsx") %>% na.omit()
# SH.pr <- SH[, 5:9] %>% as.data.frame()
# colnames(SH.pr) <- colnames(SH)[5:9]

# library(knitr)
# library(ggplot2)
# Contri.map <- function(contri, accontri) {
#   p <- length(contri)
#   class1 <- rep("Contribution", p)
#   class2 <- rep("Accumulative Contribution", p)
#   class <- c(class1, class2)
#   df <- data.frame(x = c(1:p, 1:p), 
#                    y = c(100 * contri, 100 * accontri), 
#                    class = class)
#   p <- ggplot(data = df, mapping = aes(x = x, y = y, color = class)) + 
#     geom_line() + 
#     geom_point() + 
#     xlab("Eigenvalue Number") + 
#     ylab("Percent (%)") + 
#     scale_color_discrete(name = "Annotation:") + 
#     theme(legend.position = c(.85, .5), 
#           legend.background = element_blank(), 
#           legend.key = element_blank())
#   return(p)
# }
# PCA <- function(data) {
#   p <- ncol(data)
#   n <- nrow(data)
#   R <- cor(data)
#   eigen.val <- eigen(R)$values
#   eigen.vec <- eigen(R)$vectors
#   contri <- eigen.val / sum(eigen.val)
#   accontri <- cumsum(eigen.val) / sum(eigen.val)
#   pic <- Contri.map(contri, accontri)
#   target.p <- 0
#   for (i in 1:p) {
#     if (accontri[i] > 0.85) {
#       target.p <- i
#       break
#     }
#   }
#   rho2 <- sweep(eigen.vec^2, 2, eigen.val, "*")
#   colnames(rho2) <- paste("Principal Component", 1:p, sep = " ")
#   rownames(rho2) <- colnames(data)
#   table <- as.data.frame(rho2[, 1:target.p]) %>% 
#     kable(caption = "Rate of contribution of principal component to variable")
#   loadings <- sweep(eigen.vec, 2, sqrt(eigen.val), "*")
#   absload <- loadings[, 1:target.p] %>% abs()
#   loadsum <- apply(absload, 2, sum)
#   abscom <- rbind(absload, loadsum)
#   w <- matrix(nrow = p, ncol = target.p)
#   for (i in 1:p) {
#     for (j in 1:target.p) {
#       w[i, j] <- abscom[i, j] / abscom[(p+1), j]
#     }
#   }
#   weight <- numeric(length = p)
#   for (i in 1:p) {
#     weight[i] <- mean(w[i, ])
#   }
#   score <- rep(0, n)
#   for (i in 1:p) {
#     score <- score + weight[i] * data[, i]
#   }
#   PCA_level <- vector(length = n)
#   for (i in 1:n) {
#     if (score[i] < 2) {
#       PCA_level[i] <- "CCC"
#     } else if (score[i] < 3) {
#       PCA_level[i] <- "B"
#     } else if (score[i] < 4) {
#       PCA_level[i] <- "BB"
#     } else if (score[i] < 5) {
#       PCA_level[i] <- "BBB"
#     } else if (score[i] < 6) {
#       PCA_level[i] <- "A"
#     } else if (score[i] < 7) {
#       PCA_level[i] <- "AA"
#     } else if (score[i] < 8) {
#       PCA_level[i] <- "AAA"
#     }
#   }
#   score.level <- data.frame(PCA.score = score, 
#                             PCA.level = PCA_level)
#   result <- list(Contri.pic = pic, 
#                  Effect.pc = table, 
#                  PCA.level = score.level)
#   return(result)
# }

# # Data of Shenzhen stock
# SZ.PCA <- PCA(SZ.pr)
# SZ.PCA$Contri.pic
# SZ.PCA$Effect.pc
# SZ <- cbind(SZ, SZ.PCA$PCA.level)
# head(SZ[, c(2:4, 12:13)]) %>% kable(caption = "深股ESG评级与PCA评级对比")
# # Data of Shanghai stock
# SH.PCA <- PCA(SH.pr)
# SH.PCA$Contri.pic
# SH.PCA$Effect.pc
# SH <- cbind(SH, SH.PCA$PCA.level)
# head(SH[, c(2:4, 12:13)]) %>% kable(caption = "上股ESG评级与PCA评级对比")

# Evaluate <- function(data) {
#   num <- 0
#   n <- nrow(data)
#   for (i in 1:n) {
#     if (data[i, 3] == data[i, 13]) {
#       num <- num + 1
#     }
#   }
#   result <- num / n 
#   result <- round(result, 4)
#   result <- result * 100
#   result <- paste(result, "%", sep = "")
#   return(result)
# }

# # Data of Shenzhen stock
# Evaluate(SZ)
# # Data of Shanghai stock
# Evaluate(SH)

# Accept <- function(data) {
#   c <- FALSE
#   if (data[3] == "CCC") {
#     if (data[13] == "CCC" | data[13] == "B") {
#       c <- TRUE
#     }
#   } else if (data[3] == "B") {
#     if (data[13] == "CCC" | data[13] == "B" | data[13] == "BB") {
#       c <- TRUE
#     }
#   } else if (data[3] == "BB") {
#     if (data[13] == "B" | data[13] == "BB" | data[13] == "BBB") {
#       c <- TRUE
#     }
#   } else if (data[3] == "BBB") {
#     if (data[13] == "BB" | data[13] == "BBB" | data[13] == "A") {
#       c <- TRUE
#     }
#   } else if (data[3] == "A") {
#     if (data[13] == "BBB" | data[13] == "A" | data[13] == "AA") {
#       c <- TRUE
#     }
#   } else if (data[3] == "AA") {
#     if (data[13] == "A" | data[13] == "AA" | data[13] == "AAA") {
#       c <- TRUE
#     }
#   } else if (data[3] == "AAA") {
#     if (data[13] == "AA" | data[13] == "AAA") {
#       c <- TRUE
#     }
#   }
#   return(c)
# }
# Accept.Evaluate <- function(data) {
#   n <- nrow(data)
#   num <- vector(length = n)
#   for (i in 1:n) {
#     num[i] <- Accept(data[i, ])
#   }
#   result <- sum(num) / n 
#   result <- round(result, 4)
#   result <- result * 100
#   result <- paste(result, "%", sep = "")
#   return(result)
# }

# # Data of Shenzhen stock
# Accept.Evaluate(SZ)
# # Data of Shanghai stock
# Accept.Evaluate(SH)