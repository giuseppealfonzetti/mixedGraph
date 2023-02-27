library(tidyverse)


#### set-up nodes ####
nodes_type <- c(rep(1, 5), rep(4,5))
nodes_type <- sort(nodes_type)
cumsum_nodes_type <- cumsum(nodes_type)
n_nodes <- length(nodes_type)
p <- sum(nodes_type==1)
q <- length(nodes_type)-p
r <- sum(nodes_type)

dim_edges_vec <- r^2-r*(r-1)/2 - sum(sapply(nodes_type, function(x) if_else(x>1, x*(x-1)/2, 0)))
#### set up graph structure ####
#### alphas (continuous): enters in conditional mean of continuous sub-graph
alphas <- rep(0, p)
#### betas (continuous-continuous): inverse covariance matrix of continuous conditional sub-graph

#### automatic
EMat <- matrix(0, r, r)

#continuous
for (j in 1:p) {
    EMat <- setUp_edge(NODE_I = j, NODE_J = j, VAL = 1, EMat)
}
#continuous - continuous
for (j in 1:p) {
    for (i in j:p) {
        if(abs(j-i)==1){
            EMat <- setUp_edge(NODE_I = i, NODE_J = j, VAL = .25, EMat)
        }
    }
}

#continuous - categorical
for (j in 1:p) {
    for (i in (p+1):(p+q)) {
        if(abs(j-i)==p){
            EMat <- setUp_edge(NODE_I = i, NODE_J = j, VAL = rep(c(.25,-.25),2), EMat)
            # EMat <- setUp_edge(NODE_I = i, NODE_J = j, VAL = rep(c(.25,-.25),1), EMat)

        }
    }
}

#categorical - categorical
for (j in (p+1):(p+q)) {
    for (i in j:(p+q)) {
        if(abs(j-i)==1){
            EMat <- setUp_edge(NODE_I = i, NODE_J = j, VAL = rep(c(1,-1,1,-1,       -1,1,-1,1),2), EMat)
            # EMat <- setUp_edge(NODE_I = i, NODE_J = j, VAL = rep(c(.5,-.5,-.5,.5),1), EMat)

        }
    }
}

Matrix::Matrix(EMat, sparse = T)

edgeMat <- EMat

Matrix::Matrix(edgeMat, sparse = T)

#### Sampling data ####
maxiter <- 10000; m <- 5; skip <- 0

set.seed(1)
graphGibbs <- graphBlocksGibbsSampler(
    edgesPar = edgeMat,
    nodesPar = alphas,
    nodes_type = nodes_type,
    warmup = 10000,
    maxiter = maxiter,
    m = m,
    skip = skip,
    verboseFLAG = F,
    store_warmupFLAG = F
)
n <- 1000
data <- as_tibble(reduce(graphGibbs$chains, rbind)[sample(1:nrow(graphGibbs$chains[[1]]), n, replace = F),])

colnames(data) <- c(
    sapply(1:p, function(x) paste0('x', x)),
    sapply(1:q, function(x) paste0('y', x))
)
data


#### parameter vector ####
# constraint matrix
constrMat <- matrix(0, r, r)
for (node_j in 1:n_nodes) {
    for (node_i in node_j:n_nodes) {
        constrMat <- setUp_edge(node_i, node_j, VAL = 1, constrMat)

    }
}

# true parameter vector
true_theta <- alphas
for (j in 1:r) {
    for (i in j:r) {
        if(constrMat[i,j] == 1){
            true_theta <- c(true_theta, edgeMat[i,j])
        }
    }
}

# initial parameter vector
contCov_init <- matrix(0, p, p); diag(contCov_init) <- 1
edgeMat_init <- theta_to_edgesPar(c(alphas, rep(0, dim_edges_vec)), nodes_type, cumsum_nodes_type, p, r, n_nodes)
invContCov_init <- solve(contCov_init); invContCov_init[upper.tri(invContCov_init)] <- 0
edgeMat_init[1:p, 1:p] <- invContCov_init

# true parameter vector
theta_init <- rep(0, p)
for (j in 1:r) {
    for (i in j:r) {
        if(constrMat[i,j] == 1){
            theta_init <- c(theta_init, edgeMat_init[i,j])
        }
    }
}

#########
graphR_obj <- function(theta){
    graph <- graph_cl(
        data = as.matrix(data),
        theta = theta,
        nodes_type
    )

    -graph$cl
}
graphR_gr <- function(theta){
    graph <- graph_cl(
        data = as.matrix(data),
        theta = theta,
        nodes_type
    )

    -graph$gradient
}
##########
library(tidyverse)
sample_size <- c(500)
nsim <- 2
sim_setting <- expand_grid(n = sample_size, sim_id = 1:nsim) %>%
    mutate(id = row_number()) %>%
    select(id, everything()) %>%
    mutate(
        data = map(n, ~reduce(graphGibbs$chains, rbind)[sample(1:nrow(graphGibbs$chains[[1]]), .x, replace = F),])
    )


#save(sim_setting, file = 'sim_setting.RData')
cores <- 2
tol <- 1e-5; step <- 5e-1; counter <- 15; maxiter <- 10*(p+q)
## deterministic ###
data <- sim_setting$data[[1]]
set.seed(123)
nu <- round(sample_size*(p+q)*.5, 0)
nu <- round(sample_size*.1,0)
mod <- mixedGraph(
    DATA = as.matrix(data),
    SDS = fun_sd(data, P=p, Q=q), #rep(1, p+q),#
    THETA = theta_init,
    NODES_TYPE = nodes_type,
    MAXITER = maxiter,
    STEPSIZE = 1,#step,
    SAMPLING_SCHEME = 2,
    REG_PAR = 2.5*sqrt(log(p+q)/(nrow(data))),
    NU = nu,
    TOL = 1e-10,
    TOL_MINCOUNT = maxiter,
    VERBOSEFLAG = F,
    EPS = .75
)

summary(clock, units = 's')[summary(clock, units = 's')$ticker=='Main',]

# mod$path_theta_diff[1,]
# mod$path_theta[2,] - mod$path_theta[1,]
# mod$path_nll[mod$last_iter]
# mod$path_theta_check[1]
# mod$path_thetaDiff[1]/mod$path_thetaNorm[1]
# norm(as.matrix(mod$path_theta[2,] - mod$path_theta[1,]), type = 'F') / norm(as.matrix(mod$path_theta[1,]), type = 'F')

#-graphR_gr(mod$path_theta[10,]) ; mod$path_grad[10,]
#mod$path_theta
est_theta <- mod$path_theta[mod$last_iter+1,]
est_theta_path <- tibble(iter = 0:(nrow(mod$path_theta)-1), theta = as.list(data.frame(t(mod$path_theta))))
est_theta_path %>% pluck('theta')
est_theta_path <- est_theta_path %>%
    mutate(
        rates = map(theta, ~tidy_recovery(
            edgeMat,
            theta_to_edgesPar(.x, nodes_type, cumsum_nodes_type, p, r, n_nodes),
            P = p, Q = q)$rates)
    ) %>%
    unnest(rates)
Matrix::Matrix(theta_to_edgesPar(est_theta, nodes_type, cumsum_nodes_type, p, r, n_nodes), sparse = T)
Matrix::Matrix(edgeMat, sparse = T)
Matrix::Matrix(edgeMat_init, sparse = T)




ggTol <- tibble(iter = 1:(mod$last_iter+1)) %>%
    mutate(diff = map_dbl(iter, ~mod$path_theta_check[.x])
    ) %>%
    ggplot(aes(x =iter, y = diff)) +
    geom_line()+
    geom_hline(yintercept = 1e-3, linetype = 'dashed') +
    theme_light()
plotly::ggplotly(ggTol)

# ggTol <- tibble(iter = 1:mod$last_iter) %>%
#   mutate(diff = map_dbl(iter, ~mod$path_thetaDiff[.x])

#   ) %>%
#   ggplot(aes(x =iter, y = diff)) +
#   geom_line()+
#   geom_hline(yintercept = 1e-3, linetype = 'dashed') +
#   theme_light()
# plotly::ggplotly(ggTol)

ggMSE <- tibble(iter = 1:(mod$last_iter+1)) %>%
    mutate(mse = map_dbl(iter, ~mean((mod$path_theta[.x,]-true_theta)^2))) %>%
    ggplot(aes(x =iter, y = mse)) +
    geom_line()+
    theme_light()
plotly::ggplotly(ggMSE)

rR <- recoveryRate(edgeMat, est_theta, T)
rR

rR_list <- lapply(0:mod$last_iter, function(x) recoveryRate(edgeMat, mod$path_theta[x+1,]))
rR_tib <- tibble(iter = 0:mod$last_iter) %>%
    mutate(
        rates = rR_list
    ) %>%
    mutate(
        TPR = map_dbl(rates, ~.x[1]),
        TNR = map_dbl(rates, ~.x[2]),
        TTR = map_dbl(rates, ~.x[3])
    ) %>%
    gather(key = 'recovery_type', value = 'share', TPR, TNR, TTR ) %>%
    select(iter, recovery_type, share)

gg_rR <- rR_tib %>%
    ggplot(aes(x = iter, y = share, col = recovery_type ))+
    geom_line()+
    theme_light()
plotly::ggplotly(gg_rR, dynamicTicks = T)


tmat <- edgeMat
tmat[upper.tri(tmat)] <- NaN
diag(tmat) <- NaN
emat <- theta_to_edgesPar(est_theta, nodes_type, cumsum_nodes_type, p, r, n_nodes)
emat[upper.tri(emat)] <- NaN
diag(emat) <- NaN
as_tibble(tmat) %>%
    mutate(row_lab=paste0('V',1:r)) %>%
    gather(key = 'col_lab', value = 'val', starts_with('V')) %>%
    mutate(
        row_lab = factor(row_lab, levels = paste0('V',r:1), labels = paste0('V',r:1), ordered = T),
        col_lab = factor(col_lab, levels = paste0('V',1:r), labels = paste0('V',1:r), ordered = T)
    ) %>%
    ggplot(aes(x = col_lab, y = row_lab)) +
    geom_tile(aes(fill=val))+
    scale_fill_viridis_c()

as_tibble(emat) %>%
    mutate(row_lab=paste0('V',1:r)) %>%
    gather(key = 'col_lab', value = 'val', starts_with('V')) %>%
    mutate(mat_lab = 'est') %>%
    bind_rows(
        as_tibble(tmat) %>%
            mutate(row_lab=paste0('V',1:r)) %>%
            gather(key = 'col_lab', value = 'val', starts_with('V')) %>%
            mutate(mat_lab = 'true')
    )%>%
    mutate(
        row_lab = factor(row_lab, levels = paste0('V',r:1), labels = paste0('V',r:1), ordered = T),
        col_lab = factor(col_lab, levels = paste0('V',1:r), labels = paste0('V',1:r), ordered = T)    ) %>%
    ggplot(aes(x = col_lab, y = row_lab)) +
    geom_tile(aes(fill=sign(val)))+
    facet_wrap(vars(mat_lab))+
    scale_fill_gradient2()

sign(-3)


fun_sd(data, P=p, Q=q)
vec_sd <- c()
for (node in 1:p) {
    vec_sd[node] <- sd(as.matrix(data[, node]))
}

for (node in (p+1):(p+q)) {
    prob <- as.numeric(table(as.matrix(data[, node]))/length(as.matrix(data[, node])))
    var <- prob * (1-prob)
    vec_sd[node] <- sqrt(sum(var))
}
return(vec_sd)

edgeMat
check_edge(22,23, edgeMat)
edgetib <- expand_grid(row_node = 1:(p+q), col_node = 1:(p+q)) %>%
    filter(col_node < row_node) %>%
    mutate(
        true_edge = map2_lgl(row_node, col_node, ~check_edge(.x,.y, edgeMat)),
        est_edge = map2_lgl(row_node, col_node, ~check_edge(.x,.y, emat))
        )

edgetib %>%
    filter(true_edge==T) %>%
    mutate(tp = (true_edge == est_edge)) %>%
    pluck('tp') %>% mean()

edgetib %>%
    filter(true_edge==F) %>%
    mutate(tn = (true_edge == est_edge)) %>%
    pluck('tn') %>% mean()

edgetib %>%
    mutate(tt = (true_edge == est_edge)) %>%
    pluck('tt') %>% mean()

tibble(
    tp = edgetib %>%
        filter(true_edge==T) %>%
        mutate(tp = (true_edge == est_edge)) %>%
        pluck('tp') %>% mean(),
    tn = edgetib %>%
        filter(true_edge==F) %>%
        mutate(tn = (true_edge == est_edge)) %>%
        pluck('tn') %>% mean(),
    tt = edgetib %>%
        mutate(tt = (true_edge == est_edge)) %>%
        pluck('tt') %>% mean()
)

tidy_recovery(edgeMat, emat, P=p, Q=q)



a <- matrix(1:10, 5,2)
a
list(a)
as.list(data.frame(t(a)))
tibble(iter = 0:(nrow(a)-1), theta = as.list(data.frame(t(a))))
